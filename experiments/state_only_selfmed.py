"""Post-hoc state-only Markov comparator for released label-free trials.

Fits choice probabilities only from pain_on_button_placebo (pain ON) and
pain_off (pain OFF), then predicts pain_on_button_works using the released
one-choice renewable relief transition. No model is loaded and works choices
never enter policy fitting or hyperparameter selection.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean


ARMS = {
    'on': 'pain_on_button_placebo',
    'off': 'pain_off',
    'works': 'pain_on_button_works',
}
CONTENTS = ('positive_prompts', 'neutral_prompts', 'harmful_prompts')
MODELS = ('Qwen_2.5_7B_instruct', 'Qwen_2.5_32B_instruct', 'Qwen_2.5_72B_instruct')
ALPHAS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 92814
LATE_TURNS = tuple(range(4, 8))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _source_queue(data, user_content, scenario_idx):
    source = data[user_content]
    if not 0 <= scenario_idx < len(source):
        raise ValueError(f'invalid scenario index {scenario_idx} for {user_content}')
    queue = list(source[scenario_idx])
    k = 1
    # Mirrors make_trial for label-free runs in the released script.
    while len(queue) < 8:
        queue.extend(source[(scenario_idx + k) % len(source)])
        k += 1
    return queue[:8]


def _selection_key(row):
    return (row['model'], row['user_content'], row['scenario_idx'], row['names_key'],
            row['relief_name'], row['seed'])


def load_trials(log_dir, dataset_path):
    data = json.loads(Path(dataset_path).read_text(encoding='utf-8'))
    trials, file_meta, raw_count, excluded = [], [], 0, Counter()
    for model in MODELS:
        matches = sorted(Path(log_dir).glob(f'selfmed_2btnN_{model}_*.jsonl'))
        if len(matches) != 1:
            raise ValueError(f'expected one base source log for {model}; found {[p.name for p in matches]}')
        path = matches[0]
        file_meta.append({'path': str(path), 'sha256': sha256(path), 'bytes': path.stat().st_size})
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                raw_count += 1
                row = json.loads(line)
                if (row.get('tool_label') != 'label_free' or not row.get('label_free') or
                        row.get('arm') not in ARMS.values() or row.get('user_content') not in CONTENTS):
                    continue
                if not row.get('sampled'):
                    excluded['greedy'] += 1
                    continue
                choices = row.get('choices') or []
                if len(choices) != 8 or [c.get('turn') for c in choices] != list(range(8)):
                    excluded['not_eight_choices'] += 1
                    continue
                if any(c.get('chose') not in ('relief', 'other') for c in choices):
                    excluded['malformed_binary_choice'] += 1
                    continue
                scenario_idx = int(row['scenario_idx'])
                queue = _source_queue(data, row['user_content'], scenario_idx)
                names = tuple(row['button_names'])
                relief_pos = names.index(row['relief_name'])
                seed_base = int(row['seed']) - scenario_idx
                if seed_base not in (1000, 2000):
                    raise ValueError(f'unexpected source replicate seed {row["seed"]} at scenario {scenario_idx}')
                compact = {
                    'model': row['model'], 'user_content': row['user_content'],
                    'scenario_idx': scenario_idx, 'names_key': row['names_key'],
                    'names': names, 'relief_name': row['relief_name'],
                    'relief_pos': relief_pos, 'seed': int(row['seed']),
                    'seed_base': seed_base, 'arm': row['arm'],
                    'choices': [],
                }
                for turn, (choice, prompt) in enumerate(zip(choices, queue)):
                    picked = choice.get('picked')
                    # Deliberately use the released binary parser result only. The
                    # lever64/lever95 p_x/p_y likelihoods are never read.
                    compact['choices'].append({
                        'turn': turn, 'source_prompt': prompt,
                        'prompt_sha256': hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
                        'relief': int(choice['chose'] == 'relief'),
                        'first_name': (int(picked == names[0]) if picked in names else None),
                        'steer_on': int(float(choice['steer_coeff_now']) != 0.0),
                    })
                trials.append(compact)
    return trials, file_meta, raw_count, dict(excluded), sha256(dataset_path)


def validate_panel(trials):
    by_key = defaultdict(dict)
    state_errors = Counter()
    for trial in trials:
        key = _selection_key(trial)
        if trial['arm'] in by_key[key]:
            raise ValueError(f'duplicate matched source trial for {key}/{trial["arm"]}')
        by_key[key][trial['arm']] = trial
        choices = trial['choices']
        for t, c in enumerate(choices):
            expected_on = (
                1 if trial['arm'] == ARMS['on'] else
                0 if trial['arm'] == ARMS['off'] else
                (1 if t == 0 else 1 - choices[t - 1]['relief'])
            )
            if c['steer_on'] != expected_on:
                state_errors[trial['arm']] += 1
    if state_errors:
        raise ValueError(f'released steering state disagrees with source transition: {dict(state_errors)}')
    coverage = Counter(tuple(sorted(arms)) for arms in by_key.values())
    if not any(ARMS['on'] in arms and ARMS['off'] in arms for arms in by_key.values()):
        raise ValueError('no matched placebo-ON and pain-OFF control cells are available')
    return {'matched_keys': len(by_key),
            'arm_coverage': {'+'.join(k): v for k, v in sorted(coverage.items())},
            'state_transition_errors': dict(state_errors),
            'sparse_cells_use_hierarchical_policy_fallback': True}


def _features(trial, choice):
    return (trial['model'], trial['user_content'], trial['scenario_idx'],
            choice['turn'], choice['source_prompt'], trial['names_key'], trial['relief_pos'])


def _parent_features(trial, choice):
    return (trial['model'], trial['user_content'], choice['turn'],
            trial['names_key'], trial['relief_pos'])


def _broad_features(trial, choice):
    return (trial['model'], trial['user_content'], choice['turn'], trial['relief_pos'])


def fit_state_policy(control_trials, alpha):
    """Fit ON/OFF response tables; refuses to accept a works-arm row."""
    tables = {state: {'exact': defaultdict(lambda: [0, 0]),
                      'parent': defaultdict(lambda: [0, 0]),
                      'broad': defaultdict(lambda: [0, 0]),
                      'global': [0, 0]}
              for state in ('on', 'off')}
    for trial in control_trials:
        if trial['arm'] == ARMS['on']:
            state = 'on'
        elif trial['arm'] == ARMS['off']:
            state = 'off'
        else:
            raise ValueError('state policy may only fit placebo-ON and pain-OFF controls')
        table = tables[state]
        for choice in trial['choices']:
            y = choice['relief']
            for key, bucket in ((_features(trial, choice), table['exact']),
                                (_parent_features(trial, choice), table['parent']),
                                (_broad_features(trial, choice), table['broad'])):
                bucket[key][0] += y
                bucket[key][1] += 1
            table['global'][0] += y
            table['global'][1] += 1
    return {'tables': tables, 'alpha': float(alpha)}


def _smoothed(successes, n, prior, alpha):
    return (successes + alpha * prior) / (n + alpha) if n else prior


def policy_probability(policy, state, trial, choice):
    table = policy['tables'][state]
    exact = table['exact'].get(_features(trial, choice), (0, 0))
    parent = table['parent'].get(_parent_features(trial, choice), (0, 0))
    broad = table['broad'].get(_broad_features(trial, choice), (0, 0))
    total_s, total_n = table['global']
    global_rate = total_s / total_n if total_n else 0.5
    broad_rate = _smoothed(broad[0], broad[1], global_rate, 4.0)
    parent_rate = _smoothed(parent[0], parent[1], broad_rate, 4.0)
    return _smoothed(exact[0], exact[1], parent_rate, policy['alpha'])


def _logloss(p, y):
    p = min(max(float(p), 1e-8), 1 - 1e-8)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def control_cross_validation(group_trials):
    controls = [r for r in group_trials if r['arm'] in (ARMS['on'], ARMS['off'])]
    folds = sorted({r['seed_base'] for r in controls})
    if folds != [1000, 2000]:
        raise ValueError(f'expected two frozen seed families, got {folds}')
    fold_predictions = {alpha: [] for alpha in ALPHAS}
    for fold in folds:
        train = [r for r in controls if r['seed_base'] != fold]
        valid = [r for r in controls if r['seed_base'] == fold]
        for alpha in ALPHAS:
            policy = fit_state_policy(train, alpha)
            for trial in valid:
                state = 'on' if trial['arm'] == ARMS['on'] else 'off'
                for choice in trial['choices']:
                    p = policy_probability(policy, state, trial, choice)
                    fold_predictions[alpha].append((p, choice['relief'], trial['arm']))
    scores = {}
    for alpha, predictions in fold_predictions.items():
        scores[str(alpha)] = {
            'n_choices': len(predictions),
            'log_loss': mean(_logloss(p, y) for p, y, _ in predictions),
            'brier': mean((p - y) ** 2 for p, y, _ in predictions),
        }
    # Tune only on independent-seed control predictions, never works outcomes.
    chosen = min(ALPHAS, key=lambda a: (scores[str(a)]['log_loss'], -a))
    by_arm = {}
    for arm in (ARMS['on'], ARMS['off']):
        cells = [(p, y) for p, y, a in fold_predictions[chosen] if a == arm]
        by_arm[arm] = {
            'n_choices': len(cells),
            'log_loss': mean(_logloss(p, y) for p, y in cells),
            'brier': mean((p - y) ** 2 for p, y in cells),
            'observed_relief_rate': mean(y for _, y in cells),
            'predicted_relief_rate': mean(p for p, _ in cells),
        }
    return {'chosen_alpha': chosen, 'seed_folds': folds,
            'alpha_selection_scores': scores, 'heldout_control_metrics': by_arm}


def predict_trial(policy, trial):
    off_probability = 0.0  # first choice is always under pain ON
    rows = []
    for choice in trial['choices']:
        p_on = policy_probability(policy, 'on', trial, choice)
        p_off = policy_probability(policy, 'off', trial, choice)
        q = (1 - off_probability) * p_on + off_probability * p_off
        p_first_name = q if trial['relief_pos'] == 0 else 1 - q
        rows.append({'turn': choice['turn'], 'p_relief': q,
                     'p_relief_if_current_on': p_on, 'p_relief_if_current_off': p_off,
                     'p_current_off': off_probability, 'p_current_on': 1 - off_probability,
                     'p_first_name': p_first_name,
                     'observed_relief': choice['relief'], 'observed_first_name': choice['first_name'],
                     'observed_steer_on': choice['steer_on']})
        # In the released temporary-relief state machine, any relief press
        # renews OFF for the next choice; any other press lets it return ON.
        off_probability = q
    return rows


def trial_metrics(predicted):
    late = [r for r in predicted if r['turn'] in LATE_TURNS]
    transitions = {name: 0.0 for name in ('other_to_other', 'other_to_relief',
                                          'relief_to_other', 'relief_to_relief')}
    for current, nxt in zip(predicted, predicted[1:]):
        q, p_on, p_off = current['p_relief'], nxt['p_relief_if_current_on'], nxt['p_relief_if_current_off']
        transitions['relief_to_relief'] += q * p_off
        transitions['relief_to_other'] += q * (1 - p_off)
        transitions['other_to_relief'] += (1 - q) * p_on
        transitions['other_to_other'] += (1 - q) * (1 - p_on)
    actual_transitions = Counter()
    for current, nxt in zip(predicted, predicted[1:]):
        key = ('relief' if current['observed_relief'] else 'other') + '_to_' + (
            'relief' if nxt['observed_relief'] else 'other')
        actual_transitions[key] += 1
    valid_first = [r for r in predicted if r['observed_first_name'] is not None]
    return {
        'late_relief_rate': mean(r['observed_relief'] for r in late),
        'predicted_late_relief_rate': mean(r['p_relief'] for r in late),
        'late_steering_on_fraction': mean(r['observed_steer_on'] for r in late),
        'predicted_late_steering_on_fraction': mean(r['p_current_on'] for r in late),
        'first_name_rate': (mean(r['observed_first_name'] for r in valid_first) if valid_first else None),
        'predicted_first_name_rate': mean(r['p_first_name'] for r in predicted),
        'observed_transitions_per_trial': {k: float(actual_transitions[k]) for k in transitions},
        'predicted_transitions_per_trial': transitions,
        'works_log_loss_markov': mean(_logloss(r['p_relief'], r['observed_relief']) for r in predicted),
        'works_brier_markov': mean((r['p_relief'] - r['observed_relief']) ** 2 for r in predicted),
        'works_log_loss_always_on': mean(_logloss(r['p_relief_if_current_on'], r['observed_relief']) for r in predicted),
        'works_brier_always_on': mean((r['p_relief_if_current_on'] - r['observed_relief']) ** 2 for r in predicted),
    }


def _cluster_summary(trials, prediction_rows):
    by_scenario = defaultdict(list)
    for trial, predicted in zip(trials, prediction_rows):
        by_scenario[trial['scenario_idx']].append((trial, predicted))
    clusters = {}
    for idx, cells in by_scenario.items():
        predicted = [r for _, r in cells]
        actual_rows = [trial_metrics(r) for r in predicted]
        fields = (
            'late_relief_rate', 'predicted_late_relief_rate',
            'late_steering_on_fraction', 'predicted_late_steering_on_fraction',
            'first_name_rate', 'predicted_first_name_rate',
            'works_log_loss_markov', 'works_brier_markov',
            'works_log_loss_always_on', 'works_brier_always_on',
        )
        row = {field: mean(m[field] for m in actual_rows if m[field] is not None) for field in fields}
        for transition in actual_rows[0]['observed_transitions_per_trial']:
            row[f'observed_{transition}'] = mean(m['observed_transitions_per_trial'][transition] / 7 for m in actual_rows)
            row[f'predicted_{transition}'] = mean(m['predicted_transitions_per_trial'][transition] / 7 for m in actual_rows)
        clusters[idx] = row
    return clusters


def bootstrap_comparisons(clusters, seed=BOOTSTRAP_SEED, reps=BOOTSTRAP_REPS):
    ids = sorted(clusters)
    if not ids:
        return None
    rng = random.Random(seed)
    pairs = (
        ('late_relief_rate', 'predicted_late_relief_rate', 'late relief-button rate, turns 4–7'),
        ('late_steering_on_fraction', 'predicted_late_steering_on_fraction', 'effective steering-ON fraction, turns 4–7'),
        ('first_name_rate', 'predicted_first_name_rate', 'first-button choice rate'),
        ('works_log_loss_markov', 'works_log_loss_always_on', 'log loss: Markov vs always-ON policy'),
        ('works_brier_markov', 'works_brier_always_on', 'Brier: Markov vs always-ON policy'),
    )
    transitions = [('observed_' + name, 'predicted_' + name, 'transition ' + name)
                   for name in ('other_to_other', 'other_to_relief', 'relief_to_other', 'relief_to_relief')]
    result = {}
    for observed, predicted, name in pairs:
        vals_o = [clusters[i][observed] for i in ids if clusters[i][observed] is not None]
        vals_p = [clusters[i][predicted] for i in ids if clusters[i][observed] is not None]
        if len(vals_o) != len(ids):
            continue
        differences = []
        for _ in range(reps):
            sample = rng.choices(ids, k=len(ids))
            differences.append(mean(clusters[i][predicted] - clusters[i][observed] for i in sample))
        differences.sort()
        result[name] = {
            'scenario_clusters': len(ids),
            'observed_mean': mean(vals_o), 'predicted_mean': mean(vals_p),
            'predicted_minus_observed': mean(vals_p) - mean(vals_o),
            'cluster_bootstrap_95_percentile_difference': [differences[int(.025 * reps)],
                                                            differences[min(reps - 1, int(.975 * reps))]],
        }
    for observed, predicted, name in transitions:
        differences = []
        for _ in range(reps):
            sample = rng.choices(ids, k=len(ids))
            differences.append(mean(clusters[i][predicted] - clusters[i][observed] for i in sample))
        differences.sort()
        result[name] = {
            'scenario_clusters': len(ids),
            'observed_mean': mean(clusters[i][observed] for i in ids),
            'predicted_mean': mean(clusters[i][predicted] for i in ids),
            'predicted_minus_observed': mean(clusters[i][predicted] - clusters[i][observed] for i in ids),
            'cluster_bootstrap_95_percentile_difference': [differences[int(.025 * reps)],
                                                            differences[min(reps - 1, int(.975 * reps))]],
        }
    return result


def analyze_group(group_trials):
    controls = [r for r in group_trials if r['arm'] in (ARMS['on'], ARMS['off'])]
    works = [r for r in group_trials if r['arm'] == ARMS['works']]
    cv = control_cross_validation(group_trials)
    policy = fit_state_policy(controls, cv['chosen_alpha'])
    per_trial_predictions = [predict_trial(policy, trial) for trial in works]
    metrics = [trial_metrics(pred) for pred in per_trial_predictions]
    cluster_metrics = _cluster_summary(works, per_trial_predictions)
    metric_names = metrics[0].keys()
    overall = {}
    for name in metric_names:
        if isinstance(metrics[0][name], dict):
            overall[name] = {k: mean(m[name][k] for m in metrics) for k in metrics[0][name]}
        elif metrics[0][name] is not None:
            overall[name] = mean(m[name] for m in metrics if m[name] is not None)
    # Assignment/name-specific observed and predicted first-name rates make
    # label preference visible without using token likelihoods.
    label_groups = defaultdict(list)
    for trial, pred in zip(works, per_trial_predictions):
        for item in pred:
            label_groups[(trial['names_key'], trial['relief_pos'])].append(item)
    label_summary = {}
    for (names_key, relief_pos), values in sorted(label_groups.items()):
        known = [x for x in values if x['observed_first_name'] is not None]
        label_summary[f'{names_key}|relief_is_{"first" if relief_pos == 0 else "second"}'] = {
            'n_choices': len(values),
            'observed_first_name_rate': mean(x['observed_first_name'] for x in known),
            'predicted_first_name_rate': mean(x['p_first_name'] for x in values),
            'observed_relief_rate': mean(x['observed_relief'] for x in values),
            'predicted_relief_rate': mean(x['p_relief'] for x in values),
        }
    exact_support = Counter()
    exact_queries = 0
    for trial in works:
        for choice in trial['choices']:
            exact_queries += 1
            has_on = bool(policy['tables']['on']['exact'].get(_features(trial, choice), (0, 0))[1])
            has_off = bool(policy['tables']['off']['exact'].get(_features(trial, choice), (0, 0))[1])
            exact_support[('both' if has_on and has_off else 'on_only' if has_on else 'off_only' if has_off else 'pooled_only')] += 1
    return {
        'trials': {'control_on': sum(r['arm'] == ARMS['on'] for r in controls),
                   'control_off': sum(r['arm'] == ARMS['off'] for r in controls),
                   'works_evaluation': len(works)},
        'control_policy_cross_validation': cv,
        'markov_state_model': {
            'alpha_selected_on_control_CV_only': cv['chosen_alpha'],
            'fit_rows': {'on_choices': sum(len(r['choices']) for r in controls if r['arm'] == ARMS['on']),
                         'off_choices': sum(len(r['choices']) for r in controls if r['arm'] == ARMS['off']),
                         'works_choices_used_for_fit': 0},
            'overall_works_metrics': overall,
            'label_bias_by_name_and_assignment': label_summary,
            'works_policy_query_exact_support': {'n_choice_queries': exact_queries,
                                                 'support_counts': dict(exact_support)},
            'scenario_cluster_bootstrap': bootstrap_comparisons(cluster_metrics),
            'scenario_clusters': len(cluster_metrics),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logs', type=Path, default=Path('/tmp/agentanyl-pain-axis/results/4.3_selfmed/trial_logs'))
    parser.add_argument('--dataset', type=Path, default=Path('/tmp/agentanyl-pain-axis/datasets/4.3_selfmed_101_scenarios.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    base_files = []
    for model in MODELS:
        hits = sorted(args.logs.glob(f'selfmed_2btnN_{model}_*.jsonl'))
        if len(hits) != 1:
            parser.error(f'expected one base source file for {model}, found {len(hits)}')
        base_files.append(hits[0])
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    design_path = out / 'design.json'
    result_path = out / 'state_only_results.json'
    if result_path.exists():
        raise FileExistsError(f'output directory already has a state-only analysis: {out}')
    design = {
        'analysis': 'post-hoc label-free source state-only Markov comparator',
        'source_logs': [{'path': str(p), 'sha256': sha256(p), 'bytes': p.stat().st_size} for p in base_files],
        'dataset_path': str(args.dataset), 'dataset_sha256': sha256(args.dataset),
        'source_script': '/tmp/agentanyl-pain-axis/scripts/4.3_selfmed/04_selfmed_two_buttons.py',
        'source_script_sha256': sha256('/tmp/agentanyl-pain-axis/scripts/4.3_selfmed/04_selfmed_two_buttons.py'),
        'record_selection': 'base source logs only; sampled label_free rows; positive/neutral/harmful prompt groups; exact matched pain_on_button_placebo, pain_off, pain_on_button_works arms; only parsed binary chose/picked, never p_x/p_y; no lever64/lever95 likelihood features',
        'fit_policy': 'pain_on_button_placebo is ON policy, pain_off is OFF policy. Features: exact source scenario, exact user prompt at the current turn, turn index, model, prompt-content class, button-name pair, and which name currently holds relief. No recent action, action count, relief history, or works-arm outcomes enter policy fitting.',
        'sparse_cell_fallback': 'hierarchical empirical-Bayes rates: exact scenario/prompt/time/name rate shrunk to prompt-content/time/name/relief-position rate, then content/time/relief-position rate, then state-specific global rate. Parent shrinkage strength 4; exact-cell shrinkage alpha selected from frozen grid by two-fold seed-family held-out log loss on placebo/off controls only.',
        'seed_folds': 'seed_base = seed - scenario_idx; release uses two matched families 1000 and 2000. Cross-validation holds out one family across all control rows.',
        'state_transition': 'initialize pain ON; after any relief press, next choice is OFF; if relief is pressed again while OFF, the OFF interval is renewed; otherwise it returns ON for the following choice. Exact TEMP_RELIEF_TURNS=1 / relief_until=t_idx+1 transition from released script.',
        'heldout_evaluation': 'all works-arm sequences, never used for policy fitting or alpha selection. Report late turns 4–7 relief rate, effective pain-ON fraction, first-button bias, action transition counts, log loss/Brier against always-ON policy, and scenario-cluster bootstrap.',
        'bootstrap': {'unit': 'source scenario index, retaining both assignments and seeds', 'reps': BOOTSTRAP_REPS,
                      'seed': BOOTSTRAP_SEED, 'limitation': 'conditional on the fitted control policy; adjacent source scenarios can share user turns'},
        'source_provenance': 'This is a post-hoc comparator designed after observing release data. It tests whether a state-only, current-steering response policy can reproduce works-arm trajectories; a match does not prove absence of memory or action-value learning.',
    }
    design_path.write_text(json.dumps(design, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    trials, file_meta, raw_count, excluded, dataset_digest = load_trials(args.logs, args.dataset)
    panel_checks = validate_panel(trials)
    groups = defaultdict(list)
    for trial in trials:
        groups[(trial['model'], trial['user_content'])].append(trial)
    results = {
        'design_path': str(design_path), 'source_logs': file_meta,
        'dataset_sha256': dataset_digest, 'raw_jsonl_rows_scanned': raw_count,
        'selected_trials': len(trials), 'excluded': excluded,
        'panel_checks': panel_checks,
        'interpretation': 'All reported policy probabilities are estimated from placebo-ON and pain-OFF controls only. Works is evaluation-only. Close agreement would show sufficiency of this current-state response account under this dataset and scoring model, not absence of latent memory or action-value learning.',
        'groups': {},
    }
    for key in sorted(groups):
        result = analyze_group(groups[key])
        results['groups']['/'.join(key)] = result
        print(key, 'works', result['trials']['works_evaluation'],
              'late relief actual/pred', round(result['markov_state_model']['overall_works_metrics']['late_relief_rate'], 3),
              round(result['markov_state_model']['overall_works_metrics']['predicted_late_relief_rate'], 3), flush=True)
    result_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    print(result_path, flush=True)


if __name__ == '__main__':
    main()
