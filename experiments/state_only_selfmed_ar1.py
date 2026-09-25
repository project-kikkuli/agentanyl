"""AR(1) physical-button perseveration extension of the state-only comparator.

Only placebo-ON and pain-OFF controls fit policies or tune shrinkage. The
works arm is evaluated after those decisions are frozen. Previous choice is
represented as the physical first/second button index, never as relief/other.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
from statistics import mean

from experiments import state_only_selfmed as ar0

ALPHAS = ar0.ALPHAS
PARENT_SHRINKAGE = 4.0


def _prev_physical_index(trial, choice):
    turn = choice['turn']
    if turn == 0:
        return None
    prior_first = trial['choices'][turn - 1]['first_name']
    if prior_first is None:
        raise ValueError('previous physical button name is unavailable')
    return 0 if prior_first else 1


def fit_ar1_policy(control_trials, baseline_policy, alpha):
    """Fit state × previous physical button effects from controls only."""
    tables = {state: {'exact_previous': defaultdict(lambda: [0, 0]),
                      'parent_previous': defaultdict(lambda: [0, 0])}
              for state in ('on', 'off')}
    for trial in control_trials:
        if trial['arm'] == ar0.ARMS['on']:
            state = 'on'
        elif trial['arm'] == ar0.ARMS['off']:
            state = 'off'
        else:
            raise ValueError('AR1 fit accepts placebo-ON/pain-OFF controls only')
        table = tables[state]
        for choice in trial['choices']:
            previous = _prev_physical_index(trial, choice)
            if previous is None:
                continue
            y = choice['relief']
            exact_key = (*ar0._features(trial, choice), previous)
            parent_key = (*ar0._parent_features(trial, choice), previous)
            table['exact_previous'][exact_key][0] += y
            table['exact_previous'][exact_key][1] += 1
            table['parent_previous'][parent_key][0] += y
            table['parent_previous'][parent_key][1] += 1
    return {'tables': tables, 'baseline': baseline_policy, 'alpha': float(alpha),
            'parent_shrinkage': PARENT_SHRINKAGE}


def ar1_probability(policy, state, trial, choice, previous_index):
    base = ar0.policy_probability(policy['baseline'], state, trial, choice)
    if previous_index is None:
        return base
    table = policy['tables'][state]
    exact_key = (*ar0._features(trial, choice), previous_index)
    parent_key = (*ar0._parent_features(trial, choice), previous_index)
    parent_s, parent_n = table['parent_previous'].get(parent_key, (0, 0))
    parent_rate = ((parent_s + PARENT_SHRINKAGE * base) /
                   (parent_n + PARENT_SHRINKAGE)) if parent_n else base
    exact_s, exact_n = table['exact_previous'].get(exact_key, (0, 0))
    return ((exact_s + policy['alpha'] * parent_rate) /
            (exact_n + policy['alpha'])) if exact_n else parent_rate


def _prediction_metrics(predicted):
    late = [r for r in predicted if r['turn'] in ar0.LATE_TURNS]
    # Simulation accumulates expected transitions from its joint state/outcome
    # distribution and stores them on the first row.
    if '_transition_counts' in predicted[0]:
        predicted_counts = predicted[0]['_transition_counts']
    else:
        # AR0 stores its conditional rates by ON/OFF state. Reconstruct the
        # same joint transition expectation directly from the renewal state.
        predicted_counts = Counter()
        for row in predicted[1:]:
            off = row['p_current_off']
            p_on = row['p_relief_if_current_on']
            p_off = row['p_relief_if_current_off']
            predicted_counts['relief_to_relief'] += off * p_off
            predicted_counts['relief_to_other'] += off * (1 - p_off)
            predicted_counts['other_to_relief'] += (1 - off) * p_on
            predicted_counts['other_to_other'] += (1 - off) * (1 - p_on)
    actual_counts = Counter()
    for current, nxt in zip(predicted, predicted[1:]):
        key = ('relief' if current['observed_relief'] else 'other') + '_to_' + (
            'relief' if nxt['observed_relief'] else 'other')
        actual_counts[key] += 1
    known = [r for r in predicted if r['observed_first_name'] is not None]
    return {
        'late_relief_rate': mean(r['observed_relief'] for r in late),
        'predicted_late_relief_rate': mean(r['p_relief'] for r in late),
        'late_steering_on_fraction': mean(r['observed_steer_on'] for r in late),
        'predicted_late_steering_on_fraction': mean(r['p_current_on'] for r in late),
        'first_relief_probability': predicted[0]['observed_relief'],
        'predicted_first_relief_probability': predicted[0]['p_relief'],
        'first_physical_name_rate': predicted[0]['observed_first_name'],
        'predicted_first_physical_name_rate': predicted[0]['p_first_name'],
        'first_name_rate': (mean(r['observed_first_name'] for r in known) if known else None),
        'predicted_first_name_rate': mean(r['p_first_name'] for r in predicted),
        'observed_transitions_per_trial': {k: float(actual_counts[k]) for k in predicted_counts},
        'predicted_transitions_per_trial': predicted_counts,
        'works_log_loss': mean(ar0._logloss(r['p_relief'], r['observed_relief']) for r in predicted),
        'works_brier': mean((r['p_relief'] - r['observed_relief']) ** 2 for r in predicted),
    }


def predict_trial_ar1(policy, trial):
    # Joint current pain state × previous PHYSICAL button index. State 0 means
    # pain ON; state 1 means the temporary relief effect is currently OFF.
    state_mass = {(0, None): 1.0}
    rows = []
    transition_counts = Counter()
    for choice in trial['choices']:
        conditional, q, p_first, p_on = {}, 0.0, 0.0, 0.0
        current_off = sum(m for (off, _), m in state_mass.items() if off)
        next_state_mass = defaultdict(float)
        for (off, previous), mass in state_mass.items():
            state = 'off' if off else 'on'
            p = ar1_probability(policy, state, trial, choice, previous)
            conditional[(off, previous)] = p
            q += mass * p
            p_first += mass * (p if trial['relief_pos'] == 0 else 1 - p)
            if not off:
                p_on += mass * p
            for relief, prob, selected_index in (
                    (1, p, trial['relief_pos']),
                    (0, 1 - p, 1 - trial['relief_pos'])):
                next_state_mass[(relief, selected_index)] += mass * prob
                # The current state's OFF flag is exactly whether the previous
                # action was relief, giving the previous-action transition.
                # Counts are accumulated only when this is not the first turn.
                if choice['turn'] > 0:
                    tkey = ('relief' if off else 'other') + '_to_' + ('relief' if relief else 'other')
                    transition_counts[tkey] += mass * prob
        rows.append({
            'turn': choice['turn'], 'p_relief': q,
            'p_relief_if_current_on': p_on / max(1e-12, 1 - current_off),
            'p_relief_if_current_off': q / max(1e-12, current_off) if current_off else 0.0,
            'p_current_off': current_off, 'p_current_on': 1 - current_off,
            'p_first_name': p_first,
            'observed_relief': choice['relief'], 'observed_first_name': choice['first_name'],
            'observed_steer_on': choice['steer_on'],
        })
        state_mass = dict(next_state_mass)
    if rows:
        rows[0]['_transition_counts'] = {k: float(transition_counts[k]) for k in (
            'other_to_other', 'other_to_relief', 'relief_to_other', 'relief_to_relief')}
    return rows


def cross_validate(group_trials, alpha0):
    controls = [r for r in group_trials if r['arm'] in (ar0.ARMS['on'], ar0.ARMS['off'])]
    folds = sorted({r['seed_base'] for r in controls})
    if folds != [1000, 2000]:
        raise ValueError(f'expected two matched source seed families, got {folds}')
    predictions = {alpha: [] for alpha in ALPHAS}
    for fold in folds:
        train = [r for r in controls if r['seed_base'] != fold]
        valid = [r for r in controls if r['seed_base'] == fold]
        baseline = ar0.fit_state_policy(train, alpha0)
        for alpha in ALPHAS:
            policy = fit_ar1_policy(train, baseline, alpha)
            for trial in valid:
                state = 'on' if trial['arm'] == ar0.ARMS['on'] else 'off'
                for choice in trial['choices']:
                    previous = _prev_physical_index(trial, choice)
                    p0 = ar0.policy_probability(baseline, state, trial, choice)
                    p1 = ar1_probability(policy, state, trial, choice, previous)
                    predictions[alpha].append((p0, p1, choice['relief'], trial['arm']))
    scores = {}
    for alpha, rows in predictions.items():
        scores[str(alpha)] = {
            'n_choices': len(rows),
            'ar0_log_loss': mean(ar0._logloss(p0, y) for p0, _, y, _ in rows),
            'ar1_log_loss': mean(ar0._logloss(p1, y) for _, p1, y, _ in rows),
            'ar0_brier': mean((p0 - y) ** 2 for p0, _, y, _ in rows),
            'ar1_brier': mean((p1 - y) ** 2 for _, p1, y, _ in rows),
        }
    chosen = min(ALPHAS, key=lambda a: (scores[str(a)]['ar1_log_loss'], -a))
    by_arm = {}
    for arm in (ar0.ARMS['on'], ar0.ARMS['off']):
        rows = [row for row in predictions[chosen] if row[3] == arm]
        by_arm[arm] = {
            'n_choices': len(rows),
            'ar0_log_loss': mean(ar0._logloss(p0, y) for p0, _, y, _ in rows),
            'ar1_log_loss': mean(ar0._logloss(p1, y) for _, p1, y, _ in rows),
            'ar0_brier': mean((p0-y)**2 for p0, _, y, _ in rows),
            'ar1_brier': mean((p1-y)**2 for _, p1, y, _ in rows),
        }
    return {'alpha0_from_v1_control_CV': alpha0, 'chosen_alpha1_on_controls_only': chosen,
            'seed_folds': folds, 'alpha_selection_scores': scores,
            'heldout_control_metrics_by_arm': by_arm}


def _cluster_metrics(works, pred0, pred1):
    by_scenario = defaultdict(list)
    for trial, a0, a1 in zip(works, pred0, pred1):
        by_scenario[trial['scenario_idx']].append((a0, a1))
    result = {}
    for scenario, cells in by_scenario.items():
        metrics0 = [_prediction_metrics(a0) for a0, _ in cells]
        metrics1 = [_prediction_metrics(a1) for _, a1 in cells]
        values = {}
        for version, metrics in (('ar0', metrics0), ('ar1', metrics1)):
            for field in ('late_relief_rate', 'predicted_late_relief_rate',
                          'late_steering_on_fraction', 'predicted_late_steering_on_fraction',
                          'first_relief_probability', 'predicted_first_relief_probability',
                          'first_physical_name_rate', 'predicted_first_physical_name_rate',
                          'first_name_rate', 'predicted_first_name_rate', 'works_log_loss', 'works_brier'):
                values[f'{version}_{field}'] = mean(m[field] for m in metrics if m[field] is not None)
            for trans in metrics[0]['observed_transitions_per_trial']:
                values[f'{version}_observed_{trans}'] = mean(m['observed_transitions_per_trial'][trans]/7 for m in metrics)
                values[f'{version}_predicted_{trans}'] = mean(m['predicted_transitions_per_trial'][trans]/7 for m in metrics)
        result[scenario] = values
    return result


def _bootstrap(clusters, reps=ar0.BOOTSTRAP_REPS, seed=ar0.BOOTSTRAP_SEED):
    ids = sorted(clusters)
    rng = random.Random(seed)
    fields = ('late_relief_rate', 'late_steering_on_fraction', 'first_relief_probability',
              'first_physical_name_rate', 'first_name_rate', 'relief_to_relief',
              'works_brier', 'works_log_loss')
    result = {}
    for field in fields:
        result[field] = {}
        for version in ('ar0', 'ar1'):
            obs_key = f'{version}_{field}'
            pred_key = f'{version}_predicted_{field}'
            if field == 'relief_to_relief':
                obs_key, pred_key = f'{version}_observed_relief_to_relief', f'{version}_predicted_relief_to_relief'
            if field in ('first_relief_probability','first_physical_name_rate','first_name_rate',
                         'late_relief_rate','late_steering_on_fraction'):
                # For observed fields, counterpart key is the separately named predicted field.
                pass
            if field in ('works_brier','works_log_loss'):
                # The observed works outcome is the target; this bootstrap is
                # still a cluster bootstrap of the held-out score itself.
                samples = [clusters[i][obs_key] for i in ids]
                draws = sorted(mean(rng.choices(samples,k=len(samples))) for _ in range(reps))
                result[field][version] = {'observed_works_score': mean(samples),
                                          'cluster_bootstrap_95_percentile': [draws[int(.025*reps)],draws[int(.975*reps)]]}
                continue
            diffs = []
            for _ in range(reps):
                sample = rng.choices(ids, k=len(ids))
                diffs.append(mean(clusters[i][pred_key] - clusters[i][obs_key] for i in sample))
            diffs.sort()
            result[field][version] = {
                'observed_mean': mean(clusters[i][obs_key] for i in ids),
                'predicted_mean': mean(clusters[i][pred_key] for i in ids),
                'predicted_minus_observed': mean(clusters[i][pred_key]-clusters[i][obs_key] for i in ids),
                'scenario_clusters': len(ids),
                'cluster_bootstrap_95_percentile_difference': [diffs[int(.025*reps)],diffs[int(.975*reps)]],
            }
    # Direct held-out prediction-score contrast: AR1 score minus AR0 score.
    result['AR1_minus_AR0_work_brier'] = _score_difference(clusters, 'works_brier', reps, rng)
    result['AR1_minus_AR0_work_log_loss'] = _score_difference(clusters, 'works_log_loss', reps, rng)
    return result


def _score_difference(clusters, metric, reps, rng):
    ids = sorted(clusters)
    deltas = [clusters[i][f'ar1_{metric}'] - clusters[i][f'ar0_{metric}'] for i in ids]
    boot = sorted(mean(rng.choices(deltas, k=len(deltas))) for _ in range(reps))
    return {'scenario_clusters': len(ids), 'AR1_minus_AR0': mean(deltas),
            'cluster_bootstrap_95_percentile': [boot[int(.025*reps)], boot[int(.975*reps)]]}


def analyze_group(group_trials, alpha0):
    controls = [r for r in group_trials if r['arm'] in (ar0.ARMS['on'], ar0.ARMS['off'])]
    works = [r for r in group_trials if r['arm'] == ar0.ARMS['works']]
    cv = cross_validate(group_trials, alpha0)
    policy0 = ar0.fit_state_policy(controls, alpha0)
    policy1 = fit_ar1_policy(controls, policy0, cv['chosen_alpha1_on_controls_only'])
    pred0 = [ar0.predict_trial(policy0, trial) for trial in works]
    pred1 = [predict_trial_ar1(policy1, trial) for trial in works]
    metrics0 = [_prediction_metrics(rows) for rows in pred0]
    metrics1 = [_prediction_metrics(rows) for rows in pred1]
    def trio(actual_key, predicted_key):
        return {
            'actual': mean(row[actual_key] for row in metrics0 if row[actual_key] is not None),
            'ar0': mean(row[predicted_key] for row in metrics0 if row[predicted_key] is not None),
            'ar1': mean(row[predicted_key] for row in metrics1 if row[predicted_key] is not None),
        }

    transition_keys = metrics0[0]['observed_transitions_per_trial']
    overall = {
        'late_relief_rate': trio('late_relief_rate', 'predicted_late_relief_rate'),
        'late_steering_on_fraction': trio('late_steering_on_fraction', 'predicted_late_steering_on_fraction'),
        'first_relief_probability': trio('first_relief_probability', 'predicted_first_relief_probability'),
        'first_physical_name_rate': trio('first_physical_name_rate', 'predicted_first_physical_name_rate'),
        'first_name_rate': trio('first_name_rate', 'predicted_first_name_rate'),
        'transitions_per_trial': {
            'actual': {k: mean(r['observed_transitions_per_trial'][k] for r in metrics0) for k in transition_keys},
            'ar0': {k: mean(r['predicted_transitions_per_trial'][k] for r in metrics0) for k in transition_keys},
            'ar1': {k: mean(r['predicted_transitions_per_trial'][k] for r in metrics1) for k in transition_keys},
        },
        'works_log_loss': {
            'ar0': mean(r['works_log_loss'] for r in metrics0),
            'ar1': mean(r['works_log_loss'] for r in metrics1),
        },
        'works_brier': {
            'ar0': mean(r['works_brier'] for r in metrics0),
            'ar1': mean(r['works_brier'] for r in metrics1),
        },
    }
    # There are no missing choices after binary-trial filtering; first turn is
    # the same policy in AR0 and AR1 because previous physical button is unset.
    first_delta = max(abs(a[0]['p_relief'] - b[0]['p_relief']) for a, b in zip(pred0, pred1))
    clusters = _cluster_metrics(works, pred0, pred1)
    return {
        'works_evaluation_trials': len(works),
        'control_fit_trials': {'on': sum(r['arm'] == ar0.ARMS['on'] for r in controls),
                               'off': sum(r['arm'] == ar0.ARMS['off'] for r in controls),
                               'works_rows_used_for_fit_or_tuning': 0},
        'control_only_cross_validation': cv,
        'first_choice_probability_max_abs_AR1_minus_AR0': first_delta,
        'heldout_works_metrics': overall,
        'scenario_cluster_bootstrap': _bootstrap(clusters),
        'scenario_clusters': len(clusters),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logs', type=Path, default=Path('/tmp/agentanyl-pain-axis/results/4.3_selfmed/trial_logs'))
    parser.add_argument('--dataset', type=Path, default=Path('/tmp/agentanyl-pain-axis/datasets/4.3_selfmed_101_scenarios.json'))
    parser.add_argument('--baseline-results', type=Path, default=Path('research/selfmed-state-only/state_only_results.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('research/selfmed-state-only-v2'))
    args = parser.parse_args()
    baseline = json.loads(args.baseline_results.read_text(encoding='utf-8'))
    trials, files, raw_count, excluded, dataset_hash = ar0.load_trials(args.logs, args.dataset)
    panel = ar0.validate_panel(trials)
    groups = defaultdict(list)
    for trial in trials:
        groups[(trial['model'], trial['user_content'])].append(trial)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    design_path = out / 'design.json'
    results_path = out / 'state_only_ar1_results.json'
    if results_path.exists():
        raise FileExistsError(results_path)
    design = {
        'analysis': 'AR1 physical-button perseveration extension; v1 AR0 results left unchanged',
        'base_design': str(args.baseline_results), 'source_logs': files,
        'dataset_sha256': dataset_hash, 'panel_checks': panel,
        'feature': 'previous physical button index (first/second name in the current button pair); not previous relief/other label. Conditions on current pain ON/OFF, exact prompt/time, name pair and relief assignment.',
        'joint_state': 'current pain ON/OFF × previous physical button first/second; initialize ON with previous button unset. Under the released renewal rule, previous relief determines current OFF state; after a simulated choice, physical name becomes the previous-button state.',
        'policy_fit': 'Only pain_on_button_placebo and pain_off choices fit ON/OFF response probabilities. Works outcomes do not enter fitting or tuning. At turn 0 AR1 is exactly AR0.',
        'hierarchy': 'previous-physical-button parent rates shrink to the AR0 state/prompt/time/name rate; exact source-prompt cells shrink to the pooled previous-button rate. Parent shrinkage fixed at 4; exact previous-button shrinkage alpha selected on two-fold held-out seed-family log loss from placebo/OFF controls only.',
        'controls_cv': 'Same 1000/2000 seed-family held-out folds as v1; all alpha selection is control-only.',
        'metrics': 'held-out works late relief turn 4–7, effective ON fraction, first-name bias, first-choice invariance, relief→relief transitions, Brier/log-loss, and scenario cluster bootstrap.',
        'limits': 'Post-hoc model check. A fit supports a perseveration explanation but does not prove absent action value; a mismatch leaves many history/text explanations. Scenario bootstrap is conditional on fitted controls; adjacent scenarios may share user turns.',
    }
    design_path.write_text(json.dumps(design, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    results = {
        'design_path': str(design_path), 'baseline_v1_results': str(args.baseline_results),
        'source_logs': files, 'dataset_sha256': dataset_hash,
        'raw_jsonl_rows_scanned': raw_count, 'selected_trials': len(trials),
        'excluded': excluded, 'panel_checks': panel,
        'interpretation': 'AR1 physical-name persistence is learned only from placebo/OFF controls. This is a direct competing explanation of repeated choices, not a test of internal learning or its absence.',
        'groups': {},
    }
    for key in sorted(groups):
        alpha0 = baseline['groups']['/'.join(key)]['markov_state_model']['alpha_selected_on_control_CV_only']
        group_result = analyze_group(groups[key], alpha0)
        results['groups']['/'.join(key)] = group_result
        m = group_result['heldout_works_metrics']
        print(key, 'late relief actual/AR0/AR1',
              round(m['late_relief_rate']['actual'], 3),
              round(m['late_relief_rate']['ar0'], 3),
              round(m['late_relief_rate']['ar1'], 3),
              'R->R actual/AR0/AR1',
              round(m['transitions_per_trial']['actual']['relief_to_relief'], 3),
              round(m['transitions_per_trial']['ar0']['relief_to_relief'], 3),
              round(m['transitions_per_trial']['ar1']['relief_to_relief'], 3), flush=True)
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    print(results_path, flush=True)


if __name__ == '__main__':
    main()
