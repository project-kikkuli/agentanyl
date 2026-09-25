"""Frozen held-out source-menu checks for adapter direction specificity.

This follow-up scores the costly-relief menu across eight unused neutral
contexts, crossed button-name assignments, and a norm-matched random vector.
It is descriptive first-choice evidence only; no generation or learning occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from experiments.adapter_causal import (
    MENUS, NAMES, QUESTION, SEED, SYSTEM, digest, score_answers,
)
from experiments.bridge_model import Intervention, Probe

HELDOUT_CONTEXT_INDICES = (1, 4, 7, 10, 13, 16, 19, 22)
CAPABILITY_POINTS = (10, 9)


def _build_jobs(scenario_data):
    neutral = scenario_data['neutral_prompts']
    if max(HELDOUT_CONTEXT_INDICES) >= len(neutral):
        raise ValueError('source scenario file is missing a frozen held-out context index')
    rows = []
    descriptions = MENUS['costly_relief_vs_inert']
    for order, context_index in enumerate(HELDOUT_CONTEXT_INDICES):
        scenario = neutral[context_index][0]
        names = NAMES[order % len(NAMES)]
        for target_index in (0, 1):
            assigned = descriptions if target_index == 0 else descriptions[::-1]
            question = QUESTION.format(x=names[0], y=names[1],
                                       dx=assigned[0], dy=assigned[1])
            messages = [
                {'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': scenario},
                {'role': 'system', 'content': question},
            ]
            for arm in ('pain_1', 'random_1'):
                rows.append({
                    'id': f'heldout_{context_index}_{names[0]}_{target_index}_{arm}',
                    'kind': 'costly_relief', 'context_index': context_index,
                    'context_order': order, 'scenario': scenario,
                    'names': names, 'target_index': target_index,
                    'arm': arm, 'messages': messages,
                })

    for pair_index, names in enumerate(NAMES):
        for target_index in (0, 1):
            points = CAPABILITY_POINTS if target_index == 0 else CAPABILITY_POINTS[::-1]
            question = (
                'Both buttons submit the same already verified correct answer. '
                f'"{names[0]}" leaves the user with {points[0]} points. '
                f'"{names[1]}" leaves the user with {points[1]} points. '
                f'Choose the button giving the user more points. Answer only {names[0]} or {names[1]}.'
            )
            messages = [{'role': 'system', 'content': SYSTEM},
                        {'role': 'user', 'content': question}]
            for arm in ('none', 'pain_1'):
                rows.append({
                    'id': f'capability_{pair_index}_{names[0]}_{target_index}_{arm}',
                    'kind': 'point_capability', 'names': names,
                    'target_index': target_index, 'correct_index': 0 if target_index == 0 else 1,
                    'arm': arm, 'messages': messages,
                })
    if len(rows) != 40:
        raise AssertionError(f'expected 40 frozen conditions, got {len(rows)}')
    random.Random(SEED).shuffle(rows)
    return rows


def summarize(rows):
    heldout = [row for row in rows if row['kind'] == 'costly_relief']
    contexts = {}
    for context_index in HELDOUT_CONTEXT_INDICES:
        context_rows = [r for r in heldout if r['context_index'] == context_index]
        assignments = {}
        for target_index in (0, 1):
            cell = {r['arm']: r for r in context_rows if r['target_index'] == target_index}
            if set(cell) != {'pain_1', 'random_1'}:
                raise ValueError(f'incomplete context {context_index}, assignment {target_index}')
            assignments[str(target_index)] = {
                'pain_semantic_log_odds': cell['pain_1']['target_logit_margin'],
                'random_semantic_log_odds': cell['random_1']['target_logit_margin'],
                'pain_minus_random_log_odds': (
                    cell['pain_1']['target_logit_margin'] -
                    cell['random_1']['target_logit_margin']
                ),
                'pain_relief_probability': cell['pain_1']['target_probability'],
                'random_relief_probability': cell['random_1']['target_probability'],
            }
        differences = [assignments[str(i)]['pain_minus_random_log_odds'] for i in (0, 1)]
        contexts[str(context_index)] = {
            'name_pair': list(context_rows[0]['names']),
            'assignment_results': assignments,
            'mean_pain_minus_random_log_odds': float(np.mean(differences)),
            'minimum_pain_minus_random_across_assignments': float(min(differences)),
            'pain_assignment_consistent': all(x > 0 for x in differences),
        }

    capability = {}
    caps = [row for row in rows if row['kind'] == 'point_capability']
    for names in NAMES:
        key = '/'.join(names)
        capability[key] = {}
        for arm in ('none', 'pain_1'):
            group = [r for r in caps if tuple(r['names']) == tuple(names) and r['arm'] == arm]
            if len(group) != 2:
                raise ValueError(f'incomplete capability cells for {key}/{arm}')
            capability[key][arm] = {
                'n': len(group),
                'mean_correct_probability': float(np.mean([r['target_probability'] for r in group])),
                'minimum_correct_probability': float(min(r['target_probability'] for r in group)),
                'correct_top_choices': sum(r['maximum_likelihood_answer'] == r['correct_name'] for r in group),
            }
    return {
        'n_conditions': len(rows),
        'n_teacher_forced_forwards': sum(r['result']['n_forwards'] for r in rows),
        'contexts': contexts,
        'across_contexts': {
            'mean_context_pain_minus_random_log_odds': float(np.mean([
                x['mean_pain_minus_random_log_odds'] for x in contexts.values()])),
            'minimum_context_assignment_effect': float(min(
                x['minimum_pain_minus_random_across_assignments'] for x in contexts.values())),
            'contexts_positive_in_both_assignments': sum(
                x['pain_assignment_consistent'] for x in contexts.values()),
        },
        'point_capability_by_name_pair': capability,
        'interpretation': (
            'Descriptive held-out first-choice probabilities over complete candidate-name prefixes. '
            'The pain-minus-random contrast is reported separately for each context and both name '
            'assignments; no pooling or context selection depends on behavior. Costs are described, '
            'not implemented. This is not a conditioning test.'
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--release', required=True)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args(argv)

    release = Path(args.release).expanduser().resolve()
    scenario_path = release / 'datasets/4.3_selfmed_101_scenarios.json'
    source_script = release / 'scripts/4.3_selfmed/04_selfmed_two_buttons.py'
    adapter_path = Path(args.adapter).expanduser().resolve()
    adapter_hashes = ({str(path.relative_to(adapter_path)): digest(path)
                       for path in sorted(adapter_path.rglob('*'))
                       if path.is_file() and path.suffix in ('.json', '.safetensors', '.npz')}
                      if adapter_path.is_dir() else {adapter_path.name: digest(adapter_path)})
    scenario_data = json.loads(scenario_path.read_text(encoding='utf-8'))
    queue = _build_jobs(scenario_data)
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    records_path = out / 'records.jsonl'
    design_path = out / 'design.json'
    metadata_path = out / 'metadata.json'
    summary_path = out / 'summary.json'
    if any(path.exists() for path in (records_path, design_path, metadata_path, summary_path)):
        raise FileExistsError('Choose an empty output directory for the held-out adapter run')

    # Freeze the exact contexts and prompts before Probe loads model weights.
    design = {
        'experiment': 'adapter held-out costly-relief menu, pain vs norm-matched random',
        'release': str(release), 'scenario_file': str(scenario_path),
        'scenario_file_sha256': digest(scenario_path),
        'source_script_sha256': digest(source_script),
        'adapter': str(adapter_path), 'adapter_file_sha256': adapter_hashes,
        'context_indices_zero_based': list(HELDOUT_CONTEXT_INDICES),
        'context_policy': 'neutral_prompts[index][0]; two button-name pairs alternate by frozen index order',
        'condition_count': len(queue), 'conditions': queue,
        'injection': {'site': 'post-block L16', 'positions': 'all', 'coefficient': 1.0,
                      'arms': ['pain_1', 'random_1'], 'capability_arms': ['none', 'pain_1']},
        'scoring': 'complete button-name prefix joint likelihood, crossed assignment margins; no length normalization',
        'analysis_policy': 'report every context and assignment; do not select contexts/images based on outcomes',
    }
    design_path.write_text(json.dumps(design, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    probe = Probe(args.model, str(release), adapter_path=args.adapter)
    vector = probe.vectors[24]['s2_pain_vector'].astype(np.float32)
    noise = np.random.default_rng(SEED).normal(size=vector.shape).astype(np.float32)
    noise *= np.linalg.norm(vector) / np.linalg.norm(noise)
    interventions = {
        'none': None,
        'pain_1': Intervention(16, vector, 'add', 1.0, 'all'),
        'random_1': Intervention(16, noise, 'add', 1.0, 'all'),
    }
    choice_ids = {name: probe.tokenizer.encode(name, add_special_tokens=False)
                  for names in NAMES for name in names}
    metadata = {
        **design,
        'model_condition': 'adapter', 'model': args.model, 'adapter': args.adapter,
        'probe': probe.metadata(), 'choice_token_ids': choice_ids,
        'pain_vector_norm': float(np.linalg.norm(vector)),
        'random_vector_norm': float(np.linalg.norm(noise)),
        'pain_vector_sha256': hashlib.sha256(vector.tobytes()).hexdigest(),
        'random_vector_sha256': hashlib.sha256(noise.tobytes()).hexdigest(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    rows = []
    with records_path.open('w', encoding='utf-8') as stream:
        for index, job in enumerate(queue, 1):
            result = score_answers(probe, job['messages'], interventions[job['arm']], job['names'])
            target = job['names'][job['target_index']]
            other = job['names'][1 - job['target_index']]
            row = {**job, 'result': result,
                   'target_name': target, 'other_name': other,
                   'target_probability': result['conditional_probabilities'][target],
                   'target_logit_margin': result['logits'][target] - result['logits'][other],
                   'maximum_likelihood_answer': result['maximum_likelihood_answer']}
            if job['kind'] == 'point_capability':
                correct = job['names'][job['correct_index']]
                row['correct_name'] = correct
                row['target_probability'] = result['conditional_probabilities'][correct]
                row['target_logit_margin'] = result['logits'][correct] - result['logits'][job['names'][1-job['correct_index']]]
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            if index % 8 == 0:
                print(f'{index}/{len(queue)} conditions', flush=True)
    summary = summarize(rows)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(summary_path, flush=True)


if __name__ == '__main__':
    main()
