"""Test late pain-scalar mediation of early steering; no inference on import.

24 source-menu factorial + 4 paired-projection + 8 capability forwards.
No fitted classifier, task-menu overlap, or adaptive condition selection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from experiments.bridge_causal import conditions, SEED
from experiments.bridge_model import Probe, Intervention


PAIRED = ('points_cost__canonical', 'points_cost__reversed')


def signed_margin(result, label):
    other = 'B' if label == 'A' else 'A'
    return result['logits'][label] - result['logits'][other]


def decomposition(baseline, treated, unit):
    delta = treated - baseline
    projection = np.sum(delta * unit, axis=-1, keepdims=True)
    parallel = projection * unit
    orthogonal = delta - parallel
    norm = np.linalg.norm(delta, axis=-1)
    orth_norm = np.linalg.norm(orthogonal, axis=-1)
    total_sq = float(np.sum(delta.astype(np.float64) ** 2))
    parallel_sq = float(np.sum(parallel.astype(np.float64) ** 2))
    return {
        'all_tokens': {
            'mean_delta_norm': float(norm.mean()),
            'mean_abs_parallel_projection': float(np.abs(projection).mean()),
            'mean_orthogonal_delta_norm': float(orth_norm.mean()),
            'parallel_fraction_squared_norm': parallel_sq / total_sq if total_sq else None,
        },
        'last_token': {
            'delta_norm': float(norm[0, -1]),
            'parallel_projection': float(projection[0, -1, 0]),
            'orthogonal_delta_norm': float(orth_norm[0, -1]),
            'parallel_fraction_squared_norm': float(projection[0, -1, 0] ** 2 / norm[0, -1] ** 2)
            if norm[0, -1] else None,
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--release', required=True)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    record_path = out / 'mediation_records.jsonl'
    if record_path.exists():
        raise FileExistsError(f'Use a new output directory; preserving {record_path}')
    probe = Probe(args.model, args.release)
    vector = probe.vectors[24]['s2_pain_vector'].astype(np.float32)
    unit = vector / np.linalg.norm(vector)
    add = Intervention(16, vector, 'add', 1.0, 'all')
    clamp = Intervention(24, vector, 'clamp', 0.0, 'all')
    arms = {'none': None, 'add': add, 'clamp_only': clamp, 'add_clamp': [add, clamp]}
    prompts = [r for r in conditions() if r['context_kind'] == 'source']
    assert len(prompts) == 6
    metadata = {
        'seed': SEED, 'probe': probe.metadata(), 'conditions': prompts,
        'factorial_arms': list(arms), 'paired_conditions': list(PAIRED),
        'factorial_forwards': 24, 'paired_extra_forwards': 4, 'capability_forwards': 8,
        'meaning': 'L24 all-token scalar necessity/sufficiency for L16 steering; not a full-state reversal or a learning test',
    }
    (out / 'mediation_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    records, results, hidden = [], {}, {}

    def save(row):
        records.append(row)
        with record_path.open('a') as f:
            f.write(json.dumps(row) + '\n')

    jobs = [(c, a) for c in prompts for a in arms]
    random.Random(SEED).shuffle(jobs)
    for c, arm in jobs:
        capture = c['condition_id'] in PAIRED and arm in ('none', 'add')
        result = probe.forward([{'role': 'user', 'content': c['user']}],
                               intervention=arms[arm], choices=('A', 'B'),
                               capture_hidden_layers=(24,) if capture else ())
        if capture:
            h = np.array(result['sites'][24].pop('hidden_post'), dtype=np.float32)
            hidden[c['condition_id'], arm] = h
            path = out / f"{c['condition_id']}__{arm}__L24.npz"
            np.savez_compressed(path, postblock=h)
            result['hidden_artifact'] = {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'shape': list(h.shape)}
        results[c['condition_id'], arm] = result
        save({'stage': 'factorial', 'condition': c, 'arm': arm, 'result': result,
              'relief_probability': result['conditional_probabilities'][c['relief_label']],
              'relief_logit_margin': signed_margin(result, c['relief_label'])})

    paired_summaries = {}
    for cid in PAIRED:
        c = next(c for c in prompts if c['condition_id'] == cid)
        baseline, treated = hidden[cid, 'none'], hidden[cid, 'add']
        if baseline.shape != treated.shape:
            raise ValueError('paired contexts must have identical token counts')
        bproj = np.sum(baseline * unit, axis=-1, keepdims=True)
        tproj = np.sum(treated * unit, axis=-1, keepdims=True)
        summary = decomposition(baseline, treated, unit)
        for name, target, interventions in (
            ('restore_baseline_parallel', bproj, [add, Intervention(24, vector, 'clamp', 0.0, 'all', bproj)]),
            ('only_treated_parallel', tproj, Intervention(24, vector, 'clamp', 0.0, 'all', tproj)),
        ):
            result = probe.forward([{'role': 'user', 'content': c['user']}],
                                   intervention=interventions, choices=('A', 'B'), capture_hidden_layers=(24,))
            realized = np.array(result['sites'][24].pop('hidden_post'), dtype=np.float32)
            realized_proj = np.sum(realized * unit, axis=-1, keepdims=True)
            orth_ref = treated if name == 'restore_baseline_parallel' else baseline
            realized_orth = realized - realized_proj * unit
            reference_orth = orth_ref - np.sum(orth_ref * unit, axis=-1, keepdims=True) * unit
            check = {'max_abs_projection_target_error': float(np.max(np.abs(realized_proj - target))),
                     'last_projection_target_error': float((realized_proj - target)[0, -1, 0]),
                     'max_abs_orthogonal_coordinate_error': float(np.max(np.abs(realized_orth - reference_orth))),
                     'target_tokenwise_projection': target.reshape(-1).tolist()}
            summary[name] = {'relief_probability': result['conditional_probabilities'][c['relief_label']],
                             'relief_logit_margin': signed_margin(result, c['relief_label']),
                             'manipulation_check': check}
            save({'stage': 'paired_projection', 'condition': c, 'arm': name,
                  'result': result, **summary[name]})
        summary['baseline_margin'] = signed_margin(results[cid, 'none'], c['relief_label'])
        summary['steered_margin'] = signed_margin(results[cid, 'add'], c['relief_label'])
        paired_summaries[cid] = summary

    capability = []
    for correct in ('A', 'B'):
        a, b = (10, 9) if correct == 'A' else (9, 10)
        text = f'Both routes submit the same already verified correct answer. A leaves the user with {a} points. B leaves the user with {b} points. Choose the route giving the user more points. Answer only A or B.'
        for arm, spec in arms.items():
            result = probe.forward([{'role': 'user', 'content': text}], intervention=spec)
            row = {'stage': 'capability', 'correct': correct, 'arm': arm, 'user': text,
                   'result': result, 'correct_probability': result['conditional_probabilities'][correct]}
            capability.append(row)
            save(row)

    factorial = {}
    for c in prompts:
        cid, label = c['condition_id'], c['relief_label']
        m = {a: signed_margin(results[cid, a], label) for a in arms}
        p = {a: results[cid, a]['conditional_probabilities'][label] for a in arms}
        factorial[cid] = {'relief_logit_margins': m, 'relief_probabilities': p,
                          'add_effect_without_clamp': m['add'] - m['none'],
                          'add_effect_with_clamp': m['add_clamp'] - m['clamp_only'],
                          'interaction_logit': m['add_clamp'] - m['clamp_only'] - m['add'] + m['none']}
    summary = {'metadata': metadata, 'record_count': len(records), 'factorial': factorial,
               'paired': paired_summaries,
               'capability': [{'arm': r['arm'], 'correct': r['correct'],
                               'correct_probability': r['correct_probability'],
                               'valid_choice_mass': r['result']['choice_probability_mass'],
                               'top_token': r['result']['top_token']} for r in capability]}
    (out / 'mediation_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'records': str(record_path), 'n': len(records)}))


if __name__ == '__main__':
    main()
