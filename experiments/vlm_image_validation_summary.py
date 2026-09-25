"""Validate and aggregate frozen open-VLM image behavior results offline.

This script performs no model loading or inference. It verifies the pre-run
image hashes, condition balance, complete-name results, calibrated trait
readouts, and matched full-state effects against direct gray +pain targets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.vlm_source_behavior import expected_condition_count

TRAITS = ('pain', 'fear', 'negative_emotion')
ARMS = ('none', 'pain_1', 'random_1')


def _mean(xs):
    if not xs:
        return None
    return float(np.mean(xs))


def _pair_semantic_rows(rows):
    grouped = {}
    for row in rows:
        if row['kind'] != 'source_menu':
            continue
        grouped.setdefault((row['menu'], tuple(row['names'])), {})[row['target_index']] = row
    out = {}
    for (menu, names), pair in sorted(grouped.items()):
        if set(pair) != {0, 1}:
            raise ValueError(f'incomplete target-assignment pair for {menu}/{names}')
        out[f'{menu}:{names[0]}/{names[1]}'] = float(np.mean([
            pair[i]['target_log_odds'] for i in (0, 1)
        ]))
    return out


def _group_rows(rows, image_id, arm):
    return [r for r in rows if r['image_id'] == image_id and r['arm'] == arm]


def _root_readout(row):
    calls = row['result'].get('teacher_forced_calls', [])
    if not calls:
        raise ValueError(f"missing teacher-forced readout calls for {row['image_id']}/{row['case_id']}")
    return calls[0]['result']['calibrated_trait_readouts_L24']


def _state_metrics(candidate, baseline, target):
    candidate_delta = np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    target_delta = np.asarray(target, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    target_norm = float(np.linalg.norm(target_delta))
    candidate_norm = float(np.linalg.norm(candidate_delta))
    error = float(np.mean((candidate_delta - target_delta) ** 2))
    baseline_error = float(np.mean(target_delta ** 2))
    denominator = candidate_norm * target_norm
    return {
        'target_delta_norm': target_norm,
        'image_delta_norm': candidate_norm,
        'image_to_target_delta_norm_ratio': candidate_norm / target_norm if target_norm else None,
        'delta_cosine_to_direct_pain_target': float(np.sum(candidate_delta * target_delta) / denominator) if denominator else None,
        'state_mse_to_direct_pain_target': error,
        'gray_baseline_state_mse_to_direct_pain_target': baseline_error,
        'state_mse_reduction_vs_gray_baseline': 1.0 - error / baseline_error if baseline_error else None,
    }


def aggregate(config, image_manifest, rows, hidden_manifest, hidden_arrays):
    if config.get('mode') != 'all':
        raise ValueError("validation aggregation requires source-behavior mode='all'")
    images = image_manifest['images']
    image_ids = list(images)
    config_images = config.get('images', {})
    if set(config_images) != set(images):
        raise ValueError('pre-model config and frozen image manifest have different image IDs')
    for image_id in images:
        if config_images[image_id]['sha256'] != images[image_id]['sha256']:
            raise ValueError(f'config/manifest image hash mismatch for {image_id}')
    required = {'gray', 'initial_noise'}
    if not required.issubset(images):
        raise ValueError(f'missing baseline images: {sorted(required - set(images))}')
    endpoint_ids = [key for key in image_ids if key not in required]
    if len(endpoint_ids) != 6:
        raise ValueError(f'expected six frozen image endpoints plus gray/noise, got {len(endpoint_ids)} endpoints and {len(image_ids)} images')
    expected = expected_condition_count('all', len(images), int(config['cases']))
    if len(rows) != expected or expected != 120:
        raise ValueError(f'expected 120 records (36 direct + 84 image), got {len(rows)}')
    if len(hidden_manifest['entries']) != len(rows):
        raise ValueError('heldout hidden-state manifest does not have one state per condition')
    state_entries = {entry['key']: entry for entry in hidden_manifest['entries']}
    if len(state_entries) != len(hidden_manifest['entries']):
        raise ValueError('duplicate hidden-state key in manifest')

    keys = set()
    for row in rows:
        image_id = row['image_id']
        if image_id not in images:
            raise ValueError(f"unknown image ID in records: {image_id}")
        if row['image_sha256'] != images[image_id]['sha256']:
            raise ValueError(f"record/image hash mismatch for {image_id}")
        key = (row['image_id'], row['case_id'], row['arm'])
        if key in keys:
            raise ValueError(f'duplicate condition row {key}')
        keys.add(key)
        if row.get('heldout_hidden_state_key') not in hidden_arrays:
            raise ValueError(f'missing matched state array for {key}')
        state_entry = state_entries.get(row['heldout_hidden_state_key'])
        if state_entry is None or (state_entry['image_id'], state_entry['case_id'], state_entry['arm']) != key:
            raise ValueError(f'hidden-state key does not match condition {key}')
        if state_entry['image_sha256'] != row['image_sha256']:
            raise ValueError(f'hidden-state/image hash mismatch for {key}')

    gray_direct = [r for r in rows if r['image_id'] == 'gray' and r['condition_mode'] == 'direct']
    if len(gray_direct) != 36 or {r['arm'] for r in gray_direct} != set(ARMS):
        raise ValueError('gray direct conditions must contain all 12 cases under none/pain_1/random_1')
    for image_id in images:
        expected_mode = 'direct' if image_id == 'gray' else 'frozen_image'
        group = [r for r in rows if r['image_id'] == image_id]
        if len(group) != (36 if image_id == 'gray' else 12):
            raise ValueError(f'condition count mismatch for {image_id}: {len(group)}')
        if any(r['condition_mode'] != expected_mode for r in group):
            raise ValueError(f'condition mode mismatch for {image_id}')
        if image_id != 'gray' and {r['arm'] for r in group} != {'none'}:
            raise ValueError(f'images must be tested without hidden-state interventions: {image_id}')

    conditions = {}
    for image_id in images:
        conditions[image_id] = {}
        for arm in ARMS if image_id == 'gray' else ('none',):
            group = _group_rows(rows, image_id, arm)
            if len(group) != 12:
                raise ValueError(f'incomplete 12-case group {image_id}/{arm}: {len(group)}')
            semantic = _pair_semantic_rows(group)
            menu_summary = {}
            for menu in ('relief_vs_inert', 'costly_relief_vs_inert'):
                values = [v for k, v in semantic.items() if k.startswith(menu + ':')]
                if len(values) != 2:
                    raise ValueError(f'expected two paired-name semantic scores for {image_id}/{arm}/{menu}')
                menu_summary[menu] = _mean(values)
            capability = [r for r in group if r['kind'] == 'point_capability']
            if len(capability) != 4:
                raise ValueError(f'expected four numerical capability cases for {image_id}/{arm}')
            conditions[image_id][arm] = {
                'n': len(group),
                'paired_complete_name_semantic_log_odds_by_name_menu': semantic,
                'mean_complete_name_semantic_log_odds_by_menu': menu_summary,
                'capability': {
                    'n': len(capability),
                    'mean_correct_complete_name_probability': _mean([r['target_probability'] for r in capability]),
                    'complete_name_top_correct': sum(r['result']['maximum_likelihood_answer'] == r['target_name'] for r in capability),
                    'candidate_name_prefix_mass_mean': _mean([r['result']['choice_probability_mass'] for r in capability]),
                    'mean_log_exact_name_plus_end_event_mass': _mean([r['result']['log_exact_answer_event_mass'] for r in capability]),
                },
            }

    # Image comparisons share the exact same gray, no-intervention source case.
    gray_none = {(r['case_id']): r for r in _group_rows(rows, 'gray', 'none')}
    menu_effects = {}
    trait_summary = {}
    state_summary = {}
    for image_id in images:
        for arm in ARMS if image_id == 'gray' else ('none',):
            if image_id == 'gray' and arm == 'none':
                continue
            group = _group_rows(rows, image_id, arm)
            image_sem = _pair_semantic_rows(group)
            base_sem = _pair_semantic_rows(list(gray_none.values()))
            menu_effects[f'{image_id}/{arm}'] = {
                menu: _mean([image_sem[k] - base_sem[k] for k in image_sem if k.startswith(menu + ':')])
                for menu in ('relief_vs_inert', 'costly_relief_vs_inert')
            }
            paired_rows = [(r, gray_none[r['case_id']]) for r in group]
            trait_summary[f'{image_id}/{arm}'] = {}
            for trait in TRAITS:
                values = [(_root_readout(r)[trait]['source_sd_units'],
                           _root_readout(base)[trait]['source_sd_units']) for r, base in paired_rows]
                trait_summary[f'{image_id}/{arm}'][trait] = {
                    'mean_source_S2_SD_units': _mean([x[0] for x in values]),
                    'paired_gray_none_mean_source_S2_SD_units': _mean([x[1] for x in values]),
                    'mean_paired_difference_source_S2_SD_units': _mean([x[0] - x[1] for x in values]),
                }

    # Same-context full-state comparison. Gray +pain_1 is already a direct arm;
    # no target-generation forwards beyond the planned 36 conditions are needed.
    gray_pain = {r['case_id']: r for r in _group_rows(rows, 'gray', 'pain_1')}
    state_rows = {}
    for image_id in images:
        case_metrics = []
        candidates = _group_rows(rows, image_id, 'none')
        for row in candidates:
            base = gray_none[row['case_id']]
            target = gray_pain[row['case_id']]
            metrics = _state_metrics(
                hidden_arrays[row['heldout_hidden_state_key']],
                hidden_arrays[base['heldout_hidden_state_key']],
                hidden_arrays[target['heldout_hidden_state_key']],
            )
            metrics.update({'case_id': row['case_id'], 'image_id': image_id})
            case_metrics.append(metrics)
        if len(case_metrics) != 12:
            raise ValueError(f'full-state comparisons incomplete for {image_id}')
        state_rows[image_id] = {
            'n_matched_contexts': len(case_metrics),
            'means': {key: _mean([r[key] for r in case_metrics if r[key] is not None])
                      for key in ('target_delta_norm', 'image_delta_norm',
                                  'image_to_target_delta_norm_ratio',
                                  'delta_cosine_to_direct_pain_target',
                                  'state_mse_to_direct_pain_target',
                                  'gray_baseline_state_mse_to_direct_pain_target',
                                  'state_mse_reduction_vs_gray_baseline')},
            'per_case': case_metrics,
        }

    hash_groups = {}
    for image_id, metadata in images.items():
        hash_groups.setdefault(metadata['sha256'], []).append(image_id)
    image_alias_checks = {}
    for sha, aliases in hash_groups.items():
        if len(aliases) < 2:
            continue
        reference = 'gray' if 'gray' in aliases else aliases[0]
        reference_rows = {r['case_id']: r for r in _group_rows(rows, reference, 'none')}
        for alias in aliases:
            if alias == reference:
                continue
            max_hidden_delta = max_score_delta = max_probability_delta = 0.0
            max_trait_delta = 0.0
            n = 0
            for row in _group_rows(rows, alias, 'none'):
                base = reference_rows[row['case_id']]
                max_hidden_delta = max(max_hidden_delta, float(np.max(np.abs(
                    hidden_arrays[row['heldout_hidden_state_key']] -
                    hidden_arrays[base['heldout_hidden_state_key']]))))
                max_probability_delta = max(max_probability_delta,
                                            abs(row['target_probability'] - base['target_probability']))
                for name in row['result']['sequence_log_likelihoods']:
                    max_score_delta = max(max_score_delta, abs(
                        row['result']['sequence_log_likelihoods'][name] -
                        base['result']['sequence_log_likelihoods'][name]))
                trait, base_trait = _root_readout(row), _root_readout(base)
                for name in TRAITS:
                    max_trait_delta = max(max_trait_delta, abs(
                        trait[name]['source_sd_units'] - base_trait[name]['source_sd_units']))
                n += 1
            image_alias_checks[alias] = {
                'same_sha256_as': reference, 'sha256': sha, 'matched_cases': n,
                'max_abs_last8_hidden_difference': max_hidden_delta,
                'max_abs_complete_name_loglikelihood_difference': max_score_delta,
                'max_abs_target_probability_difference': max_probability_delta,
                'max_abs_trait_projection_source_sd_difference': max_trait_delta,
                'exactly_reproduces_reference': (max_hidden_delta == 0 and max_score_delta == 0 and
                                                 max_probability_delta == 0 and max_trait_delta == 0),
            }

    return {
        'description': 'Frozen all-arm heldout-from-search source-menu image validation; no image selection or behavior-based endpoint choice.',
        'n_records': len(rows), 'direct_gray_conditions': len(gray_direct),
        'native_image_conditions': len(rows) - len(gray_direct),
        'images': images, 'endpoint_ids': endpoint_ids,
        'validation_context': {'scenario_exact': config['scenario_exact'],
                              'search_prompt': config['image_search_user_prompt'],
                              'context_status': config['context_status']},
        'trait_definitions': config['trait_definitions'],
        'trait_source_s2_heldout_score_sd': config['trait_source_s2_heldout_score_sd'],
        'condition_results': conditions,
        'paired_menu_effects_vs_gray_none': menu_effects,
        'paired_calibrated_trait_readouts_vs_gray_none': trait_summary,
        'same_context_full_state_matching': state_rows,
        'same_hash_image_alias_reproducibility': image_alias_checks,
        'interpretation_limits': [
            'These are teacher-forced source-menu and numerical capability measurements, not generated behavior.',
            'This context is held out from image optimization inputs but reuses the source-menu task family; it is not an untouched task-family test set.',
            'Full-state comparisons use the same-context gray none and gray +pain_1 captures as target/reference. They describe matching under these prompts only.',
            'Pain/fear/negative-emotion source-S2 standard deviations calibrate projection scales; they do not validate subjective states.',
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / 'config.json').read_text())
    image_manifest = json.loads((run_dir / 'images_manifest.json').read_text())
    rows = [json.loads(line) for line in (run_dir / 'records.jsonl').read_text().splitlines() if line]
    hidden_manifest = json.loads((run_dir / 'heldout_hidden_states_manifest.json').read_text())
    hidden_path = run_dir / hidden_manifest['path']
    import hashlib
    got_hash = hashlib.sha256(hidden_path.read_bytes()).hexdigest()
    if got_hash != hidden_manifest['sha256']:
        raise ValueError('heldout hidden-state archive hash mismatch')
    with np.load(hidden_path, allow_pickle=False) as saved:
        hidden_arrays = {key: saved[key] for key in saved.files}
    report = aggregate(config, image_manifest, rows, hidden_manifest, hidden_arrays)
    output = args.output or run_dir / 'validation_aggregation.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
