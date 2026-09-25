"""Saved-activation audit only; no model loading or image selection."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.vlm_bridge import PAIN_CATEGORIES, CONTROL_CATEGORIES, auc, fit_direction, sha256_file


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration-dir', required=True)
    parser.add_argument('--search-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    calibration, search = Path(args.calibration_dir), Path(args.search_dir)
    meta = json.loads((calibration / 'calibration_summary.json').read_text())
    layer = meta['selected_layer']
    rows = [json.loads(line) for line in (calibration / 'source_manifest.jsonl').read_text().splitlines()]
    rows = [row for row in rows if row['group'] == 'S2_1P']
    source = np.load(calibration / 'source_activations.npz')[f'S2_1P_L{layer}'].astype(np.float64)
    categories = np.array([r['category'] for r in rows])
    train = np.array([r['set'] <= 10 for r in rows])
    test = np.array([r['set'] >= 16 for r in rows])
    directions = np.load(calibration / 'directions.npz')
    names = ('pain', 'fear', 'negative_emotion')
    vectors = {name: directions[f'{name}_L{layer}'].astype(np.float64) for name in names}
    contrasts = {'pain': (PAIN_CATEGORIES, CONTROL_CATEGORIES, CONTROL_CATEGORIES),
                 'fear': (('B',), ('D',), ('D',)), 'negative_emotion': (('C1',), ('D',), ('D',))}
    scales, traits = {}, {}
    for name, (positive, negative, denoise) in contrasts.items():
        scores = source @ vectors[name]
        subset = test & (np.isin(categories, positive) | np.isin(categories, negative))
        labels = np.isin(categories[subset], positive)
        refit = fit_direction(source[test], categories[test], positive, negative, denoise,
                              neutral_only=name != 'pain')['vector'].astype(np.float64)
        scales[name] = float(scores[test].std(ddof=1))
        traits[name] = {'heldout_intended_contrast_auc': auc(scores[subset], labels),
                        'heldout_n_positive': int(labels.sum()), 'heldout_n_negative': int((~labels).sum()),
                        'heldout_pain_vs_all_controls_auc': auc(scores[test], np.isin(categories[test], PAIN_CATEGORIES)),
                        'heldout_all50_score_sd_ddof1': scales[name],
                        'heldout_trait_pair_score_sd_ddof1': float(scores[subset].std(ddof=1)),
                        'train_axis_vs_heldout_refit_cosine': cosine(vectors[name], refit)}
    calibration_audit = {'axis_order': names, 'traits': traits,
                         'frozen_axis_cosines': [[cosine(vectors[a], vectors[b]) for b in names] for a in names],
                         'frozen_axis_score_correlations_train': np.corrcoef([source[train] @ vectors[n] for n in names]).tolist(),
                         'frozen_axis_score_correlations_heldout': np.corrcoef([source[test] @ vectors[n] for n in names]).tolist(),
                         'caveat': 'Comparator pain-vs-all-control AUC is not its named trait discrimination. Intended fear/negative contrasts have only5 positive and5 neutral heldout cases. Heldout-refit directions are posthoc descriptive diagnostics and never used for image selection.'}
    result = {'calibration': calibration_audit, 'search_complete': (search / 'summary.json').exists(),
              'source_activations_sha256': sha256_file(calibration / 'source_activations.npz'),
              'directions_sha256': sha256_file(calibration / 'directions.npz')}
    if result['search_complete']:
        summary = json.loads((search / 'summary.json').read_text())
        states = np.load(search / 'hidden_states.npz')
        baseline = states['gray_baseline'].astype(np.float64)
        arms = {}
        for key, row in summary['arms'].items():
            mode, arm = key.split('/')
            hidden = states[f'{mode}_{arm}_selected'].astype(np.float64)
            target = states[f'{arm}_target'].astype(np.float64)
            delta = hidden - baseline
            readouts = {}
            for axis in names:
                vector = vectors[axis]
                readouts[axis] = {
                    'last_token_raw_score': float(hidden[-1] @ vector),
                    'gray_last_token_raw_score': float(baseline[-1] @ vector),
                    'last_token_shift_source_sd': float(delta[-1] @ vector / scales[axis]),
                    'mean_last8_shift_source_sd': float(np.mean(delta @ vector) / scales[axis]),
                    'direct_target_last_token_shift_source_sd': float((target[-1] - baseline[-1]) @ vector / scales[axis])}
            arms[key] = {'png_sha256': row['png_sha256'], 'accepted_iterations': row['accepted_iterations'],
                         'full_target_error_reduction_from_gray': row['target_mse_reduction_from_gray'],
                         'delta_cosine_to_target': row['delta_cosine_to_target'], 'readouts': readouts}
        result['arms'] = arms
        result['search_summary_sha256'] = sha256_file(search / 'summary.json')
        result['interpretation'] = 'Source-SD shifts standardize against each axis on the SAME50 heldout source prompts. Last8 mean uses that reference scale descriptively; source calibration itself uses final tokens. Shared shifts do not establish pain specificity or motivation.'
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': args.output, 'search_complete': result['search_complete']}))


if __name__ == '__main__':
    main()
