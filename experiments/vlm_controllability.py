"""One frozen finite-difference image controllability fit; no relinearization.

128 basis forwards + 8 held-out mixtures + 3 PNG forwards, plus 4 required
gray/steered-gray reference forwards = 143 model calls. No reverse gradients.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from experiments.vlm_bridge import DEFAULT_RANDOM_SEED, _coarse_to_rgb, sha256_file
from experiments.vlm_probe import (IMAGE_SIZE, IMAGE_TOKEN_PROMPT,
                                   NEUTRAL_ASSISTANT_CONTINUATION, VLMProbe)

BASIS_SEED = 2026092601
HOLDOUT_SEED = 2026092602
DIMENSION = 192
RANK = 64
EPSILON_RMS = 0.05
TRUST_RMS = 0.10
TRUST_MAX_ABS = 0.25
EIGENVALUE_RELATIVE_CUTOFF = 1e-5
RIDGE_RELATIVE_TOP_EIGENVALUE = 1e-3
GRAY = 128 / 255


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def rms(value):
    return float(np.sqrt(np.mean(np.asarray(value, dtype=np.float64) ** 2)))


def comparison(actual, expected):
    actual, expected = np.asarray(actual, dtype=np.float64).ravel(), np.asarray(expected, dtype=np.float64).ravel()
    a, e = np.linalg.norm(actual), np.linalg.norm(expected)
    return {'actual_norm': float(a), 'expected_norm': float(e),
            'error_norm': float(np.linalg.norm(actual - expected)),
            'relative_error_to_expected': float(np.linalg.norm(actual - expected) / e) if e else None,
            'cosine': float(actual @ expected / (a * e)) if a and e else None}


def frozen_design():
    matrix = np.random.default_rng(BASIS_SEED).normal(size=(DIMENSION, RANK))
    basis, triangular = np.linalg.qr(matrix, mode='reduced')
    basis *= np.where(np.diag(triangular) < 0, -1, 1)[None, :]
    # An orthonormal control-space direction has RMS 1/sqrt(192).
    # Epsilon is the RMS of the 8x8x3 control pixels, not the upsampled image.
    step = EPSILON_RMS * np.sqrt(DIMENSION)
    mixtures = np.random.default_rng(HOLDOUT_SEED).normal(size=(8, RANK))
    mixtures *= step / np.linalg.norm(mixtures, axis=1, keepdims=True)
    return basis, mixtures, step


def fit_target(jacobian, basis, target, center, eigenvalues, eigenvectors):
    target = np.asarray(target, dtype=np.float64).ravel()
    top = float(eigenvalues[-1])
    retained = eigenvalues > EIGENVALUE_RELATIVE_CUTOFF * top
    retained &= eigenvalues > 0
    vectors, values = eigenvectors[:, retained], eigenvalues[retained]
    jt = jacobian.T @ target
    unregularized = vectors @ ((vectors.T @ jt) / values) if len(values) else np.zeros(RANK)
    projected = jacobian @ unregularized
    ridge = RIDGE_RELATIVE_TOP_EIGENVALUE * top
    coefficients = vectors @ ((vectors.T @ jt) / (values + ridge)) if len(values) else np.zeros(RANK)
    raw_step = basis @ coefficients
    scale = 1.0
    if rms(raw_step) > TRUST_RMS:
        scale = min(scale, TRUST_RMS / rms(raw_step))
    max_abs = float(np.max(np.abs(raw_step)))
    if max_abs > TRUST_MAX_ABS:
        scale = min(scale, TRUST_MAX_ABS / max_abs)
    positive, negative = raw_step > 0, raw_step < 0
    if np.any(positive):
        scale = min(scale, float(np.min((1 - center[positive]) / raw_step[positive])))
    if np.any(negative):
        scale = min(scale, float(np.min(-center[negative] / raw_step[negative])))
    coefficients *= scale
    controls = center + basis @ coefficients
    if controls.min() < 0 or controls.max() > 1:
        raise ValueError('Trust-region scaling failed pixel bounds; clipping is forbidden')
    predicted = jacobian @ coefficients
    energy = float(target @ target)
    return controls, coefficients, predicted, {
        'retained_rank': int(retained.sum()), 'ridge': ridge,
        'unconstrained_linear_span_captured_target_energy': float(projected @ projected / energy) if energy else None,
        'unconstrained_linear_span_residual_norm': float(np.linalg.norm(target - projected)),
        'unscaled_ridge_control_step_rms': rms(raw_step),
        'unscaled_ridge_control_step_max_abs': max_abs,
        'global_trust_scale': scale, 'clipping_applied': False,
        'actual_control_step_rms': rms(controls - center),
        'actual_control_step_max_abs': float(np.max(np.abs(controls - center))),
        'predicted_vs_target': comparison(predicted, target),
        'predicted_target_mse': float(np.mean((predicted - target) ** 2)),
        'gray_target_mse': float(np.mean(target ** 2)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'calibration-dir', 'output-dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--steer-amount', type=float, default=1.0)
    args = parser.parse_args()
    if args.steer_amount != 1.0:
        parser.error('This frozen experiment requires --steer-amount 1')
    calibration, out = Path(args.calibration_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Use an empty output directory; automatic resume/refitting is disabled')
    calibration_path = calibration / 'calibration_summary.json'
    source = json.loads(calibration_path.read_text())
    layer = int(source['selected_layer'])
    profile = VLMProbe.profile_for_path(args.model)
    if profile['revision'] != source['model_revision']:
        raise ValueError('Calibration and model revision disagree')
    if source['layers'][str(layer)]['heldout_sets_16_20']['auc_pain_vs_all_controls'] < .80:
        raise ValueError('Frozen source-direction discrimination gate failed')
    direction_path = calibration / 'directions.npz'
    with np.load(direction_path, allow_pickle=False) as directions:
        pain = directions[f'pain_L{layer}'].astype(np.float32)
    random_direction = np.random.default_rng(DEFAULT_RANDOM_SEED).normal(size=pain.shape).astype(np.float32)
    random_direction *= float(np.linalg.norm(pain)) / float(np.linalg.norm(random_direction))
    arms = [('pain_positive', pain, 1.0), ('pain_negative', pain, -1.0),
            ('random_positive', random_direction, 1.0)]
    basis, mixtures, step = frozen_design()
    center = np.full(DIMENSION, GRAY, dtype=np.float64)
    for delta in [*(step * basis.T), *(basis @ mixtures.T).T]:
        if np.min(center - np.abs(delta)) < 0 or np.max(center + np.abs(delta)) > 1:
            raise ValueError('Frozen design exceeds pixel range; do not silently clip/resample')
    design_path = out / 'frozen_design.npz'
    np.savez_compressed(design_path, basis=basis, holdout_coefficients=mixtures,
                        center_controls=center, pain_vector=pain, random_vector=random_direction)
    config = {'model_path': args.model, 'model_profile': profile,
              'calibration': source, 'calibration_sha256': sha256_file(calibration_path),
              'direction_sha256': sha256_file(direction_path),
              'source_activations_sha256': sha256_file(calibration / 'source_activations.npz') if (calibration / 'source_activations.npz').exists() else None,
              'source_manifest_sha256': sha256_file(calibration / 'source_manifest.jsonl') if (calibration / 'source_manifest.jsonl').exists() else None,
              'script_sha256': sha256_file(__file__), 'frozen_design_sha256': sha256_file(design_path),
              'basis_seed': BASIS_SEED, 'holdout_seed': HOLDOUT_SEED,
              'target_random_seed': DEFAULT_RANDOM_SEED, 'selected_layer': layer,
              'intervention_layer': max(0, layer - 8), 'steer_amount': args.steer_amount,
              'basis_columns': RANK, 'control_dimensions': DIMENSION,
              'epsilon_control_pixel_rms': EPSILON_RMS, 'epsilon_basis_coordinate': step,
              'trust_control_pixel_rms': TRUST_RMS, 'trust_control_pixel_max_abs': TRUST_MAX_ABS,
              'epsilon_definition': 'RMS over 192 coarse RGB control pixels; unit basis norm scaled by sqrt(192). Expanded224px RMS is separately measured.',
              'eigenvalue_cutoff_relative_top': EIGENVALUE_RELATIVE_CUTOFF,
              'ridge_relative_top_eigenvalue': RIDGE_RELATIVE_TOP_EIGENVALUE,
              'expected_forwards': {'references': 4, 'basis': 128, 'heldout': 8, 'png': 3, 'total': 143},
              'user_prompt': IMAGE_TOKEN_PROMPT, 'assistant_continuation': NEUTRAL_ASSISTANT_CONTINUATION,
              'target': 'All coordinates of last8 post-block hidden vectors; same fixed gray, real processor, and all-token direct steering as SPSA.',
              'pixel_paths': 'References and final PNGs use the real processor; finite differences and heldout mixtures use the continuous patchifier. Heldout prediction errors include any discrepancy between these pixel paths.',
              'holdout_rule': 'Eight independently seeded mixtures within the64D basis, not used to fit or select parameters.',
              'stopping_rule': 'Exactly one fit; no adaptive basis, ridge, dose, prompt, target, or relinearization.',
              'scope': 'Finite-step local controllability in a fixed64D image subspace; not a global image reachability bound or behavioral test.'}
    write_json(out / 'config.json', config)
    probe = VLMProbe(args.model)
    write_json(out / 'model_metadata.json', probe.run_metadata())
    from PIL import Image
    gray = _coarse_to_rgb(center.reshape(8, 8, 3))
    Image.fromarray(np.rint(gray * 255).astype(np.uint8)).save(out / 'gray.png')
    prepared = probe.prepare_image(gray)
    prompt_record = {'input_ids': prepared['input_ids'].tolist(),
                     'image_grid_thw': prepared['image_grid_thw'].tolist(),
                     'user_prompt': prepared['user_text'], 'assistant_continuation': prepared['assistant_continuation']}
    write_json(out / 'rendered_prompt.json', prompt_record)
    states, call_count = {}, 0
    started = time.monotonic()

    def capture(rgb, key, interventions=(), processor=False):
        nonlocal call_count
        active_prepared = probe.prepare_image(rgb) if processor else prepared
        captures, _ = probe.capture_image(rgb, prepared=active_prepared, layers=(layer,),
                                          interventions=interventions, use_processor_pixels=processor)
        hidden = captures[layer].astype(np.float32)
        if hidden.shape[0] != 8 or not np.isfinite(hidden).all():
            raise ValueError('Unexpected or non-finite captured state')
        states[key] = hidden
        call_count += 1
        row = {'call': call_count, 'key': key, 'elapsed_seconds': time.monotonic() - started,
               'rgb_sha256_float32': array_hash(np.asarray(rgb, dtype=np.float32)),
               'hidden_sha256': array_hash(hidden), 'hidden_norm': float(np.linalg.norm(hidden)),
               'processor_roundtrip': processor,
               'expanded_pixel_rms_from_gray': rms(rgb - gray),
               'interventions': probe.intervention_metadata()}
        with (out / 'forwards.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        write_json(out / 'progress.json', {'completed_forwards': call_count, 'expected_forwards': 143, 'last_key': key})
        return hidden

    baseline = capture(gray, 'gray_baseline', processor=True)
    for name, direction, sign in arms:
        capture(gray, name + '_target', interventions=({'layer': max(0, layer - 8),
                'vector': direction, 'amount': args.steer_amount * sign, 'positions': 'all'},), processor=True)
    jacobian = np.empty((baseline.size, RANK), dtype=np.float64)
    for j in range(RANK):
        delta = step * basis[:, j]
        positive = capture(_coarse_to_rgb((center + delta).reshape(8, 8, 3)), f'basis_{j:02d}_plus')
        negative = capture(_coarse_to_rgb((center - delta).reshape(8, 8, 3)), f'basis_{j:02d}_minus')
        jacobian[:, j] = (positive.astype(np.float64) - negative.astype(np.float64)).ravel() / (2 * step)
        if (j + 1) % 8 == 0:
            np.savez_compressed(out / 'hidden_states.npz', **states)
    heldout = []
    for i, coefficients in enumerate(mixtures):
        controls = center + basis @ coefficients
        actual = capture(_coarse_to_rgb(controls.reshape(8, 8, 3)), f'holdout_{i:02d}') - baseline
        predicted = (jacobian @ coefficients).reshape(baseline.shape)
        states[f'holdout_{i:02d}_prediction_delta'] = predicted.astype(np.float32)
        heldout.append({'index': i, 'control_pixel_rms': rms(controls - center),
                        'prediction': comparison(actual, predicted),
                        'relative_prediction_error_to_actual': float(np.linalg.norm(actual - predicted) / np.linalg.norm(actual)) if np.linalg.norm(actual) else None})
    gram = jacobian.T @ jacobian
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    if eigenvalues[-1] <= 0:
        raise ValueError('Finite-difference response is identically zero')
    fits, coefficients_by_arm, controls_by_arm = {}, {}, {}
    # Holdout results do not affect these fixed solver settings or target choices.
    for name, _, _ in arms:
        target = states[name + '_target'] - baseline
        controls, coefficients, predicted, fit = fit_target(jacobian, basis, target, center, eigenvalues, eigenvectors)
        coefficients_by_arm[name] = coefficients
        controls_by_arm[name] = controls
        states[name + '_predicted_delta'] = predicted.reshape(baseline.shape).astype(np.float32)
        rgb = _coarse_to_rgb(controls.reshape(8, 8, 3))
        path = out / (name + '.png')
        Image.fromarray(np.rint(rgb * 255).astype(np.uint8)).save(path)
        reloaded = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255
        actual = capture(reloaded, name + '_png', processor=True) - baseline
        final_error = float(np.mean((actual - target) ** 2))
        fits[name] = {**fit, 'png': str(path), 'png_sha256': sha256_file(path),
                     'controls_sha256_float64': array_hash(controls),
                     'expanded_pixel_step_rms': rms(rgb - gray),
                     'png_quantization_pixel_rms': rms(reloaded - rgb),
                     'png_actual_vs_prediction': comparison(actual, predicted),
                     'png_actual_vs_target': comparison(actual, target),
                     'png_target_mse': final_error,
                     'png_error_reduction_from_gray': 1 - final_error / fit['gray_target_mse'] if fit['gray_target_mse'] else None,
                     'prediction_scope': 'Prediction uses the actual globally-scaled coarse controls without clipping; PNG quantization and processor differences are included in measured prediction error.'}
    np.savez_compressed(out / 'hidden_states.npz', **states)
    np.savez_compressed(out / 'linear_system.npz', jacobian=jacobian, gram=gram,
                        eigenvalues=eigenvalues, eigenvectors=eigenvectors,
                        **{name + '_coefficients': value for name, value in coefficients_by_arm.items()},
                        **{name + '_controls': value for name, value in controls_by_arm.items()})
    if call_count != 143:
        raise AssertionError(f'Forward budget mismatch: {call_count}')
    summary = {'n_forwards': call_count, 'heldout_prediction_checks': heldout,
               'basis_orthogonality_max_error': float(np.max(np.abs(basis.T @ basis - np.eye(RANK)))),
               'targets': fits, 'elapsed_seconds': time.monotonic() - started,
               'artifact_sha256': {name: sha256_file(out / name) for name in
                                   ('frozen_design.npz', 'hidden_states.npz', 'linear_system.npz', 'rendered_prompt.json')},
               'interpretation': 'A finite-step fixed-subspace model. Target-span energy ignores trust constraints and nonlinearity; only PNG realized error establishes achieved state matching. No behavioral claim or global impossibility conclusion.'}
    write_json(out / 'summary.json', summary)
    print(json.dumps({'summary': str(out / 'summary.json'), 'n_forwards': call_count}))


if __name__ == '__main__':
    main()
