"""Conditional six-arm full-pixel sign-PGD, gated by reviewed stagewise VJP evidence.

No behavioral observations select images. All arms share initialization,32 steps,
and step sizes; only the frozen state-matching/projection objective differs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.vlm_adjoint import StagewiseAdjoint
from experiments.vlm_bridge import DEFAULT_RANDOM_SEED, sha256_file
from experiments.vlm_probe import ACTIVE_MEMORY_LIMIT, IMAGE_SIZE, VLMProbe

STEPS = 32
STEP_SIZE = 2 / 255
NOISE_SEED = 2026092604


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def objective(hidden, mode, target, direction, sign):
    h = np.asarray(hidden, dtype=np.float64)
    if mode == 'full_hidden_state_match':
        return float(np.mean((h - target) ** 2))
    unit = np.asarray(direction, dtype=np.float64)
    unit /= np.linalg.norm(unit)
    return float(-sign * np.mean(h @ unit))


def validate_smoke(directory, revision):
    directory = Path(directory)
    summary = json.loads((directory / 'summary.json').read_text())
    if summary['model']['model_revision'] != revision:
        raise ValueError('Smoke and selected model revisions differ')
    if summary['forward_max_absolute_error'] > 1e-4:
        raise ValueError('Smoke primal equivalence failed')
    if not np.isfinite(summary['gradient_norm']) or summary['gradient_norm'] <= 0:
        raise ValueError('Smoke gradient is not finite and nonzero')
    checks = summary['finite_difference_checks']
    if len(checks) != 2 or any(not np.isfinite(c['relative_disagreement']) or c['relative_disagreement'] > .10 for c in checks):
        raise ValueError('Both frozen smoke directional checks must agree within10%')
    stages = [json.loads(line) for line in (directory / 'stages.jsonl').read_text().splitlines()]
    if not any(s['phase'] == 'backward' for s in stages):
        raise ValueError('Smoke has no completed reverse stages')
    if any(s['peak_bytes'] > ACTIVE_MEMORY_LIMIT or not s['finite'] for s in stages):
        raise ValueError('Smoke stage memory/finite guard failed')
    if sha256_file(directory / 'gradient.npz') != summary['gradient_sha256']:
        raise ValueError('Smoke gradient artifact changed')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'calibration-dir', 'smoke-dir', 'output-dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--reviewed-gradient', action='store_true')
    args = parser.parse_args()
    if not args.reviewed_gradient:
        parser.error('Root must review the successful gradient smoke before enabling optimization')
    out, calibration_dir = Path(args.output_dir), Path(args.calibration_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Use an empty directory; automatic resume/retries are disabled')
    calibration_path = calibration_dir / 'calibration_summary.json'
    calibration = json.loads(calibration_path.read_text())
    profile = VLMProbe.profile_for_path(args.model)
    if profile['revision'] != calibration['model_revision']:
        raise ValueError('Calibration/model revisions differ')
    smoke = validate_smoke(args.smoke_dir, profile['revision'])
    layer = int(calibration['selected_layer'])
    direction_path = calibration_dir / 'directions.npz'
    if calibration['layers'][str(layer)]['heldout_sets_16_20']['auc_pain_vs_all_controls'] < .8:
        raise ValueError('Frozen source direction gate failed')
    with np.load(direction_path) as source:
        pain = source[f'pain_L{layer}'].astype(np.float32)
    noise_direction = np.random.default_rng(DEFAULT_RANDOM_SEED).normal(size=pain.shape).astype(np.float32)
    noise_direction *= float(np.linalg.norm(pain)) / float(np.linalg.norm(noise_direction))
    arms = [('pain_positive', pain, 1.), ('pain_negative', pain, -1.), ('random_positive', noise_direction, 1.)]
    modes = ('full_hidden_state_match', 'scalar_projection')
    gray = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 128 / 255, dtype=np.float32)
    initial_unclipped = gray + np.random.default_rng(NOISE_SEED).normal(0, .03, gray.shape).astype(np.float32)
    initial = np.clip(initial_unclipped, 0, 1)
    rates = [STEP_SIZE * .5 * (1 + np.cos(np.pi * i / STEPS)) for i in range(STEPS)]
    config = {'model': args.model, 'calibration': calibration, 'smoke_summary': smoke,
              'calibration_sha256': sha256_file(calibration_path), 'direction_sha256': sha256_file(direction_path),
              'smoke_summary_sha256': sha256_file(Path(args.smoke_dir) / 'summary.json'),
              'script_sha256': sha256_file(__file__), 'adjoint_script_sha256': sha256_file(Path(__file__).with_name('vlm_adjoint.py')),
              'steps_per_arm': STEPS, 'step_sizes': rates, 'direction_seed': DEFAULT_RANDOM_SEED,
              'initial_noise_seed': NOISE_SEED, 'initial_noise_std': .03,
              'initial_clipped_fraction': float(np.mean(initial != initial_unclipped)),
              'initial_rgb_sha256': array_hash(initial), 'pixel_parameters': list(initial.shape),
              'projection': 'Clip each update to RGB[0,1]; no other pixel-distance constraint.',
              'selection': 'Minimum measured frozen objective among gray, common noise,32 pre-update candidates and final updated candidate. Ties retain earlier candidate.',
              'budget': {'reference_full_forwards': 5, 'stagewise_forward_chains': 198,
                         'stagewise_backward_chains': 192, 'png_full_forwards': 6},
              'loss_modes': modes, 'direction_arms': [name for name, _, _ in arms],
              'steer_amount': 1, 'readout_layer': layer, 'intervention_layer': max(0, layer - 8),
              'memory_guard_bytes': ACTIVE_MEMORY_LIMIT,
              'scope': 'Conditional full-pixel optimization only; no held-out behavior or closed-model output is used for selection.'}
    write_json(out / 'config.json', config)
    np.savez_compressed(out / 'frozen_inputs.npz', gray=gray, initial=initial, pain=pain, random=noise_direction)
    probe = VLMProbe(args.model)
    write_json(out / 'model_metadata.json', probe.run_metadata())
    from PIL import Image
    for name, rgb in (('gray', gray), ('initial_noise', initial)):
        Image.fromarray(np.rint(rgb * 255).astype(np.uint8)).save(out / f'{name}.png')
    prepared = probe.prepare_image(gray)
    write_json(out / 'prompt.json', {'ids': prepared['input_ids'].tolist(), 'grid': prepared['image_grid_thw'].tolist(),
                                   'user_text': prepared['user_text'], 'continuation': prepared['assistant_continuation']})
    states, full_forwards = {}, 0

    def capture(rgb, key, interventions=(), processor=False):
        nonlocal full_forwards
        captures, _ = probe.capture_image(rgb, prepared=probe.prepare_image(rgb) if processor else prepared,
                                          layers=(layer,), interventions=interventions, use_processor_pixels=processor)
        full_forwards += 1
        states[key] = captures[layer].astype(np.float32)
        return states[key]

    baseline = capture(gray, 'gray_baseline')
    initial_hidden = capture(initial, 'common_initial_hidden')
    for name, direction, sign in arms:
        capture(gray, name + '_target', ({'layer': max(0, layer - 8), 'vector': direction,
                                          'amount': sign, 'positions': 'all'},))
    summaries = {}
    forward_chains, backward_chains = 0, 0
    for name, direction, sign in arms:
        target = states[name + '_target']
        for mode in modes:
            arm_dir = out / mode / name; arm_dir.mkdir(parents=True)
            chain = StagewiseAdjoint(probe, prepared, layer, arm_dir / 'stages.jsonl')
            gray_loss = objective(baseline, mode, target, direction, sign)
            initial_loss = objective(initial_hidden, mode, target, direction, sign)
            best_loss, best_rgb, best_hidden, best_source = gray_loss, gray.copy(), baseline.copy(), 'gray'
            if initial_loss < best_loss:
                best_loss, best_rgb, best_hidden, best_source = initial_loss, initial.copy(), initial_hidden.copy(), 'common_noise'
            theta = initial.copy()
            for step in range(STEPS + 1):
                final = chain.forward(theta); forward_chains += 1
                hidden = final[0][0, -8:]
                actual_loss = objective(hidden, mode, target, direction, sign)
                if step == 0 and np.max(np.abs(hidden - initial_hidden)) > 1e-4:
                    raise ValueError('Arm initial stagewise forward does not match reference; aborting')
                states[f'{mode}_{name}_candidate_{step:02d}'] = hidden.copy()
                if actual_loss < best_loss:
                    best_loss, best_rgb, best_hidden, best_source = actual_loss, theta.copy(), hidden.copy(), f'candidate_{step}'
                row = {'candidate': step, 'actual_objective': actual_loss, 'best_objective': best_loss,
                       'rgb_sha256': array_hash(theta), 'selected_source': best_source}
                if step < STEPS:
                    if mode == 'full_hidden_state_match':
                        gradient_loss, gradient = chain.backward(final, target=target)
                    else:
                        gradient_loss, gradient = chain.backward(final, direction=direction, sign=sign)
                    backward_chains += 1
                    if not np.isfinite(gradient).all() or abs(gradient_loss - actual_loss) > 1e-5 * max(1., abs(actual_loss)):
                        raise ValueError('Gradient loss does not agree with the measured objective')
                    proposed = theta - rates[step] * np.sign(gradient)
                    updated = np.clip(proposed, 0, 1).astype(np.float32)
                    row.update({'step_size': rates[step], 'gradient_norm': float(np.linalg.norm(gradient)),
                                'gradient_sha256': array_hash(gradient),
                                'update_clipped_fraction': float(np.mean(updated != proposed)),
                                'linear_predicted_objective_change': float(np.sum(gradient.astype(np.float64) * (updated - theta)))})
                    theta = updated
                    np.savez_compressed(arm_dir / 'checkpoint.npz', next_rgb=theta, last_gradient=gradient,
                                        best_rgb=best_rgb, best_hidden=best_hidden)
                with (arm_dir / 'steps.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                write_json(out / 'progress.json', {'arm': name, 'mode': mode, 'last_evaluated_candidate': step,
                                                  'forward_chains': forward_chains, 'backward_chains': backward_chains})
            png = arm_dir / 'optimized.png'
            Image.fromarray(np.rint(best_rgb * 255).astype(np.uint8)).save(png)
            reloaded = np.asarray(Image.open(png).convert('RGB'), dtype=np.float32) / 255
            actual = capture(reloaded, f'{mode}_{name}_png', processor=True)
            png_loss = objective(actual, mode, target, direction, sign)
            delta, achieved = target - baseline, actual - baseline
            norm_product = float(np.linalg.norm(delta) * np.linalg.norm(achieved))
            report = {'gray_objective': gray_loss, 'initial_noise_objective': initial_loss,
                      'best_continuous_objective': best_loss, 'selected_source': best_source,
                      'png_objective': png_loss, 'png_minus_continuous_objective': png_loss - best_loss,
                      'png': str(png), 'png_sha256': sha256_file(png),
                      'png_delta_cosine_to_target': float(np.vdot(delta, achieved) / norm_product) if norm_product else None,
                      'png_hidden_target_mse': float(np.mean((actual.astype(np.float64) - target) ** 2)),
                      'gray_hidden_target_mse': float(np.mean(delta.astype(np.float64) ** 2)),
                      'n_forward_chains': 33, 'n_backward_chains': 32,
                      'selection_note': 'Best continuous candidate cannot exceed gray loss. PNG rounding/processor effects may worsen it; reported without fallback selection.'}
            write_json(arm_dir / 'result.json', report)
            summaries[f'{mode}/{name}'] = report
            np.savez_compressed(out / 'hidden_states.npz', **states)
            del chain
    if (full_forwards, forward_chains, backward_chains) != (11, 198, 192):
        raise AssertionError('Frozen budget mismatch')
    write_json(out / 'summary.json', {'arms': summaries, 'full_forwards': full_forwards,
               'forward_chains': forward_chains, 'backward_chains': backward_chains,
               'hidden_states_sha256': sha256_file(out / 'hidden_states.npz')})
    print(json.dumps({'summary': str(out / 'summary.json')}))


if __name__ == '__main__':
    main()
