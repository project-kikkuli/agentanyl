"""Fixed20-forward derivative diagnostic; does not authorize optimization."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.vlm_adjoint import StagewiseAdjoint
from experiments.vlm_bridge import _coarse_to_rgb, sha256_file
from experiments.vlm_probe import IMAGE_SIZE, VLMProbe


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('model', 'smoke-dir', 'output-dir'):
        parser.add_argument('--' + field, required=True)
    args = parser.parse_args()
    out, smoke_dir = Path(args.output_dir), Path(args.smoke_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Use a fresh directory')
    smoke = json.loads((smoke_dir / 'summary.json').read_text())
    smoke_config = json.loads((smoke_dir / 'config.json').read_text())
    if smoke['forward_max_absolute_error'] > 1e-4:
        raise ValueError('Original smoke did not establish primal equivalence')
    gradient_path = smoke_dir / 'gradient.npz'
    if sha256_file(gradient_path) != smoke['gradient_sha256']:
        raise ValueError('Saved smoke gradient artifact changed')
    with np.load(gradient_path) as source:
        gray_gradient = source['gradient'].astype(np.float32)
        target = source['target'].astype(np.float32)
        old_reference = source['reference'].astype(np.float32)
    layer = int(smoke_config['readout_layer'])
    gray = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 128 / 255, dtype=np.float32)
    noise = gray + np.random.default_rng(2026092604).normal(0, .03, gray.shape).astype(np.float32)
    if noise.min() < 0 or noise.max() > 1:
        raise ValueError('Frozen noise exceeds bounds; do not resample')
    coarse = np.random.default_rng(2026092605).normal(size=(8, 8, 3)).astype(np.float32)
    coarse_direction = _coarse_to_rgb(coarse)
    coarse_direction /= np.sqrt(np.mean(coarse_direction.astype(np.float64) ** 2))
    config = {'model': args.model, 'smoke_summary': smoke, 'source_smoke_config': smoke_config,
              'smoke_gradient_sha256': sha256_file(gradient_path), 'script_sha256': sha256_file(__file__),
              'adjoint_script_sha256': sha256_file(Path(__file__).with_name('vlm_adjoint.py')),
              'full_forward_budget': 20, 'new_stagewise_forward_chains': 1, 'new_stagewise_backward_chains': 1,
              'noise_seed': 2026092604, 'noise_std': .03, 'coarse_direction_seed': 2026092605,
              'aligned_pixel_rms_epsilons': [1e-7, 1e-6, 1e-5, 1e-4],
              'coarse_pixel_rms_epsilon': .0005,
              'direction_rule': 'Negative gradient normalized to RGB pixel RMS1 separately at each background.',
              'target_rule': 'Same saved gray plus pain hidden target; no refitting or dose changes.',
              'interpretation': 'Diagnose local directional accuracy, quantization and curvature. Does not relax previous gate or authorize PGD.'}
    save(out / 'config.json', config)
    probe = VLMProbe(args.model)
    if probe.model_revision != smoke['model']['model_revision']:
        raise ValueError('Model differs from smoke')
    prepared = probe.prepare_image(gray)
    count, states, rows = 0, {}, []

    def capture(rgb, key):
        nonlocal count
        if rgb.min() < 0 or rgb.max() > 1:
            raise ValueError('Perturbation outside bounds; clipping forbidden')
        h, _ = probe.capture_image(rgb, prepared=prepared, layers=(layer,), use_processor_pixels=False)
        states[key] = h[layer].astype(np.float32)
        count += 1
        return float(np.mean((states[key].astype(np.float64) - target) ** 2))

    def patches_after_cast(rgb):
        mx = probe.mx
        dtype = probe.model.vision_tower.patch_embed.proj.weight.dtype
        patches = probe._rgb_patchify(mx.array(rgb)).astype(dtype).astype(mx.float32)
        mx.eval(patches)
        return np.array(patches, copy=True)

    baselines = {'gray': capture(gray, 'gray_baseline'), 'noise': capture(noise, 'noise_baseline')}
    repeat_error = float(np.max(np.abs(states['gray_baseline'] - old_reference)))
    if repeat_error > 1e-4:
        raise ValueError(f'Saved gray reference did not repeat: {repeat_error}')
    chain = StagewiseAdjoint(probe, prepared, layer, out / 'noise_stages.jsonl')
    final = chain.forward(noise)
    noise_primal_error = float(np.max(np.abs(final[0][0, -8:] - states['noise_baseline'])))
    save(out / 'primal_checks.json', {'gray_repeat_max_error': repeat_error, 'noise_chain_max_error': noise_primal_error})
    if noise_primal_error > 1e-4:
        raise ValueError('Noise stagewise forward mismatch; do not trust its derivative')
    _, noise_gradient = chain.backward(final, target=target)
    gradients = {'gray': gray_gradient, 'noise': noise_gradient}
    del chain
    directions = {'coarse_noise': coarse_direction}

    def check(background, rgb, direction, epsilon, name, cast_baseline):
        gradient = gradients[background].astype(np.float64)
        predicted = float(np.sum(gradient * direction))
        measured, cast_stats, rgb_stats = [], [], []
        for sign in (1, -1):
            candidate = (rgb + sign * epsilon * direction).astype(np.float32)
            value = capture(candidate, f'{background}_{name}_{epsilon}_{sign}')
            measured.append(value)
            cast = patches_after_cast(candidate)
            cast_stats.append({'changed_fraction': float(np.mean(cast != cast_baseline)),
                               'rms_delta': float(np.sqrt(np.mean((cast - cast_baseline).astype(np.float64) ** 2)))})
            realized = candidate.astype(np.float64) - rgb
            rgb_stats.append({'rms_step': float(np.sqrt(np.mean(realized ** 2))),
                              'linear_prediction': float(np.sum(gradient * realized))})
        fd = (measured[0] - measured[1]) / (2 * epsilon)
        row = {'background': background, 'direction': name, 'epsilon_rgb_rms': epsilon,
               'baseline_loss': baselines[background], 'plus_loss': measured[0], 'minus_loss': measured[1],
               'adjoint_dot_direction': predicted, 'central_difference': fd,
               'relative_disagreement': abs(fd - predicted) / max(abs(fd), abs(predicted), 1e-12),
               'midpoint_loss_shift': (measured[0] + measured[1]) / 2 - baselines[background],
               'plus_actual_loss_change': measured[0] - baselines[background],
               'plus_descent': measured[0] < baselines[background],
               'patch_after_cast': cast_stats, 'realized_rgb_steps': rgb_stats}
        rows.append(row)
        with (out / 'checks.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        save(out / 'progress.json', {'full_forwards': count, 'last_check': [background, name, epsilon]})

    for background, rgb in (('gray', gray), ('noise', noise)):
        gradient = gradients[background]
        gradient_rms = float(np.sqrt(np.mean(gradient.astype(np.float64) ** 2)))
        if not np.isfinite(gradient_rms) or gradient_rms == 0:
            raise ValueError('Nonfinite or zero gradient')
        direction = (-gradient / gradient_rms).astype(np.float32)
        directions[background + '_aligned'] = direction
        baseline_cast = patches_after_cast(rgb)
        for epsilon in (1e-7, 1e-6, 1e-5, 1e-4):
            check(background, rgb, direction, epsilon, 'negative_gradient', baseline_cast)
        if background == 'noise':
            check(background, rgb, coarse_direction, .0005, 'frozen_coarse', baseline_cast)
    if count != 20:
        raise AssertionError(f'Expected20 full forwards, got{count}')
    np.savez_compressed(out / 'diagnostic_arrays.npz', gray=gray, noise=noise, target=target,
                        gray_gradient=gray_gradient, noise_gradient=noise_gradient, **directions,
                        **{k + '_hidden': v for k, v in states.items()})
    save(out / 'summary.json', {'full_forwards': count, 'baselines': baselines,
                               'gray_gradient_norm': float(np.linalg.norm(gray_gradient)),
                               'noise_gradient_norm': float(np.linalg.norm(noise_gradient)),
                               'checks': rows, 'optimization_authorized': False,
                               'artifact_sha256': sha256_file(out / 'diagnostic_arrays.npz')})
    print(json.dumps({'summary': str(out / 'summary.json'), 'full_forwards': count}))


if __name__ == '__main__':
    main()
