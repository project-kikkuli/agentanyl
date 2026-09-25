"""Single guarded stagewise VJP smoke test; never builds a full-model backward tape."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from experiments.vlm_bridge import _coarse_to_rgb, sha256_file
from experiments.vlm_probe import ACTIVE_MEMORY_LIMIT, IMAGE_SIZE, PATCH_SIZE, TEMPORAL_PATCH_SIZE, VLMProbe


def manual_patch_adjoint(cotangent, weight, window_index, image_size=IMAGE_SIZE,
                         patch_size=PATCH_SIZE, temporal_size=TEMPORAL_PATCH_SIZE):
    """Analytic adjoint of nonoverlapping Conv3D, patch ordering and RGB normalize.

    MLX Conv3D weights are [out,T,p,p,C]. Serialized input patches are
    [C,T,p,p]. There is no Conv3D-transpose or scatter-autodiff invocation.
    Floating-point derivative arithmetic is deliberately float32 NumPy.
    """
    weight = np.asarray(weight, dtype=np.float32)
    if weight.shape[1:] != (temporal_size, patch_size, patch_size, 3):
        raise ValueError(f'Unexpected patch convolution layout: {weight.shape}')
    grid = image_size // patch_size
    if image_size % patch_size or grid % 2:
        raise ValueError('Expected an even spatial patch grid')
    order = np.asarray(window_index, dtype=np.int64)
    if not np.array_equal(np.sort(order), np.arange(grid * grid // 4)):
        raise ValueError('Window index is not the expected permutation of4-patch groups')
    grad_features = np.asarray(cotangent, dtype=np.float32).reshape(grid * grid // 4, 4, -1)
    grad_features = grad_features[np.argsort(order)].reshape(grid * grid, weight.shape[0])
    weight_flat = weight.transpose(0, 4, 1, 2, 3).reshape(weight.shape[0], -1)
    grad_patches = grad_features @ weight_flat
    # Invert _rgb_patchify's reshape/transpose, then sum the repeated frames.
    perm = (0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    ordered = grad_patches.reshape(1, 1, grid // 2, grid // 2, 2, 2, 3, temporal_size, patch_size, patch_size)
    temporal = ordered.transpose(np.argsort(perm)).reshape(1, 1, temporal_size, 3, image_size, image_size)
    grad_rgb = temporal.sum(axis=(0, 1, 2)).transpose(1, 2, 0)
    return grad_rgb / np.array([.26862954, .26130258, .27577711], dtype=np.float32)


class StagewiseAdjoint:
    def __init__(self, probe, prepared, layer, log_path, patch_offset=None):
        self.probe, self.mx = probe, probe.mx
        self.patch_offset = patch_offset
        self.log_path = Path(log_path)
        self.layer = layer
        self.stages = self._stages(prepared)
        self.inputs = []

    def _stages(self, prepared):
        from mlx_vlm.models.base import create_attention_mask
        mx, model = self.mx, self.probe.model
        vision, language = model.vision_tower, model.language_model.model
        ids, grid = prepared['input_ids'], prepared['image_grid_thw']
        window, cumulative = vision.get_window_index(grid)
        mx.eval(window, cumulative)
        mx.eval(vision.patch_embed.proj.weight)
        self.patch_weight = np.array(vision.patch_embed.proj.weight.astype(mx.float32), copy=True)
        self.window_index = np.asarray(window, dtype=np.int64).copy()
        if vision.spatial_merge_unit != 4:
            raise ValueError('Manual patch adjoint requires the validated4-patch merge unit')
        cu = cumulative.tolist()
        keep = [i for i, x in enumerate(cu) if x not in cu[:i]]
        cumulative = cumulative[mx.array(keep, dtype=mx.int32)]
        cells = [h * w for t, h, w in grid.tolist() for _ in range(t)]
        full = mx.array(np.concatenate(([0], np.cumsum(cells))).astype(np.int32))
        count = sum(cells)
        merge = vision.spatial_merge_unit
        rotary = vision.rot_pos_emb(grid).reshape(count // merge, merge, -1)[window].reshape(count, -1)
        reverse = mx.argsort(window, axis=0)
        text = language.embed_tokens(ids)
        positions, _ = model.language_model.get_rope_index(ids, grid, None, None)
        mask = create_attention_mask(text, None)
        mx.eval(rotary, cumulative, full, reverse, text, positions)

        def pixels_to_windows(rgb):
            patches = self.probe._rgb_patchify(rgb)
            if self.patch_offset is not None:
                patches = patches + mx.array(self.patch_offset, dtype=mx.float32)
            patches = patches.astype(vision.patch_embed.proj.weight.dtype)
            h = vision.patch_embed(patches)
            return h.reshape(count // merge, merge, -1)[window].reshape(count, -1)

        stages = [('pixels_patch_embed_window', pixels_to_windows)]
        for i, block in enumerate(vision.blocks):
            cu_now = full if i in vision.fullatt_block_indexes else cumulative
            stages.append((f'vision_{i}', lambda h, b=block, c=cu_now: b(h, cu_seqlens=c, rotary_pos_emb=rotary)))

        def merge_to_text(h):
            features = vision.merger(h)[reverse, :]
            return model.merge_input_ids_with_image_features(model.config.image_token_id,
                       model.config.video_token_id, features, text, ids)

        stages.append(('merger_inverse_window_scatter', merge_to_text))
        for i, block in enumerate(language.layers[:self.layer + 1]):
            def decoder(h, b=block):
                rotary_module = language.layers[0].self_attn.rotary_emb
                pos_emb = None if rotary_module.fused_apply else rotary_module(h, positions)
                return b(h, mask, None, positions, pos_emb)
            stages.append((f'decoder_{i}', decoder))
        return stages

    def _snapshot(self, value):
        self.mx.eval(value)
        data = np.array(value.astype(self.mx.float32), copy=True)
        if not np.isfinite(data).all():
            raise FloatingPointError('Non-finite stage primal/adjoint')
        return data, value.dtype

    def _finish(self, phase, name, data, started):
        mx = self.mx
        peak, active = int(mx.get_peak_memory()), int(mx.get_active_memory())
        record = {'phase': phase, 'stage': name, 'peak_bytes': peak, 'active_bytes': active,
                  'seconds': time.monotonic() - started, 'shape': list(data.shape),
                  'norm': float(np.linalg.norm(data)), 'finite': bool(np.isfinite(data).all())}
        with self.log_path.open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        if peak > ACTIVE_MEMORY_LIMIT:
            raise MemoryError(f'{phase}/{name} exceeded9GiB: {peak}; no next stage allowed')
        self.probe._capture.clear()
        self.probe._intervention_records.clear()
        mx.clear_cache()

    def forward(self, rgb):
        mx = self.mx
        self.inputs = []
        current = (np.asarray(rgb, dtype=np.float32), mx.float32)
        self.probe._active_interventions = ()
        self.probe._checkpoint_blocks = False
        for name, function in self.stages:
            self.inputs.append(current)
            started = time.monotonic(); mx.reset_peak_memory()
            x = mx.array(current[0], dtype=current[1])
            y = function(x)
            current = self._snapshot(y)
            del x, y
            self._finish('forward', name, current[0], started)
        return current

    def backward(self, final, target=None, direction=None, sign=1.0):
        mx = self.mx
        hidden, dtype = final
        if (target is None) == (direction is None):
            raise ValueError('Supply exactly one of target or scalar direction')
        cotangent = np.zeros_like(hidden)
        if target is not None:
            residual = hidden[0, -8:] - target
            loss = float(np.mean(residual.astype(np.float64) ** 2))
            cotangent[0, -8:] = 2 * residual / residual.size
        else:
            unit = np.asarray(direction, dtype=np.float64)
            unit /= np.linalg.norm(unit)
            loss = float(-sign * np.mean(hidden[0, -8:].astype(np.float64) @ unit))
            cotangent[0, -8:] = -sign * unit[None, :] / 8
        current = (cotangent, dtype)
        for (name, function), (data, primal_dtype) in zip(reversed(self.stages), reversed(self.inputs)):
            started = time.monotonic(); mx.reset_peak_memory()
            if name == 'pixels_patch_embed_window':
                gradient = manual_patch_adjoint(current[0], self.patch_weight, self.window_index)
                if not np.isfinite(gradient).all():
                    raise FloatingPointError('Non-finite analytical patch gradient')
                current = (gradient, mx.float32)
                self._finish('backward_analytic_numpy', name, current[0], started)
                continue
            x = mx.array(data, dtype=primal_dtype)
            cot = mx.array(current[0], dtype=current[1])
            outputs, gradients = mx.vjp(function, [x], [cot])
            current = self._snapshot(gradients[0])
            del x, cot, outputs, gradients
            self._finish('backward', name, current[0], started)
        return loss, current[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('model', 'calibration-dir', 'output-dir'):
        parser.add_argument('--' + field, required=True)
    args = parser.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Use an empty output directory')
    calibration_path = Path(args.calibration_dir) / 'calibration_summary.json'
    calibration = json.loads(calibration_path.read_text())
    layer = int(calibration['selected_layer'])
    direction_path = Path(args.calibration_dir) / 'directions.npz'
    with np.load(direction_path) as directions:
        direction = directions[f'pain_L{layer}'].astype(np.float32)
    config = {'model': args.model, 'calibration': calibration,
              'calibration_sha256': sha256_file(calibration_path), 'direction_sha256': sha256_file(direction_path),
              'script_sha256': sha256_file(__file__), 'readout_layer': layer, 'steer_amount': 1,
              'peak_limit_bytes': ACTIVE_MEMORY_LIMIT, 'forward_absolute_tolerance': 1e-4,
              'finite_difference_epsilons': [.002, .01], 'finite_difference_seed': 2026092603,
              'patch_backward': 'Analytical NumPy nonoverlapping convolution adjoint plus inverse patch serialization; never invokes Conv3D/scatter autodiff.',
              'scope': 'One full-pixel gradient smoke; no optimization, full-model tape, or retry after guard failure.'}
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    probe = VLMProbe(args.model)
    if probe.model_revision != calibration['model_revision']:
        raise ValueError('Calibration revision mismatch')
    probe.mx.set_memory_limit(ACTIVE_MEMORY_LIMIT)
    gray = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 128 / 255, dtype=np.float32)
    prepared = probe.prepare_image(gray)
    (out / 'prompt.json').write_text(json.dumps({'input_ids': prepared['input_ids'].tolist(),
        'grid': prepared['image_grid_thw'].tolist(), 'user_text': prepared['user_text'],
        'continuation': prepared['assistant_continuation']}, indent=2) + '\n')
    reference, _ = probe.capture_image(gray, prepared=prepared, layers=(layer,), use_processor_pixels=False)
    target, _ = probe.capture_image(gray, prepared=prepared, layers=(layer,), use_processor_pixels=False,
        interventions=({'layer': max(0, layer - 8), 'vector': direction, 'amount': 1., 'positions': 'all'},))
    probe._capture.clear(); probe._intervention_records.clear(); probe.mx.clear_cache()
    chain = StagewiseAdjoint(probe, prepared, layer, out / 'stages.jsonl')
    final = chain.forward(gray)
    error = float(np.max(np.abs(final[0][0, -8:] - reference[layer])))
    (out / 'forward_check.json').write_text(json.dumps({'max_absolute_error': error, 'passed': error <= 1e-4}) + '\n')
    if error > 1e-4:
        raise ValueError(f'Stagewise forward differs from original by {error}; gradient not attempted')
    loss, gradient = chain.backward(final, target[layer])
    rng = np.random.default_rng(2026092603)
    coarse_direction = rng.normal(size=(8, 8, 3)).astype(np.float32)
    coarse_direction /= np.sqrt(np.mean(coarse_direction ** 2))
    pixel_direction = _coarse_to_rgb(coarse_direction)
    predicted = float(np.sum(gradient.astype(np.float64) * pixel_direction))
    np.savez_compressed(out / 'gradient.npz', gradient=gradient, target=target[layer],
                        reference=reference[layer], stagewise_final=final[0][0, -8:],
                        finite_difference_pixel_direction=pixel_direction)
    checks = []
    for epsilon in (.002, .01):
        values = []
        for sign in (1, -1):
            rgb = gray + sign * epsilon * pixel_direction
            if rgb.min() < 0 or rgb.max() > 1:
                raise ValueError('Finite-difference direction exceeds image bounds')
            capture, _ = probe.capture_image(rgb, prepared=prepared, layers=(layer,), use_processor_pixels=False)
            values.append(float(np.mean((capture[layer].astype(np.float64) - target[layer]) ** 2)))
        measured = (values[0] - values[1]) / (2 * epsilon)
        checks.append({'epsilon_coarse_pixel_rms': epsilon, 'plus_loss': values[0], 'minus_loss': values[1],
                       'finite_difference': measured, 'adjoint_dot_direction': predicted,
                       'relative_disagreement': abs(measured - predicted) / max(abs(measured), abs(predicted), 1e-12)})
    summary = {'loss': loss, 'gradient_norm': float(np.linalg.norm(gradient)),
               'forward_max_absolute_error': error, 'finite_difference_checks': checks,
               'saved_primal_bytes': sum(data.nbytes for data, _ in chain.inputs),
               'gradient_sha256': sha256_file(out / 'gradient.npz'), 'model': probe.run_metadata(),
               'interpretation': 'A gradient is not validated merely by being finite; inspect both directional checks and every stage peak. No optimization has run.'}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'summary': str(out / 'summary.json')}))


if __name__ == '__main__':
    main()
