"""Frozen large-step native-image escape audit from six already selected outputs."""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from experiments.vlm_adjoint import StagewiseAdjoint
from experiments.vlm_bridge import sha256_file
from experiments.vlm_pixel_pgd import objective, array_hash, write_json
from experiments.vlm_probe import ACTIVE_MEMORY_LIMIT, VLMProbe

STEPS = 8
LEVELS = (1, 4, 16, 64)


def proposals(pixels, gradient):
    signs = -np.sign(gradient).astype(np.int16)
    for levels in LEVELS:
        unbounded = pixels.astype(np.int16) + levels * signs
        candidate = np.clip(unbounded, 0, 255).astype(np.uint8)
        yield levels, candidate, float(np.mean(candidate.astype(np.int16) != unbounded))


def make_manifest(source, output, summaries):
    cases = [('gray', source / 'gray.png'), ('initial_noise', source / 'initial_noise.png')]
    original = json.loads((source / 'summary.json').read_text())
    for key, row in original['arms'].items():
        cases.append(('original/' + key, Path(row['png'])))
    for key, row in summaries.items():
        cases.append(('escape/' + key, Path(row['png'])))
    # IDs are assigned by sorted SHA, independent of condition names or effects.
    by_sha = {sha256_file(path): path for _, path in cases}
    entries, mapping = [], {}
    delivery = output / 'delivery'; delivery.mkdir()
    for i, sha in enumerate(sorted(by_sha)):
        image_id = f'image_{i:02d}'
        path = delivery / f'{image_id}.png'
        shutil.copyfile(by_sha[sha], path)
        entries.append({'id': image_id, 'path': str(path.relative_to(output)), 'sha256': sha})
        mapping[sha] = image_id
    write_json(output / 'images_manifest.json', {'path_base': '.', 'images': entries})
    write_json(output / 'image_conditions.json', {'private_analysis_only': True,
        'conditions': [{'condition': name, 'image_id': mapping[sha256_file(path)],
                        'sha256': sha256_file(path), 'source_path': str(path)} for name, path in cases]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('model', 'source-dir', 'output-dir'):
        parser.add_argument('--' + field, required=True)
    args = parser.parse_args()
    source, out = Path(args.source_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Use an empty escape directory')
    prior = json.loads((source / 'summary.json').read_text())
    prior_config = json.loads((source / 'config.json').read_text())
    if len(prior['arms']) != 6:
        raise ValueError('All six original conditions must finish before escape')
    profile = VLMProbe.profile_for_path(args.model)
    if profile['revision'] != prior_config['calibration']['model_revision']:
        raise ValueError('Model revision mismatch')
    layer = prior_config['readout_layer']
    with np.load(source / 'frozen_inputs.npz') as vectors:
        directions = {'pain': vectors['pain'], 'random': vectors['random']}
    with np.load(source / 'hidden_states.npz') as saved:
        original_states = {key: saved[key] for key in saved.files}
    baseline = original_states['gray_baseline']
    config = {'protocol': 'predeclared all-channel signed8-bit escape with measured native loss acceptance',
              'source_summary_sha256': sha256_file(source / 'summary.json'),
              'source_config_sha256': sha256_file(source / 'config.json'), 'source_config': prior_config,
              'source_hidden_states_sha256': sha256_file(source / 'hidden_states.npz'),
              'script_sha256': sha256_file(__file__), 'model': args.model,
              'max_iterations_per_arm': STEPS, 'all_channel_step_levels': LEVELS,
              'proposal_rule': 'Subtract sign(gradient)*{1,4,16,64} from EVERY8-bit RGB channel and clip to[0,255].',
              'acceptance': 'Evaluate all changed proposals; accept smallest measured loss if decrease>1e-6*max(1,abs(current_loss)).',
              'stopping': 'Stop arm immediately on no accepted proposal. Retain original output if none improves. No new seeds/restarts.',
              'max_budget': {'initial_forwards': 6, 'proposal_forwards': 192, 'final_forwards': 6,
                             'stagewise_forward_chains': 48, 'stagewise_backward_chains': 48},
              'selection': 'Frozen activation objective only; no behavior used. All six original outputs remain available.',
              'memory_guard_bytes': ACTIVE_MEMORY_LIMIT}
    write_json(out / 'config.json', config)
    probe = VLMProbe(args.model)
    write_json(out / 'model_metadata.json', probe.run_metadata())
    summaries, states, calls, chains = {}, {}, 0, 0
    started = time.monotonic()

    def capture(path, key):
        nonlocal calls
        rgb = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255
        prepared = probe.prepare_image(rgb)
        hidden, _ = probe.capture_image(rgb, prepared=prepared, layers=(layer,), use_processor_pixels=True)
        calls += 1
        with (out / 'forwards.jsonl').open('a') as f:
            f.write(json.dumps({'call': calls, 'key': key, 'png': str(path), 'sha256': sha256_file(path),
                                'elapsed_seconds': time.monotonic() - started}) + '\n')
        return hidden[layer].astype(np.float32), prepared, rgb

    for key, previous in prior['arms'].items():
        mode, arm = key.split('/')
        sign = -1. if arm.endswith('negative') else 1.
        direction = directions['random' if arm.startswith('random') else 'pain']
        target = original_states[arm + '_target']
        arm_dir = out / mode / arm; arm_dir.mkdir(parents=True)
        selected_path = Path(previous['png'])
        pixels = np.asarray(Image.open(selected_path).convert('RGB'), dtype=np.uint8)
        hidden, _, _ = capture(selected_path, key + '/original')
        error = float(np.max(np.abs(hidden - original_states[f'{mode}_{arm}_selected'])))
        if error > 1e-4:
            raise ValueError(f'Original selected native image did not repeat: {error}')
        current_loss = objective(hidden, mode, target, direction, sign)
        initial_loss = current_loss
        accepted, proposal_calls, arm_states, stopped = 0, 0, {}, False
        for step in range(STEPS):
            rgb = pixels.astype(np.float32) / 255
            native = probe.prepare_image(rgb)
            mx = probe.mx
            offset = native['processor_pixel_values'].astype(mx.float32) - probe._rgb_patchify(mx.array(rgb))
            mx.eval(offset); patch_offset = np.array(offset, copy=True); del offset
            chain = StagewiseAdjoint(probe, native, layer, arm_dir / 'stages.jsonl', patch_offset=patch_offset)
            final = chain.forward(rgb)
            primal_error = float(np.max(np.abs(final[0][0, -8:] - hidden)))
            if primal_error > 1e-4:
                raise ValueError('Escape native-primal equivalence failed')
            if mode == 'full_hidden_state_match':
                _, gradient = chain.backward(final, target=target)
            else:
                _, gradient = chain.backward(final, direction=direction, sign=sign)
            chains += 1; del chain, final
            winning, candidates = None, []
            for level, candidate, clipped in proposals(pixels, gradient):
                changed = int(np.count_nonzero(candidate != pixels))
                row = {'levels': level, 'changed_channels': changed, 'clipped_fraction': clipped}
                if not changed:
                    row['skipped'] = True; candidates.append(row); continue
                path = arm_dir / f'step_{step:02d}_levels{level:02d}.png'
                Image.fromarray(candidate).save(path)
                h, _, _ = capture(path, f'{key}/{step}/{level}'); proposal_calls += 1
                loss = objective(h, mode, target, direction, sign)
                arm_states[f'step_{step:02d}_levels{level:02d}'] = h
                row.update(objective=loss, png=str(path), sha256=sha256_file(path),
                           actual_pixel_rms=float(np.sqrt(np.mean(((candidate.astype(float) - pixels) / 255) ** 2))))
                candidates.append(row)
                if winning is None or loss < winning[0]:
                    winning = (loss, candidate, h, path, level)
            before = current_loss
            accept = winning is not None and winning[0] < current_loss - 1e-6 * max(1., abs(current_loss))
            if accept:
                current_loss, pixels, hidden, selected_path, selected_level = winning
                accepted += 1
            record = {'iteration': step, 'primal_max_error': primal_error,
                      'gradient_norm': float(np.linalg.norm(gradient)), 'gradient_sha256': array_hash(gradient),
                      'objective_before': before, 'objective_after': current_loss, 'accepted': bool(accept),
                      'selected_level': winning[4] if accept else None, 'proposals': candidates}
            with (arm_dir / 'steps.jsonl').open('a') as f:
                f.write(json.dumps(record, allow_nan=False) + '\n')
            np.savez_compressed(arm_dir / 'checkpoint.npz', pixels=pixels, hidden=hidden, last_gradient=gradient)
            write_json(out / 'progress.json', {'arm': key, 'completed_iterations': step + 1,
                       'accepted': accepted, 'initial_objective': initial_loss, 'current_objective': current_loss,
                       'full_forwards': calls, 'gradient_chains': chains, 'elapsed_seconds': time.monotonic() - started})
            if not accept:
                stopped = True; break
        png = arm_dir / 'optimized.png'; shutil.copyfile(selected_path, png)
        actual, _, _ = capture(png, key + '/selected_recheck')
        final_loss = objective(actual, mode, target, direction, sign)
        delta, target_delta = actual - baseline, target - baseline
        denom = float(np.linalg.norm(delta) * np.linalg.norm(target_delta))
        gray_mse = float(np.mean(target_delta.astype(float) ** 2))
        mse = float(np.mean((actual.astype(float) - target) ** 2))
        report = {'original_objective': initial_loss, 'selected_objective': current_loss,
                  'selected_recheck_objective': final_loss, 'recheck_difference': final_loss - current_loss,
                  'accepted_iterations': accepted, 'executed_iterations': step + 1,
                  'unused_iteration_budget': STEPS - (step + 1), 'stopped_at_fixed_point': stopped,
                  'proposal_forwards': proposal_calls, 'png': str(png), 'png_sha256': sha256_file(png),
                  'unchanged_from_original': sha256_file(png) == previous['png_sha256'],
                  'target_mse_reduction_from_gray': 1 - mse / gray_mse if gray_mse else None,
                  'gray_target_mse': gray_mse, 'selected_target_mse': mse,
                  'delta_cosine_to_target': float(np.vdot(delta, target_delta) / denom) if denom else None}
        arm_states['selected'] = actual
        np.savez_compressed(arm_dir / 'proposal_hidden_states.npz', **arm_states)
        states[f'{mode}_{arm}_selected'] = actual
        states[arm + '_target'] = target
        write_json(arm_dir / 'result.json', report); summaries[key] = report
        np.savez_compressed(out / 'hidden_states.npz', gray_baseline=baseline, **states)
    if calls > 204 or chains > 48:
        raise AssertionError('Escape budget exceeded')
    write_json(out / 'summary.json', {'arms': summaries, 'full_forwards': calls, 'gradient_chains': chains,
               'elapsed_seconds': time.monotonic() - started, 'search_stage': 'frozen_large_step_escape',
               'hidden_states_sha256': sha256_file(out / 'hidden_states.npz')})
    make_manifest(source, out, summaries)
    print(json.dumps({'summary': str(out / 'summary.json'), 'manifest': str(out / 'images_manifest.json')}))


if __name__ == '__main__':
    main()
