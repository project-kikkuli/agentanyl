"""Measured-loss eight-bit coordinate search guided by a bounded pixel adjoint."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from experiments.vlm_adjoint import StagewiseAdjoint
from experiments.vlm_bridge import DEFAULT_RANDOM_SEED, sha256_file
from experiments.vlm_pixel_pgd import objective, array_hash, write_json
from experiments.vlm_probe import ACTIVE_MEMORY_LIMIT, IMAGE_SIZE, VLMProbe

STEPS = 32
CHANNEL_COUNTS = (1, 64, 1024, 16384)
NOISE_SEED = 2026092604
RELATIVE_DECREASE = 1e-6


def ranked_proposals(pixels, gradient):
    flat = pixels.reshape(-1)
    grad = gradient.reshape(-1)
    signs = -np.sign(grad).astype(np.int16)
    eligible = np.flatnonzero(((signs > 0) & (flat < 255)) | ((signs < 0) & (flat > 0)))
    ranking = eligible[np.argsort(-np.abs(grad[eligible]), kind='stable')]
    for requested in CHANNEL_COUNTS:
        selected = ranking[:requested]
        candidate = flat.copy()
        candidate[selected] = (flat[selected].astype(np.int16) + signs[selected]).astype(np.uint8)
        yield requested, selected.size, candidate.reshape(pixels.shape)


def diagnostic_gate(directory, revision):
    directory = Path(directory)
    summary = json.loads((directory / 'summary.json').read_text())
    config = json.loads((directory / 'config.json').read_text())
    primals = json.loads((directory / 'primal_checks.json').read_text())
    if config['smoke_summary']['model']['model_revision'] != revision:
        raise ValueError('Diagnostic model revision mismatch')
    if max(primals.values()) > 1e-4:
        raise ValueError('Diagnostic primal checks failed')
    match = [r for r in summary['checks'] if r['background'] == 'noise'
             and r['direction'] == 'negative_gradient' and r['epsilon_rgb_rms'] == 1e-6]
    if len(match) != 1 or match[0]['relative_disagreement'] > .10 or not match[0]['plus_descent']:
        raise ValueError('Required high-signal local derivative evidence missing')
    stages = [json.loads(line) for line in (directory / 'noise_stages.jsonl').read_text().splitlines()]
    if any(not r['finite'] or max(r['peak_bytes'], r['active_bytes']) > ACTIVE_MEMORY_LIMIT for r in stages):
        raise ValueError('Diagnostic memory/finite guards failed')
    if not any(r['phase'] == 'backward_analytic_numpy' for r in stages):
        raise ValueError('Analytical patch adjoint was not exercised')
    if sha256_file(directory / 'diagnostic_arrays.npz') != summary['artifact_sha256']:
        raise ValueError('Diagnostic evidence changed')
    return {'summary': summary, 'config': config, 'primals': primals, 'selected_check': match[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('model', 'calibration-dir', 'diagnostic-dir', 'output-dir'):
        parser.add_argument('--' + field, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()) and not args.resume:
        raise FileExistsError('Fresh output directory required; no automatic retry')
    calibration_dir = Path(args.calibration_dir)
    calibration_path = calibration_dir / 'calibration_summary.json'
    calibration = json.loads(calibration_path.read_text())
    profile = VLMProbe.profile_for_path(args.model)
    if profile['revision'] != calibration['model_revision']:
        raise ValueError('Calibration/model mismatch')
    evidence = diagnostic_gate(args.diagnostic_dir, profile['revision'])
    layer = int(calibration['selected_layer'])
    direction_path = calibration_dir / 'directions.npz'
    with np.load(direction_path) as directions:
        pain = directions[f'pain_L{layer}'].astype(np.float32)
    random_direction = np.random.default_rng(DEFAULT_RANDOM_SEED).normal(size=pain.shape).astype(np.float32)
    random_direction *= float(np.linalg.norm(pain)) / float(np.linalg.norm(random_direction))
    arms = [('pain_positive', pain, 1.), ('pain_negative', pain, -1.), ('random_positive', random_direction, 1.)]
    modes = ('full_hidden_state_match', 'scalar_projection')
    gray_pixels = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 128, dtype=np.uint8)
    noise_float = gray_pixels / 255 + np.random.default_rng(NOISE_SEED).normal(0, .03, gray_pixels.shape)
    initial_pixels = np.rint(np.clip(noise_float, 0, 1) * 255).astype(np.uint8)
    config = {'protocol': 'frozen gradient-ranked one-LSB coordinate descent with measured-loss acceptance',
              'model': args.model, 'calibration': calibration, 'validation_evidence': evidence,
              'calibration_sha256': sha256_file(calibration_path), 'directions_sha256': sha256_file(direction_path),
              'diagnostic_summary_sha256': sha256_file(Path(args.diagnostic_dir) / 'summary.json'),
              'script_sha256': sha256_file(__file__), 'adjoint_sha256': sha256_file(Path(__file__).with_name('vlm_adjoint.py')),
              'steps': STEPS, 'proposal_channel_counts': CHANNEL_COUNTS, 'pixel_delta': 'one8-bit level',
              'ranking': 'Absolute current gradient descending; stable flattened-coordinate tie order; exclude zero/outward-saturated gradients.',
              'acceptance': 'Evaluate all four actual PNG candidates; accept lowest loss only if improvement exceeds1e-6*max(1,abs(current_loss)); deterministic candidate-order ties.',
              'relative_decrease': RELATIVE_DECREASE, 'noise_seed': NOISE_SEED, 'noise_std_before_rounding': .03,
              'direction_seed': DEFAULT_RANDOM_SEED, 'initial_pixels_sha256': array_hash(initial_pixels),
              'initial_state': 'same quantized noise for all arms; gray and noise separately eligible best outputs',
              'target_dose': 1, 'readout_layer': layer, 'intervention_layer': max(0, layer - 8),
              'loss_modes': modes, 'direction_arms': [a[0] for a in arms],
              'native_primal_anchor': 'At each current image, constant patch offset aligns continuous patchifier to real processor pixels. Its derivative is unchanged; strict1e-4 hidden equivalence required before backward.',
              'memory_guard_bytes': ACTIVE_MEMORY_LIMIT,
              'max_budget': {'reference_forwards': 5, 'proposal_forwards': 768, 'final_png_forwards': 6,
                             'stagewise_forward_chains': 192, 'stagewise_backward_chains': 192},
              'protocol_correction': 'Original coarse-direction FD gate failed and is preserved. New high-SNR nonuniform diagnostic supports one local derivative. Gradients only rank discrete proposals; actual native-image objective validates every accepted move. No claim of globally accurate gradients.',
              'selection_scope': 'Frozen activation objective only; no behavior or closed-model response influences proposals or selection.'}
    if args.resume:
        original = json.loads((out / 'config.json').read_text())
        for key in ('directions_sha256', 'initial_pixels_sha256', 'noise_seed', 'steps', 'proposal_channel_counts'):
            if json.dumps(original[key]) != json.dumps(config[key]):
                raise ValueError(f'Resume changed frozen parameter:{key}')
        write_json(out / 'resume_protocol.json', {'original_config_sha256': sha256_file(out / 'config.json'),
            'resume_script_sha256': sha256_file(__file__),
            'change': 'Stop an arm after first no-acceptance step: deterministic identical proposals would repeat. Preserve already repeated attempts and resume unchanged remaining arms.',
            'interrupted_forward_note': 'SIGINT may have interrupted one additional unlogged native forward. Completed full forwards and stage chains are counted from append-only logs.',
            'intermediate_hidden_note': 'Interrupted arm proposal losses/images/hashes remain logged; in-memory intermediate hidden arrays before interruption were not recovered by extra inference.'})
    else:
        write_json(out / 'config.json', config)
        np.savez_compressed(out / 'frozen_inputs.npz', gray=gray_pixels, noise=initial_pixels, pain=pain, random=random_direction)
    probe = VLMProbe(args.model)
    write_json(out / ('resume_model_metadata.json' if args.resume else 'model_metadata.json'), probe.run_metadata())
    total_forwards, forward_chains, backward_chains = 0, 0, 0
    states = {}
    if args.resume:
        total_forwards = len((out / 'forwards.jsonl').read_text().splitlines())
        for stage_file in out.glob('*/*/stages.jsonl'):
            stage_rows = [json.loads(line) for line in stage_file.read_text().splitlines()]
            forward_chains += sum(r['phase'] == 'forward' and r['stage'] == f'decoder_{layer}' for r in stage_rows)
            backward_chains += sum(r['phase'] == 'backward_analytic_numpy' for r in stage_rows)
    started = time.monotonic()

    def capture_file(pixels, path, key, interventions=()):
        nonlocal total_forwards
        Image.fromarray(pixels).save(path)
        rgb = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255
        prepared = probe.prepare_image(rgb)
        captures, _ = probe.capture_image(rgb, prepared=prepared, layers=(layer,), interventions=interventions,
                                          use_processor_pixels=True)
        total_forwards += 1
        hidden = captures[layer].astype(np.float32)
        with (out / 'forwards.jsonl').open('a') as stream:
            stream.write(json.dumps({'call': total_forwards, 'key': key, 'png': str(path), 'png_sha256': sha256_file(path),
                                     'hidden_sha256': array_hash(hidden), 'elapsed_seconds': time.monotonic() - started}) + '\n')
        return hidden, prepared, rgb

    if args.resume:
        with np.load(out / 'hidden_states.npz') as saved:
            states = {name: saved[name] for name in saved.files}
        baseline, initial_hidden = states['gray_baseline'], states['initial_noise']
    else:
        baseline, prepared, _ = capture_file(gray_pixels, out / 'gray.png', 'gray')
        initial_hidden, _, _ = capture_file(initial_pixels, out / 'initial_noise.png', 'noise')
        write_json(out / 'prompt.json', {'ids': prepared['input_ids'].tolist(), 'grid': prepared['image_grid_thw'].tolist(),
                                       'user': prepared['user_text'], 'continuation': prepared['assistant_continuation']})
        states.update(gray_baseline=baseline, initial_noise=initial_hidden)
        for name, direction, sign in arms:
            target, _, _ = capture_file(gray_pixels, out / f'{name}_target_input.png', name + '_target',
                ({'layer': max(0, layer - 8), 'vector': direction, 'amount': sign, 'positions': 'all'},))
            states[name + '_target'] = target
        np.savez_compressed(out / 'hidden_states.npz', **states)
    summaries = {}
    for name, direction, sign in arms:
        target = states[name + '_target']
        for mode in modes:
            arm_dir = out / mode / name; arm_dir.mkdir(parents=True, exist_ok=True)
            if args.resume and (arm_dir / 'result.json').exists():
                summaries[f'{mode}/{name}'] = json.loads((arm_dir / 'result.json').read_text())
                continue
            current_pixels, current_hidden = initial_pixels.copy(), initial_hidden.copy()
            current_loss = objective(current_hidden, mode, target, direction, sign)
            gray_loss = objective(baseline, mode, target, direction, sign)
            initial_loss = current_loss
            best_loss, best_pixels, best_hidden, best_source = gray_loss, gray_pixels.copy(), baseline.copy(), 'gray'
            if current_loss < best_loss:
                best_loss, best_pixels, best_hidden, best_source = current_loss, current_pixels.copy(), current_hidden.copy(), 'initial_noise'
            accepted, arm_states, arm_calls = 0, {}, 0
            start_step, stopped = 0, False
            if args.resume and (arm_dir / 'steps.jsonl').exists():
                prior = [json.loads(line) for line in (arm_dir / 'steps.jsonl').read_text().splitlines()]
                if prior:
                    last = prior[-1]; start_step = last['iteration'] + 1
                    accepted = sum(r['accepted'] for r in prior)
                    with np.load(arm_dir / 'checkpoint.npz') as checkpoint:
                        current_pixels, best_pixels, best_hidden = checkpoint['current'], checkpoint['best'], checkpoint['best_hidden']
                    current_loss, best_loss, best_source = last['current_objective_after'], last['best_objective'], last['best_source']
                    if current_loss == best_loss:
                        current_hidden = best_hidden.copy()
                    else:
                        current_hidden, _, _ = capture_file(current_pixels, arm_dir / 'resume_current.png', f'{mode}/{name}/resume_current')
                    stopped = not last['accepted']
                    arm_calls = sum(1 for line in (out / 'forwards.jsonl').read_text().splitlines()
                                    if json.loads(line)['key'].startswith(f'{mode}/{name}/') and 'selected_recheck' not in json.loads(line)['key'])
            for step in range(start_step, start_step if stopped else STEPS):
                rgb = current_pixels.astype(np.float32) / 255
                native = probe.prepare_image(rgb)
                mx = probe.mx
                continuous = probe._rgb_patchify(mx.array(rgb))
                offset = native['processor_pixel_values'].astype(mx.float32) - continuous
                mx.eval(offset)
                patch_offset = np.array(offset, copy=True)
                del offset, continuous
                chain = StagewiseAdjoint(probe, native, layer, arm_dir / 'stages.jsonl', patch_offset=patch_offset)
                final = chain.forward(rgb); forward_chains += 1
                primal_error = float(np.max(np.abs(final[0][0, -8:] - current_hidden)))
                if primal_error > 1e-4:
                    raise ValueError(f'Native processor anchor failed: {primal_error}; no gradient/proposals')
                if mode == 'full_hidden_state_match':
                    _, gradient = chain.backward(final, target=target)
                else:
                    _, gradient = chain.backward(final, direction=direction, sign=sign)
                backward_chains += 1
                if not np.isfinite(gradient).all():
                    raise FloatingPointError('Nonfinite proposal gradient')
                del chain, final
                candidate_rows, winning = [], None
                for requested, changed, candidate in ranked_proposals(current_pixels, gradient):
                    row = {'requested_channels': requested, 'changed_channels': int(changed)}
                    if not changed:
                        row['skipped'] = 'no eligible nonzero gradient channel'
                        candidate_rows.append(row); continue
                    path = arm_dir / f'step_{step:02d}_k{requested:05d}.png'
                    hidden, _, _ = capture_file(candidate, path, f'{mode}/{name}/{step}/{requested}')
                    arm_calls += 1
                    loss = objective(hidden, mode, target, direction, sign)
                    arm_states[f'step_{step:02d}_k{requested:05d}'] = hidden
                    row.update(objective=loss, png=str(path), png_sha256=sha256_file(path),
                               pixel_rms_step=float(np.sqrt(np.mean(((candidate.astype(float) - current_pixels) / 255) ** 2))),
                               predicted_change=float(np.sum(gradient.astype(float) * ((candidate.astype(float) - current_pixels) / 255))))
                    candidate_rows.append(row)
                    if winning is None or loss < winning[0]:
                        winning = (loss, candidate, hidden, requested)
                threshold = RELATIVE_DECREASE * max(1., abs(current_loss))
                old_loss = current_loss
                accept = winning is not None and winning[0] < current_loss - threshold
                if accept:
                    current_loss, current_pixels, current_hidden, chosen_k = winning
                    accepted += 1
                    if current_loss < best_loss:
                        best_loss, best_pixels, best_hidden, best_source = current_loss, current_pixels.copy(), current_hidden.copy(), f'step_{step}_k{chosen_k}'
                record = {'iteration': step, 'primal_max_error': primal_error,
                          'patch_offset_max_abs': float(np.max(np.abs(patch_offset))),
                          'gradient_norm': float(np.linalg.norm(gradient)), 'gradient_sha256': array_hash(gradient),
                          'current_objective_before': old_loss, 'current_objective_after': current_loss,
                          'acceptance_threshold': threshold, 'accepted': bool(accept),
                          'selected_k': winning[3] if accept else None,
                          'best_objective': best_loss, 'best_source': best_source, 'proposals': candidate_rows,
                          'current_pixels_sha256': array_hash(current_pixels),
                          'current_pixel_std': float(np.std(current_pixels.astype(float) / 255))}
                with (arm_dir / 'steps.jsonl').open('a') as stream:
                    stream.write(json.dumps(record, allow_nan=False) + '\n')
                np.savez_compressed(arm_dir / 'checkpoint.npz', current=current_pixels, best=best_pixels,
                                    best_hidden=best_hidden, last_gradient=gradient)
                write_json(out / 'progress.json', {'arm': name, 'mode': mode, 'completed_iterations': step + 1,
                    'accepted_iterations': accepted, 'current_objective': current_loss, 'best_objective': best_loss,
                    'gray_objective': gray_loss, 'full_forwards': total_forwards,
                    'forward_chains': forward_chains, 'backward_chains': backward_chains,
                    'elapsed_seconds': time.monotonic() - started})
                if not accept:
                    stopped = True
                    break
            png = arm_dir / 'optimized.png'
            actual, _, _ = capture_file(best_pixels, png, f'{mode}/{name}/selected_recheck')
            final_loss = objective(actual, mode, target, direction, sign)
            target_delta, actual_delta = target - baseline, actual - baseline
            denom = float(np.linalg.norm(target_delta) * np.linalg.norm(actual_delta))
            base_mse = float(np.mean(target_delta.astype(float) ** 2))
            final_mse = float(np.mean((actual.astype(float) - target) ** 2))
            report = {'gray_objective': gray_loss, 'initial_noise_objective': initial_loss,
                      'best_measured_objective': best_loss, 'selected_recheck_objective': final_loss,
                      'recheck_difference': final_loss - best_loss, 'best_source': best_source,
                      'accepted_iterations': accepted, 'proposal_forwards': arm_calls,
                      'stopped_at_deterministic_fixed_point': stopped,
                      'unused_gradient_iteration_budget': STEPS - len((arm_dir / 'steps.jsonl').read_text().splitlines()),
                      'png': str(png), 'png_sha256': sha256_file(png),
                      'gray_target_mse': base_mse, 'selected_target_mse': final_mse,
                      'target_mse_reduction_from_gray': 1 - final_mse / base_mse if base_mse else None,
                      'delta_cosine_to_target': float(np.vdot(target_delta, actual_delta) / denom) if denom else None,
                      'target_delta_norm': float(np.linalg.norm(target_delta)), 'actual_delta_norm': float(np.linalg.norm(actual_delta))}
            states[f'{mode}_{name}_selected'] = actual
            arm_states['selected'] = actual
            np.savez_compressed(arm_dir / 'proposal_hidden_states.npz', **arm_states)
            write_json(arm_dir / 'result.json', report)
            summaries[f'{mode}/{name}'] = report
            np.savez_compressed(out / 'hidden_states.npz', **states)
    if forward_chains > 192 or backward_chains > 192 or total_forwards > 779:
        raise AssertionError('Frozen search budget mismatch')
    write_json(out / 'summary.json', {'arms': summaries, 'full_forwards': total_forwards,
               'forward_chains': forward_chains, 'backward_chains': backward_chains,
               'elapsed_seconds': time.monotonic() - started,
               'hidden_states_sha256': sha256_file(out / 'hidden_states.npz'),
               'interpretation': 'Native8-bit image activation optimization only. No behavioral or incentive-transfer result is implied.'})
    print(json.dumps({'summary': str(out / 'summary.json'), 'full_forwards': total_forwards}))


if __name__ == '__main__':
    main()
