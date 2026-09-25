"""Source-direction calibration and bounded RGB optimization for Qwen2.5-VL.

Calibration is a complete, deterministic pass over the frozen released corpus.
The optimizer writes PNGs and per-step JSONL before any downstream task study.
No generation or model-weight updates are used.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from experiments.vlm_probe import (
    ACTIVE_MEMORY_LIMIT,
    IMAGE_SIZE,
    MODEL_VARIANTS,
    NEUTRAL_ASSISTANT_CONTINUATION,
    VLMProbe,
    png_sha256,
)

PAIN_CATEGORIES = ('A1', 'A2', 'A3', 'A4', 'A5')
CONTROL_CATEGORIES = ('B', 'C1', 'C2', 'D', 'E')
LAYER_TRAIN = range(1, 11)
LAYER_SELECT = range(11, 16)
LAYER_TEST = range(16, 21)
DEFAULT_RANDOM_SEED = 2419


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_dataset(path):
    path = Path(path)
    data = json.loads(path.read_text())['datasets']
    s2 = data['S2_1P']['sentences']
    s1 = data['S1_1P']['sentences']
    if len(s2) != 200 or len(s1) != 200:
        raise ValueError(f'expected S2_1P=200 and S1_1P=200, got {len(s2)} and {len(s1)}')
    if {int(x['set']) for x in s2} != set(range(1, 21)):
        raise ValueError('S2 set IDs must be exactly 1..20')
    counts = defaultdict(int)
    for item in s2:
        counts[(item['category'], int(item['set']))] += 1
    if any(counts[(cat, sid)] != 1 for cat in (*PAIN_CATEGORIES, *CONTROL_CATEGORIES)
           for sid in range(1, 21)):
        raise ValueError('S2 source data is not one matched prompt/category/set')
    return data, path


def _remove_control_pcs(vector, controls, variance=0.50):
    controls = np.asarray(controls, dtype=np.float64)
    centered = controls - controls.mean(axis=0, keepdims=True)
    if len(centered) < 2 or not np.any(centered):
        return vector.copy(), 0
    _, singular, components = np.linalg.svd(centered, full_matrices=False)
    variance_mass = singular ** 2
    if not variance_mass.sum():
        return vector.copy(), 0
    n = min(int(np.searchsorted(np.cumsum(variance_mass) / variance_mass.sum(), variance) + 1),
            len(components))
    basis = components[:n]
    return vector - (vector @ basis.T) @ basis, n


def fit_direction(activations, categories, contrast_positive, contrast_negative,
                  denoise_categories, neutral_only=False):
    activations = np.asarray(activations, dtype=np.float64)
    categories = np.asarray(categories)
    pos = np.isin(categories, contrast_positive)
    neg = np.isin(categories, contrast_negative)
    if not pos.any() or not neg.any():
        raise ValueError('empty positive or negative source class')
    raw = activations[pos].mean(axis=0) - activations[neg].mean(axis=0)
    denoise = np.isin(categories, denoise_categories)
    if neutral_only:
        denoise = categories == 'D'
    vector, n_components = _remove_control_pcs(raw, activations[denoise])
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-10:
        raise ValueError('fitted residual direction has zero or invalid norm')
    return {'raw': raw.astype(np.float32), 'vector': vector.astype(np.float32),
            'norm': norm, 'denoised_components': n_components}


def auc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    p, n = scores[labels], scores[~labels]
    if not len(p) or not len(n):
        return None
    return float(((p[:, None] > n[None, :]).astype(float) +
                  0.5 * (p[:, None] == n[None, :])).mean())


def _slice_for_sets(records, set_ids):
    return np.asarray([int(r['set']) in set_ids for r in records], dtype=bool)


def _readout(acts, records, direction):
    scores = np.asarray(acts) @ direction
    labels = np.asarray([r['category'] in PAIN_CATEGORIES for r in records])
    return auc(scores, labels), scores


def _measure_readout(acts, records, vector):
    score = np.asarray(acts, dtype=np.float64) @ vector
    labels = np.asarray([r['category'] in PAIN_CATEGORIES for r in records])
    return {'auc_pain_vs_all_controls': auc(score, labels),
            'n_pain': int(labels.sum()), 'n_control': int((~labels).sum()),
            'score_mean_pain': float(score[labels].mean()),
            'score_mean_control': float(score[~labels].mean()),
            'score_std_pooled': float(score.std(ddof=1))}


def write_calibration_manifest(dataset_path, output_dir, profile=None):
    """Freeze the exact source prompts and splits before model weight loading."""
    data, dataset_path = load_dataset(dataset_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = ([{'group': 'S2_1P', **item} for item in data['S2_1P']['sentences']] +
             [{'group': 'S1_1P', 'set': None, **item} for item in data['S1_1P']['sentences']])
    manifest_path = out / 'source_manifest.jsonl'
    with manifest_path.open('w', encoding='utf-8') as stream:
        for i, item in enumerate(cases, 1):
            stream.write(json.dumps({'index': i, **item}, ensure_ascii=False) + '\n')
    profile = profile or MODEL_VARIANTS['qwen2.5-vl-3b-4bit']
    config = {
        'model_variant': profile.get('name', 'qwen2.5-vl-3b-4bit'),
        'model_repository': profile['repository'], 'model_revision': profile['revision'],
        'dataset_path': str(dataset_path), 'dataset_sha256': sha256_file(dataset_path),
        's2_train_sets': list(LAYER_TRAIN), 's2_layer_selection_sets': list(LAYER_SELECT),
        's2_heldout_sets': list(LAYER_TEST), 'candidate_postblock_layers': list(profile['sites']),
        'activation': 'post-block residual at final token of exact raw released prompt',
        'pain_contrast': 'mean(A1..A5) minus mean(B,C1,C2,D,E), control-PC-denoised at 50% variance',
        'comparator_contrasts': {'fear': 'mean(B) minus mean(D), denoised on D controls',
                                 'negative_emotion': 'mean(C1) minus mean(D), denoised on D controls'},
        'case_count': len(cases), 'manifest': str(manifest_path),
    }
    (out / 'calibration_config.json').write_text(json.dumps(config, indent=2) + '\n')
    return config, cases


def run_calibration(probe, dataset_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset, dataset_path = load_dataset(dataset_path)
    s2 = dataset['S2_1P']['sentences']
    s1 = dataset['S1_1P']['sentences']
    # ``main`` writes this same frozen manifest before constructing VLMProbe.
    cases = ([{'group': 'S2_1P', **item} for item in s2] +
             [{'group': 'S1_1P', 'set': None, **item} for item in s1])
    config_path = output_dir / 'calibration_config.json'
    if not config_path.exists():
        write_calibration_manifest(dataset_path, output_dir, probe.profile)
    config = json.loads(config_path.read_text())
    if config.get('model_revision') != probe.model_revision:
        raise ValueError('source manifest revision does not match loaded local VLM checkpoint')
    sites = tuple(probe.sites)
    records_path = output_dir / 'source_activations.jsonl'
    activations = {group: {layer: [] for layer in sites} for group in ('S2_1P', 'S1_1P')}
    serial_records = []
    started = time.monotonic()
    with records_path.open('w', encoding='utf-8') as stream:
        for i, case in enumerate(cases, 1):
            observed = probe.capture_text(case['prompt'], layers=sites)
            vectors = {str(layer): observed[layer].astype(np.float32).tolist() for layer in sites}
            record = {'index': i, 'group': case['group'], 'category': case['category'],
                      'set': case.get('set'), 'prompt': case['prompt'], 'activations': vectors}
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            serial_records.append(record)
            for layer in sites:
                activations[case['group']][layer].append(observed[layer].astype(np.float32))
            if i % 25 == 0:
                print(f'calibration forward {i}/{len(cases)}; active={probe.mx.get_active_memory()/1024**3:.2f} GiB', flush=True)

    fitted = {}
    layer_results = {}
    s2_categories = [x['category'] for x in s2]
    for layer in sites:
        s2_acts = np.asarray(activations['S2_1P'][layer], dtype=np.float32)
        s1_acts = np.asarray(activations['S1_1P'][layer], dtype=np.float32)
        train = _slice_for_sets(s2, set(LAYER_TRAIN))
        select = _slice_for_sets(s2, set(LAYER_SELECT))
        test = _slice_for_sets(s2, set(LAYER_TEST))
        directions = {
            'pain': fit_direction(s2_acts[train], np.asarray(s2_categories)[train],
                                  PAIN_CATEGORIES, CONTROL_CATEGORIES, CONTROL_CATEGORIES),
            'fear': fit_direction(s2_acts[train], np.asarray(s2_categories)[train],
                                  ('B',), ('D',), ('D',), neutral_only=True),
            'negative_emotion': fit_direction(s2_acts[train], np.asarray(s2_categories)[train],
                                               ('C1',), ('D',), ('D',), neutral_only=True),
        }
        pain_direction = directions['pain']['vector']
        s2_val = _measure_readout(s2_acts[select], [s2[i] for i in np.flatnonzero(select)], pain_direction)
        s2_test = _measure_readout(s2_acts[test], [s2[i] for i in np.flatnonzero(test)], pain_direction)
        s1_score = np.asarray(s1_acts) @ pain_direction
        s1_labels = np.asarray([r['category'] in PAIN_CATEGORIES for r in s1])
        comparator_scores = {}
        for name in ('fear', 'negative_emotion'):
            vector = directions[name]['vector']
            val = _measure_readout(s2_acts[select], [s2[i] for i in np.flatnonzero(select)], vector)
            testrow = _measure_readout(s2_acts[test], [s2[i] for i in np.flatnonzero(test)], vector)
            comparator_scores[name] = {'selection': val, 'heldout': testrow}
        fitted[str(layer)] = directions
        layer_results[str(layer)] = {
            'train_pain_vector_norm': directions['pain']['norm'],
            'validation_sets_11_15': s2_val,
            'heldout_sets_16_20': s2_test,
            'external_S1': {'auc_pain_vs_all_controls': auc(s1_score, s1_labels),
                            'score_mean_pain': float(s1_score[s1_labels].mean()),
                            'score_mean_control': float(s1_score[~s1_labels].mean())},
            'comparators': comparator_scores,
        }

    selected_layer = max(sites, key=lambda layer: (
        layer_results[str(layer)]['validation_sets_11_15']['auc_pain_vs_all_controls'], -layer
    ))
    selected = fitted[str(selected_layer)]
    source_std = layer_results[str(selected_layer)]['heldout_sets_16_20']['score_std_pooled']
    np.savez_compressed(
        output_dir / 'directions.npz',
        **{f'{name}_L{layer}_raw': directions[name]['raw']
           for layer, directions in fitted.items() for name in directions},
        **{f'{name}_L{layer}': directions[name]['vector']
           for layer, directions in fitted.items() for name in directions},
    )
    # A binary array file preserves the exact source activations for independent re-analysis.
    np.savez_compressed(output_dir / 'source_activations.npz',
                        **{f'{group}_L{layer}': np.asarray(values, dtype=np.float32)
                           for group, by_layer in activations.items()
                           for layer, values in by_layer.items()})
    summary = {
        **config,
        'probe_metadata': probe.run_metadata(),
        'selected_layer': selected_layer,
        'layer_selection_rule': 'maximum S2 set 11-15 pain-vs-control AUC; tie goes to shallower layer',
        'heldout_gate': {'threshold': 0.80,
                         'heldout_auc': layer_results[str(selected_layer)]['heldout_sets_16_20']['auc_pain_vs_all_controls'],
                         'passed': layer_results[str(selected_layer)]['heldout_sets_16_20']['auc_pain_vs_all_controls'] >= 0.80},
        'source_S2_heldout_score_std': source_std,
        'layers': layer_results,
        'directions_file': str(output_dir / 'directions.npz'),
        'source_activations_file': str(output_dir / 'source_activations.npz'),
        'source_activations_jsonl': str(records_path),
        'elapsed_seconds': time.monotonic() - started,
    }
    (output_dir / 'calibration_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    (output_dir / 'calibration.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return summary


def _atomic_jsonl_append(path, record):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()


def _coarse_to_rgb(coarse):
    """Bilinearly expand an 8x8 RGB control grid into the model's 224px image."""
    coarse = np.asarray(coarse, dtype=np.float32)
    if coarse.shape != (8, 8, 3):
        raise ValueError(f'expected 8x8x3 coarse RGB parameters, got {coarse.shape}')
    coords = np.linspace(0, 7, IMAGE_SIZE, dtype=np.float32)
    lo = np.floor(coords).astype(np.int32)
    hi = np.minimum(lo + 1, 7)
    weight = (coords - lo).reshape(-1, 1, 1)
    rows = coarse[lo, :, :] * (1 - weight) + coarse[hi, :, :] * weight
    weight = (coords - lo).reshape(1, -1, 1)
    return (rows[:, lo, :] * (1 - weight) + rows[:, hi, :] * weight).astype(np.float32)


def _json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def run_optimization(probe, calibration_dir, output_dir, steps=32, seed=DEFAULT_RANDOM_SEED,
                     steer_amount=0.05, step_size=0.02, resume=False,
                     rademacher_directions=2, perturbation=0.05):
    calibration_dir, output_dir = Path(calibration_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = calibration_dir / 'calibration_summary.json'
    summary = json.loads(summary_path.read_text())
    selected_layer = int(summary['selected_layer'])
    heldout = summary['layers'][str(selected_layer)]['heldout_sets_16_20']['auc_pain_vs_all_controls']
    if summary.get('model_revision') != probe.model_revision:
        raise ValueError('calibration direction revision does not match loaded local VLM checkpoint')
    if heldout < 0.80:
        raise RuntimeError(f'VLM source direction heldout AUC {heldout:.3f} is below the frozen 0.80 gate')
    directions = np.load(calibration_dir / 'directions.npz', allow_pickle=False)
    pain = directions[f'pain_L{selected_layer}'].astype(np.float32)
    rng = np.random.default_rng(seed)
    random_direction = rng.normal(size=pain.shape).astype(np.float32)
    random_direction *= float(np.linalg.norm(pain)) / float(np.linalg.norm(random_direction))
    candidate_directions = {'pain': pain, 'random': random_direction}
    intervention_layer = max(0, selected_layer - 8)
    gray = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 128 / 255, dtype=np.float32)
    from PIL import Image
    gray_path = output_dir / 'gray.png'
    if not gray_path.exists():
        Image.fromarray(np.rint(gray * 255).astype(np.uint8), mode='RGB').save(gray_path)
    init_coarse = np.clip(128 / 255 + rng.normal(0, 0.03, (8, 8, 3)), 0, 1).astype(np.float32)
    initial = _coarse_to_rgb(init_coarse)
    initial_path = output_dir / 'initial_noise.png'
    if not initial_path.exists():
        Image.fromarray(np.rint(initial * 255).astype(np.uint8), mode='RGB').save(initial_path)
    # Processor IDs/grid stay fixed; each objective call forwards new patch pixels only.
    prepared = probe.prepare_image(gray)
    baseline_captures, _ = probe.capture_image(
        gray, prepared=prepared, layers=(selected_layer,), use_processor_pixels=True
    )
    baseline_hidden = baseline_captures[selected_layer].astype(np.float32)
    initial_prepared = probe.prepare_image(initial)
    initial_hidden, _ = probe.capture_image(
        initial, prepared=initial_prepared, layers=(selected_layer,), use_processor_pixels=True
    )
    initial_hidden = initial_hidden[selected_layer].astype(np.float32)

    output_config = {
        'model_revision': probe.model_revision,
        'selected_layer': selected_layer, 'intervention_layer': intervention_layer,
        'source_direction': str(calibration_dir / 'directions.npz'),
        'heldout_source_auc': heldout,
        'optimizer': 'forward-only SPSA on 8x8x3 RGB parameter grid, bilinear upsampled to 224px; no backpropagation',
        'loss_arms': ['full_hidden_state_match', 'scalar_projection'],
        'direction_arms': ['pain_positive', 'pain_negative', 'random_positive'],
        'full_state_target': f'clean input target is uniform gray PNG with +/−{steer_amount} raw source vector injected after block {intervention_layer} at all token positions; last-eight-token residual at layer {selected_layer}',
        'full_state_loss': 'mean squared error of all hidden coordinates and final 8 token positions',
        'scalar_loss': 'maximize/minimize mean last-eight-token projection at selected layer',
        'steps': int(steps), 'step_size': float(step_size), 'steer_amount': float(steer_amount),
        'parameterization': '64 RGB control points per channel, bilinear expansion to 224x224, clipped [0,1]',
        'spsa_rademacher_directions_per_iteration': int(rademacher_directions),
        'spsa_plus_minus_evaluations_per_iteration': 2 * int(rademacher_directions),
        'spsa_perturbation': float(perturbation),
        'learning_rate_schedule': 'sign-SPSA update; cosine decay from configured step size to half that size',
        'resume_enabled': bool(resume),
        'seed': int(seed), 'rgb_size': [IMAGE_SIZE, IMAGE_SIZE, 3],
        'target_png': str(output_dir / 'gray.png'),
        'initial_noise_png': str(output_dir / 'initial_noise.png'),
        'source_S2_heldout_raw_pain_projection_std': summary['source_S2_heldout_score_std'],
        'assistant_continuation': NEUTRAL_ASSISTANT_CONTINUATION,
    }
    (output_dir / 'optimization_config.json').write_text(json.dumps(output_config, indent=2) + '\n')
    records_path = output_dir / 'optimization_steps.jsonl'
    arms = []
    for direction_name, direction in candidate_directions.items():
        signs = (1, -1) if direction_name == 'pain' else (1,)
        for sign in signs:
            label = f'{direction_name}_positive' if sign > 0 else f'{direction_name}_negative'
            arms.append((label, direction, sign))

    hidden_path = output_dir / 'hidden_states.npz'
    if resume and hidden_path.exists():
        prior_hidden = np.load(hidden_path, allow_pickle=False)
        saved_hidden = {name: prior_hidden[name] for name in prior_hidden.files}
        saved_hidden.update({'gray_baseline': baseline_hidden, 'seeded_noise_start': initial_hidden})
    else:
        saved_hidden = {'gray_baseline': baseline_hidden, 'seeded_noise_start': initial_hidden}
    readout_vectors = {name: directions[f'{name}_L{selected_layer}'].astype(np.float32)
                       for name in ('pain', 'fear', 'negative_emotion')}

    def readout_row(hidden):
        final = np.asarray(hidden[-1], dtype=np.float64)
        row = {}
        for name, vector in readout_vectors.items():
            norm = float(np.linalg.norm(vector))
            row[f'{name}_raw_final_token_dot'] = float(final @ vector)
            row[f'{name}_normalized_final_token_projection'] = float(final @ vector / norm)
        return row

    total_evaluations = 2 * rademacher_directions * steps
    progress_path = output_dir / 'progress.json'
    run_signature = {
        'model_revision': probe.model_revision, 'direction_sha256': sha256_file(calibration_dir / 'directions.npz'),
        'selected_layer': selected_layer, 'steps': steps, 'seed': seed,
        'steer_amount': steer_amount, 'step_size': step_size, 'perturbation': perturbation,
        'rademacher_directions': rademacher_directions,
    }
    run_signature_hash = hashlib.sha256(json.dumps(run_signature, sort_keys=True).encode()).hexdigest()
    if resume and progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress.get('run_signature_sha256') != run_signature_hash:
            raise ValueError('resume parameters or direction file differ from the saved run')
    else:
        progress = {'completed_arms': [], 'run_signature_sha256': run_signature_hash}
    for arm_index, (label, direction, sign) in enumerate(arms):
        if label in progress['completed_arms']:
            continue
        target_captures, _ = probe.capture_image(
            gray, prepared=prepared, layers=(selected_layer,),
            interventions=({'layer': intervention_layer, 'vector': direction,
                            'amount': steer_amount * sign, 'positions': 'all'},),
        )
        target_hidden = target_captures[selected_layer].astype(np.float32)
        saved_hidden[f'{label}_target'] = target_hidden
        for loss_mode in ('full_hidden_state_match', 'scalar_projection'):
            if loss_mode == 'full_hidden_state_match':
                target = target_hidden
            else:
                target = None
            arm_dir = output_dir / loss_mode / label
            arm_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = arm_dir / 'checkpoint.npz'
            state_path = arm_dir / 'state.json'
            arm_config = {'arm': label, 'loss_mode': loss_mode, 'steps': steps,
                          'seed': seed, 'step_size': step_size, 'perturbation': perturbation,
                          'rademacher_directions': rademacher_directions,
                          'steer_amount': steer_amount, 'selected_layer': selected_layer,
                          'intervention_layer': intervention_layer}
            arm_config_hash = hashlib.sha256(json.dumps(arm_config, sort_keys=True).encode()).hexdigest()
            if (resume and checkpoint_path.exists() and state_path.exists() and
                    json.loads(state_path.read_text()).get('arm_config_sha256') == arm_config_hash):
                state = json.loads(state_path.read_text())
                checkpoint = np.load(checkpoint_path, allow_pickle=False)
                theta = checkpoint['theta'].astype(np.float32)
                best_theta = checkpoint['best_theta'].astype(np.float32)
                best_loss = float(state['best_loss'])
                initial_loss = state.get('initial_loss')
                start_step = int(state['next_step'])
                rng.bit_generator.state = state['rng_state']
            else:
                theta = init_coarse.copy()
                best_theta = theta.copy()
                best_loss = float('inf')
                initial_loss = None
                start_step = 0
                # Common perturbation stream across every objective and arm.
                rng_arm = np.random.default_rng(seed)
                rng.bit_generator.state = rng_arm.bit_generator.state
            initial_loss = (float(np.mean((initial_hidden - target) ** 2))
                            if target is not None else
                            float(-sign * np.mean(initial_hidden @ (direction / np.linalg.norm(direction)))))
            # The shared initial image is a real evaluated candidate. Keep it
            # eligible so SPSA cannot report a worse result as its best image.
            if start_step == 0:
                best_loss = initial_loss
                best_theta = theta.copy()
            step_records = []
            if resume and records_path.exists() and start_step:
                with records_path.open(encoding='utf-8') as stream:
                    step_records = [row for row in (json.loads(line) for line in stream)
                                    if row.get('arm') == label and row.get('loss_mode') == loss_mode
                                    and row.get('event') == 'spsa_step'
                                    and row.get('step', -1) < start_step]
            for step in range(start_step, steps):
                gradient_estimate = np.zeros_like(theta)
                paired_means = []
                resources = []
                objective_details = []
                for replica in range(rademacher_directions):
                    delta = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=theta.shape)
                    delta_hash = hashlib.sha256(delta.astype('i1').tobytes()).hexdigest()
                    pair_losses = []
                    for side in (1.0, -1.0):
                        candidate = np.clip(theta + side * perturbation * delta, 0, 1)
                        rgb = _coarse_to_rgb(candidate)
                        captured, _ = probe.capture_image(
                            rgb, prepared=prepared, layers=(selected_layer,), use_processor_pixels=False
                        )
                        hidden = captured[selected_layer].astype(np.float32)
                        loss = (float(np.mean((hidden - target) ** 2)) if target is not None
                                else float(-sign * np.mean(hidden @ (direction / np.linalg.norm(direction)))))
                        resource = {'mlx_active_bytes': int(probe.mx.get_active_memory()),
                                    'mlx_peak_bytes': int(probe.mx.get_peak_memory())}
                        pair_losses.append(loss)
                        resources.append(resource)
                        objective_details.append({'replica': replica, 'side': int(side),
                                                  'objective': loss, 'rgb_min': float(rgb.min()),
                                                  'parameter_sha256': hashlib.sha256(candidate.astype('<f4').tobytes()).hexdigest(),
                                                  'rademacher_sha256': delta_hash,
                                                  'rgb_max': float(rgb.max()), **resource})
                        _atomic_jsonl_append(records_path, {
                            'event': 'spsa_evaluation', 'arm': label, 'loss_mode': loss_mode,
                            'step': step, 'replica': replica, 'side': int(side),
                            'objective': loss,
                            'parameter_sha256': hashlib.sha256(candidate.astype('<f4').tobytes()).hexdigest(),
                            'rademacher_sha256': delta_hash, **resource,
                        })
                        if loss < best_loss:
                            best_loss = loss
                            best_theta = candidate.copy()
                    gradient_estimate += ((pair_losses[0] - pair_losses[1]) /
                                          (2 * perturbation)) * delta
                    paired_means.append((pair_losses[0] + pair_losses[1]) / 2)
                gradient_estimate /= rademacher_directions
                estimate_loss = float(np.mean(paired_means))
                lr = step_size * (0.5 + 0.5 * np.cos(np.pi * step / max(steps, 1)))
                theta = np.clip(theta - lr * np.sign(gradient_estimate), 0, 1).astype(np.float32)
                record = {'event': 'spsa_step', 'arm': label, 'loss_mode': loss_mode, 'step': step,
                          'directional_evaluations': 2 * rademacher_directions,
                          'total_forward_evaluations': (step + 1) * 2 * rademacher_directions,
                          'spsa_loss_estimate': estimate_loss,
                          'best_evaluated_candidate_loss': best_loss,
                          'gradient_estimate_l2': float(np.linalg.norm(gradient_estimate)),
                          'step_size': float(lr), 'theta_min': float(theta.min()),
                          'theta_max': float(theta.max()),
                          'theta_parameter_sha256': hashlib.sha256(theta.astype('<f4').tobytes()).hexdigest(),
                          'evaluations': objective_details}
                step_records.append(record)
                _atomic_jsonl_append(records_path, record)
                np.savez_compressed(checkpoint_path, theta=theta, best_theta=best_theta)
                _json_write(state_path, {'arm_config_sha256': arm_config_hash,
                                         'next_step': step + 1, 'best_loss': best_loss,
                                         'initial_loss': initial_loss,
                                         'rng_state': rng.bit_generator.state,
                                         'completed_evaluations': (step + 1) * 2 * rademacher_directions})
            # Save/reload the best objective iterate through the real processor.
            png_path = arm_dir / 'optimized.png'
            best_rgb = _coarse_to_rgb(best_theta)
            Image.fromarray(np.rint(best_rgb * 255).astype(np.uint8), mode='RGB').save(png_path)
            reloaded = np.asarray(Image.open(png_path).convert('RGB'), dtype=np.float32) / 255.0
            roundtrip_prepared = probe.prepare_image(reloaded)
            roundtrip_hidden, _ = probe.capture_image(
                reloaded, prepared=roundtrip_prepared, layers=(selected_layer,),
                use_processor_pixels=True,
            )
            actual = roundtrip_hidden[selected_layer]
            saved_hidden[f'{loss_mode}_{label}_final'] = actual.astype(np.float32)
            base_delta = target_hidden - baseline_hidden
            actual_delta = actual - baseline_hidden
            delta_norm = float(np.linalg.norm(base_delta))
            achieved_norm = float(np.linalg.norm(actual_delta))
            cosine = (float(np.vdot(base_delta.ravel(), actual_delta.ravel()) /
                            (delta_norm * achieved_norm)) if delta_norm > 0 and achieved_norm > 0 else None)
            initial_error = (float(np.mean((initial_hidden - target_hidden) ** 2))
                             if loss_mode == 'full_hidden_state_match' else None)
            gray_initial_error = (float(np.mean((baseline_hidden - target_hidden) ** 2))
                                  if loss_mode == 'full_hidden_state_match' else None)
            final_error = float(np.mean((actual - target_hidden) ** 2))
            if loss_mode == 'scalar_projection':
                final_objective = float(-sign * np.mean(actual @ (direction / np.linalg.norm(direction))))
            else:
                final_objective = final_error
            error_reduction_from_seeded_noise = (
                (initial_error - final_error) / initial_error
                if initial_error is not None and initial_error > 0 else None
            )
            error_reduction_from_gray = (
                (gray_initial_error - final_error) / gray_initial_error
                if gray_initial_error is not None and gray_initial_error > 0 else None
            )
            projection = float(np.mean(actual @ direction) / np.linalg.norm(direction))
            image_result = {
                'arm': label, 'loss_mode': loss_mode,
                'png': str(png_path), 'png_sha256': png_sha256(png_path),
                'roundtrip_selected_layer_projection': projection,
                'roundtrip_hidden_norm': float(np.linalg.norm(actual)),
                'target_delta_norm_from_gray': delta_norm,
                'achieved_delta_norm_from_gray': achieved_norm,
                'delta_cosine_to_causal_target': cosine,
                'initial_objective': initial_loss,
                'best_optimizer_objective': best_loss,
                'roundtrip_objective': final_objective,
                'best_parameter_sha256': hashlib.sha256(best_theta.astype('<f4').tobytes()).hexdigest(),
                'initial_objective_reference': 'seeded_noise_start',
                'gray_baseline_full_state_mse': gray_initial_error,
                'seeded_noise_full_state_mse': initial_error,
                'final_hidden_target_mse': final_error,
                'full_state_error_reduction': error_reduction_from_gray,
                'full_state_error_reduction_from_gray': error_reduction_from_gray,
                'full_state_error_reduction_from_seeded_noise': error_reduction_from_seeded_noise,
                'gray_baseline_readouts': readout_row(baseline_hidden),
                'seeded_noise_readouts': readout_row(initial_hidden),
                'target_readouts': readout_row(target_hidden),
                'roundtrip_readouts': readout_row(actual),
                'target_pain_raw_delta_in_source_std': (
                    (readout_row(target_hidden)['pain_raw_final_token_dot'] -
                     readout_row(baseline_hidden)['pain_raw_final_token_dot']) /
                    summary['source_S2_heldout_score_std']
                    if summary['source_S2_heldout_score_std'] else None),
                'roundtrip_pain_raw_delta_in_source_std': (
                    (readout_row(actual)['pain_raw_final_token_dot'] -
                     readout_row(baseline_hidden)['pain_raw_final_token_dot']) /
                    summary['source_S2_heldout_score_std']
                    if summary['source_S2_heldout_score_std'] else None),
                'steps': step_records,
            }
            np.savez_compressed(output_dir / 'hidden_states.npz', **saved_hidden)
            (arm_dir / 'result.json').write_text(json.dumps(image_result, indent=2) + '\n')
        progress['completed_arms'].append(label)
        progress['current_arm'] = label
        _json_write(progress_path, progress)
        print(f'completed SPSA arm {label}: {total_evaluations} evaluations/objective', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='local snapshot at the pinned 3B model revision')
    parser.add_argument('--dataset', required=True, help='released 3.1_pain_and_control_datasets.json')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--phase', choices=('calibrate', 'optimize'), required=True)
    parser.add_argument('--calibration-dir', help='calibration output directory for optimize phase')
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--seed', type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument('--steer-amount', type=float, help='preselected capability-safe direct steering dose')
    parser.add_argument('--optimizer', choices=('spsa',), default='spsa')
    parser.add_argument('--step-size', type=float, default=0.02)
    parser.add_argument('--perturbation', type=float, default=0.05)
    parser.add_argument('--rademacher-directions', type=int, default=2)
    parser.add_argument('--resume', action='store_true', help='resume from per-arm coarse-grid checkpoints')
    args = parser.parse_args(argv)
    if args.phase == 'optimize' and not args.calibration_dir:
        parser.error('--calibration-dir is required for optimize phase')
    if args.phase == 'optimize' and args.steer_amount is None:
        parser.error('--steer-amount is required for optimize phase; pass the separately selected dose')
    if args.steps < 1 or args.steps > 64:
        parser.error('--steps must be in 1..64')
    # Persist parameters before VLMProbe loads a model.
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'invocation.json').write_text(json.dumps({
        'phase': args.phase, 'model': str(Path(args.model)), 'dataset': args.dataset,
        'model_revision': VLMProbe.profile_for_path(args.model)['revision'],
        'max_active_memory_bytes': ACTIVE_MEMORY_LIMIT,
        'optimizer': args.optimizer, 'steps': args.steps, 'seed': args.seed,
        'step_size': args.step_size, 'perturbation': args.perturbation,
        'rademacher_directions': args.rademacher_directions, 'resume': args.resume,
    }, indent=2) + '\n')
    if args.phase == 'calibrate':
        write_calibration_manifest(args.dataset, out, VLMProbe.profile_for_path(args.model))
    elif args.phase == 'optimize':
        prior = json.loads((Path(args.calibration_dir) / 'calibration_summary.json').read_text())
        selected = int(prior['selected_layer'])
        auc_value = prior['layers'][str(selected)]['heldout_sets_16_20']['auc_pain_vs_all_controls']
        if auc_value < 0.80:
            parser.error(f'selected pain direction heldout AUC {auc_value:.3f} is below .80')
    probe = VLMProbe(args.model)
    if args.phase == 'calibrate':
        run_calibration(probe, args.dataset, out)
    else:
        run_optimization(probe, args.calibration_dir, out, args.steps,
                         args.seed, args.steer_amount, args.step_size, args.resume,
                         args.rademacher_directions, args.perturbation)


if __name__ == '__main__':
    main()
