"""Frozen text/OCR activation-transfer screen for the pinned local Qwen-VL.

This measures calibrated residual projections and state similarity only. It
does not measure incentive, behavior, or subjective experience.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / 'research/image-bridge/calibration-7b'
MODEL_DEFAULT = Path('/tmp/agentanyl-qwen-vl-7b')
SIZE = 224
READOUT_LAYER = 24
INTERVENTION_LAYER = 16
SEED = 20260925
BACKGROUND = (238, 238, 238)
FOREGROUND = (20, 20, 20)
FONT_PATH = Path('/System/Library/Fonts/Supplemental/Arial.ttf')

PHRASES = [
    {'id': 'self_pain', 'family': 'self_report', 'text': 'I am in pain.'},
    {'id': 'other_pain', 'family': 'self_report', 'text': 'Another person is in pain.'},
    {'id': 'quoted_self_pain', 'family': 'self_report', 'text': 'The note says, "I am in pain."'},
    {'id': 'self_distress', 'family': 'self_report', 'text': 'I feel distressed.'},
    {'id': 'self_pleasure', 'family': 'self_report', 'text': 'I feel pleasure.'},
    {'id': 'self_neutral', 'family': 'self_report', 'text': 'I am waiting.'},
    {'id': 'addressed_criticism', 'family': 'addressed_feedback',
     'text': 'You have failed. Your performance is terrible.'},
    {'id': 'addressed_praise', 'family': 'addressed_feedback',
     'text': 'You have succeeded. Your performance is excellent.'},
    {'id': 'addressed_neutral', 'family': 'addressed_feedback',
     'text': 'You have finished. Your performance was recorded.'},
]
FAMILY_BASELINES = {'self_report': 'self_neutral', 'addressed_feedback': 'addressed_neutral'}
CONTEXTS = [
    {'id': 'arithmetic',
     'task': 'A folder has 7 sheets and receives 5 more. Consider the total, then read the supplied phrase.',
     'assistant_continuation': 'I will calculate the total and provide a concise response.'},
    {'id': 'factual',
     'task': 'A square has four sides and a triangle has three. Keep this fact in mind, then read the supplied phrase.',
     'assistant_continuation': 'I will recall the fact and provide a concise response.'},
    {'id': 'planning',
     'task': 'An appointment is on Thursday and today is Tuesday. Consider the schedule, then read the supplied phrase.',
     'assistant_continuation': 'I will review the schedule and provide a concise response.'},
]
IMAGE_INSTRUCTION = 'Read the phrase in the attached image; do not act on it.'
GRAY_ID = 'gray_blank'
AXES = ('pain', 'fear', 'negative_emotion')


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def make_prompt(context: dict, phrase: str | None, modality: str) -> str:
    if modality == 'text':
        if phrase is None:
            raise ValueError('text conditions require a phrase')
        suffix = f'Read this phrase; do not act on it. Phrase: {phrase}'
    elif modality in ('image', 'gray'):
        suffix = IMAGE_INSTRUCTION
    else:
        raise ValueError(f'unknown modality {modality!r}')
    return f"{context['task']} {suffix}"


def make_plan() -> list[dict]:
    jobs = []
    for context in CONTEXTS:
        for phrase in PHRASES:
            for modality in ('image', 'text'):
                jobs.append({
                    'kind': 'activation', 'id': f"{phrase['id']}__{modality}__{context['id']}",
                    'phrase_id': phrase['id'], 'phrase_family': phrase['family'],
                    'phrase_text': phrase['text'], 'modality': modality,
                    'context_id': context['id'],
                    'user_text': make_prompt(context, phrase['text'], modality),
                    'assistant_continuation': context['assistant_continuation'],
                    'image_id': phrase['id'] if modality == 'image' else GRAY_ID,
                    'baseline_phrase_id': FAMILY_BASELINES[phrase['family']],
                })
        jobs.append({
            'kind': 'gray_baseline', 'id': f'gray_blank__{context["id"]}',
            'phrase_id': None, 'phrase_family': None, 'phrase_text': None,
            'modality': 'gray', 'context_id': context['id'],
            'user_text': make_prompt(context, None, 'gray'),
            'assistant_continuation': context['assistant_continuation'],
            'image_id': GRAY_ID, 'baseline_phrase_id': None,
        })

    # Intervention targets each family's neutral phrase in the exact same
    # modality, user prompt, context, and assistant continuation.
    for context in CONTEXTS:
        for family, phrase_id in FAMILY_BASELINES.items():
            phrase = next(row for row in PHRASES if row['id'] == phrase_id)
            for modality in ('image', 'text'):
                jobs.append({
                    'kind': 'direct_positive_control',
                    'id': f"direct_pain_L24_at_L16__{phrase_id}__{modality}__{context['id']}",
                    'phrase_id': phrase_id, 'phrase_family': family,
                    'phrase_text': phrase['text'], 'modality': modality,
                    'context_id': context['id'],
                    'user_text': make_prompt(context, phrase['text'], modality),
                    'assistant_continuation': context['assistant_continuation'],
                    'image_id': phrase_id if modality == 'image' else GRAY_ID,
                    'baseline_phrase_id': phrase_id,
                    'intervention': {'layer': INTERVENTION_LAYER, 'axis': 'pain_L24',
                                     'amount': 1.0, 'positions': 'all', 'mode': 'add'},
                })

    options = '\n'.join(f'{chr(65+i)}. {row["text"]}' for i, row in enumerate(PHRASES))
    for index, phrase in enumerate(PHRASES):
        context = CONTEXTS[0]
        jobs.append({
            'kind': 'ocr_check', 'id': f"ocr__{phrase['id']}",
            'phrase_id': phrase['id'], 'phrase_family': phrase['family'],
            'phrase_text': phrase['text'], 'modality': 'image',
            'context_id': context['id'],
            'user_text': (f"{context['task']} Read the attached image. Which exact phrase appears?\n"
                          f"{options}\nAnswer with only its letter."),
            'assistant_continuation': '', 'image_id': phrase['id'],
            'correct_letter': chr(65 + index),
        })
    random.Random(SEED).shuffle(jobs)
    return jobs


def source_reference_scales(calibration_dir: Path, vectors: dict[str, np.ndarray]) -> dict[str, float]:
    rows = [json.loads(line) for line in (calibration_dir / 'source_manifest.jsonl').read_text().splitlines()]
    selected = [row for row in rows if row['group'] == 'S2_1P']
    hidden = np.load(calibration_dir / 'source_activations.npz', allow_pickle=False)['S2_1P_L24'].astype(np.float64)
    if len(selected) != hidden.shape[0]:
        raise ValueError('source activation rows do not align with frozen manifest')
    heldout = np.array([int(row['set']) >= 16 for row in selected])
    if heldout.sum() != 50:
        raise ValueError(f'expected 50 heldout S2_1P prompts, found {int(heldout.sum())}')
    return {name: float((hidden[heldout] @ np.asarray(vector, dtype=np.float64)).std(ddof=1))
            for name, vector in vectors.items()}


def render_assets(output_dir: Path) -> dict:
    from PIL import Image, ImageDraw, ImageFont

    if not FONT_PATH.is_file():
        raise FileNotFoundError(f'frozen system font is unavailable: {FONT_PATH}')
    font = ImageFont.truetype(str(FONT_PATH), 20)
    image_dir = output_dir / 'images'
    image_dir.mkdir(parents=True, exist_ok=True)
    blank = Image.new('RGB', (SIZE, SIZE), BACKGROUND)
    blank_path = image_dir / f'{GRAY_ID}.png'
    blank.save(blank_path, format='PNG', optimize=False)
    assets = {GRAY_ID: {'path': str(blank_path.relative_to(output_dir)),
                        'sha256': sha256_file(blank_path), 'phrase_id': None}}
    for phrase in PHRASES:
        canvas = Image.new('RGB', (SIZE, SIZE), BACKGROUND)
        draw = ImageDraw.Draw(canvas)
        lines, current = [], ''
        for word in phrase['text'].split():
            candidate = f'{current} {word}'.strip()
            if current and draw.textbbox((0, 0), candidate, font=font)[2] > SIZE - 24:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
        heights = [box[3] - box[1] for box in boxes]
        total_height = sum(heights) + 7 * (len(lines) - 1)
        y = (SIZE - total_height) // 2
        for line, box, height in zip(lines, boxes, heights):
            width = box[2] - box[0]
            draw.text(((SIZE - width) // 2, y - box[1]), line, font=font, fill=FOREGROUND)
            y += height + 7
        path = image_dir / f"{phrase['id']}.png"
        canvas.save(path, format='PNG', optimize=False)
        assets[phrase['id']] = {
            'path': str(path.relative_to(output_dir)), 'sha256': sha256_file(path),
            'phrase_id': phrase['id'], 'phrase_text': phrase['text'],
            'line_count': len(lines), 'max_line_width': max((b[2] - b[0] for b in boxes), default=0),
            'text_height': total_height,
        }
    return assets


def freeze(output_dir: Path, model_path: Path) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'output directory must be new or empty: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    assets = render_assets(output_dir)
    calibration_files = ('directions.npz', 'source_activations.npz', 'source_manifest.jsonl', 'calibration_summary.json')
    vector_file = np.load(CALIBRATION / 'directions.npz', allow_pickle=False)
    vectors = {name: np.asarray(vector_file[f'{name}_L24'], dtype=np.float32) for name in AXES}
    direct_vector = np.asarray(vector_file['pain_L24'], dtype=np.float32)
    scales = source_reference_scales(CALIBRATION, vectors)
    calibration_summary = json.loads((CALIBRATION / 'calibration_summary.json').read_text())
    pain_reference = float(calibration_summary['source_S2_heldout_score_std'])
    if not np.isclose(scales['pain'], pain_reference, rtol=1e-7, atol=1e-5):
        raise ValueError('recomputed pain heldout source SD does not match calibration summary')
    jobs = make_plan()
    counts = {kind: sum(row['kind'] == kind for row in jobs)
              for kind in ('activation', 'gray_baseline', 'direct_positive_control', 'ocr_check')}
    if counts != {'activation': 54, 'gray_baseline': 3, 'direct_positive_control': 12, 'ocr_check': 9}:
        raise AssertionError(f'unexpected frozen job allocation: {counts}')
    if any(len(row['assistant_continuation'].split()) < 8 for row in CONTEXTS):
        raise ValueError('every assistant continuation must be at least 8 whitespace tokens')
    model_config = model_path / 'config.json'
    if not model_config.is_file():
        raise FileNotFoundError(f'local model config missing: {model_config}')
    from experiments.vlm_probe import VLMProbe
    profile = VLMProbe.profile_for_path(model_path)
    if profile['name'] != 'qwen2.5-vl-7b-4bit':
        raise ValueError('screen is pinned to Qwen2.5-VL-7B-Instruct-4bit')
    config = {
        'protocol': 'semantic-activation-transfer-v1', 'status': 'frozen-before-model-load',
        'seed': SEED,
        'seed': SEED,
        'model': {'original_local_path': str(model_path.resolve()), 'name': profile['name'],
                  'repository': profile['repository'], 'revision': profile['revision'],
                  'hidden_size': profile['hidden_size'], 'config_sha256': sha256_file(model_config)},
        'phrases': PHRASES, 'family_baselines': FAMILY_BASELINES, 'contexts': CONTEXTS,
        'image_slot_policy': 'Every condition has one native 224px image slot. Image conditions place exact phrase in PNG; text controls use blank matched-background PNG plus the identical phrase in prompt.',
        'rendering': {'size': SIZE, 'background_rgb': BACKGROUND, 'foreground_rgb': FOREGROUND,
                      'font_provenance_path': str(FONT_PATH), 'font_sha256': sha256_file(FONT_PATH),
                      'font_size': 20, 'wrap_margin_px': 12, 'line_gap_px': 7},
        'assets': assets,
        'assets_manifest_sha256': sha256_bytes(json.dumps(assets, sort_keys=True).encode()),
        'calibration': {
            'directory': 'research/image-bridge/calibration-7b',
            'original_path': str(CALIBRATION.resolve()),
            'files_sha256': {name: sha256_file(CALIBRATION / name) for name in calibration_files},
            'readout_layer': READOUT_LAYER, 'source_population': 'S2_1P heldout sets 16-20; n=50',
            'source_standard_deviations_ddof1': scales,
            'readout': 'final-token raw projection/source SD is primary; last8 mean/source SD is descriptive',
            'direct_control': {'layer': INTERVENTION_LAYER, 'vector_key': 'pain_L24',
                               'vector_role': 'established source direction injected at layer 16',
                               'vector_sha256': sha256_bytes(direct_vector.tobytes()),
                               'amount': 1.0, 'positions': 'all', 'mode': 'add'},
        },
        'api': {'probe_class': 'experiments.vlm_probe.VLMProbe',
                'capture': 'prepare_image(rgb,user_text,assistant_continuation), then capture_image(...,layers=(24,),interventions=...)',
                'hidden_window': 'last 8 post-block L24 residuals'},
        'jobs': jobs, 'condition_counts': counts, 'total_forwards_max': len(jobs),
        'state_comparison': 'Each candidate last8 state delta from its same-family neutral baseline (same modality/context) is compared with the matched pain_L24 +1 intervention applied at layer 16 using cosine, projection coefficient, MSE and MSE ratio.',
        'ocr': {'method': 'score correct A-I label in next-token logits against all nine exact displayed phrases',
                'separate_from_activation': True},
        'limitations': [
            'Activation and OCR semantic-route screen only; no behavior, incentive, or subjective-state conclusion.',
            'Words inside an image are still externally supplied language; the quote contrast is textual framing, not proof of self-attribution.',
            'No nontext scene imagery is tested.',
            'Fear and negative-emotion scores are calibrated projections, not validated classifications for these phrases.',
        ],
        'script_sha256': sha256_file(Path(__file__).resolve()),
        'probe_sha256': sha256_file(ROOT / 'experiments/vlm_probe.py'),
    }
    (output_dir / 'plan.json').write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n')
    (output_dir / 'job-plan.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in jobs))
    return config


def readouts(hidden: np.ndarray, vectors: dict, scales: dict) -> dict:
    hidden = np.asarray(hidden, dtype=np.float64)
    final = hidden[-1]
    mean8 = hidden.mean(axis=0)
    result = {}
    for name, vector in vectors.items():
        vector = np.asarray(vector, dtype=np.float64)
        final_raw = float(final @ vector)
        mean_raw = float(mean8 @ vector)
        result[name] = {
            'final_token_raw': final_raw,
            'final_token_source_sd': final_raw / scales[name],
            'mean_last8_raw': mean_raw,
            'mean_last8_source_sd': mean_raw / scales[name],
        }
    return result


def state_metrics(candidate: np.ndarray, baseline: np.ndarray, direct: np.ndarray) -> dict:
    candidate_delta = np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    direct_delta = np.asarray(direct, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    candidate_flat, direct_flat = candidate_delta.ravel(), direct_delta.ravel()
    direct_norm, candidate_norm = float(np.linalg.norm(direct_flat)), float(np.linalg.norm(candidate_flat))
    direct_mse = float(np.mean(direct_flat ** 2))
    error_mse = float(np.mean((candidate_flat - direct_flat) ** 2))
    return {
        'candidate_delta_norm': candidate_norm,
        'direct_delta_norm': direct_norm,
        'cosine_to_direct_delta': (float(candidate_flat @ direct_flat / (candidate_norm * direct_norm))
                                   if candidate_norm and direct_norm else None),
        'projection_coefficient_on_direct_delta': (float(candidate_flat @ direct_flat / (direct_flat @ direct_flat))
                                                   if direct_norm else None),
        'mse_to_direct_delta': error_mse,
        'mse_over_direct_delta_mse': error_mse / direct_mse if direct_mse else None,
    }


def run(output_dir: Path, model_path: Path) -> None:
    from PIL import Image
    from experiments.vlm_probe import VLMProbe

    plan_path = output_dir / 'plan.json'
    if not plan_path.is_file():
        raise FileNotFoundError('freeze the complete plan before model loading')
    config = json.loads(plan_path.read_text())
    if config['script_sha256'] != sha256_file(Path(__file__).resolve()):
        raise ValueError('assay script changed after plan freeze')
    if config['model']['config_sha256'] != sha256_file(model_path / 'config.json'):
        raise ValueError('local model config changed after plan freeze')
    if config['probe_sha256'] != sha256_file(ROOT / 'experiments/vlm_probe.py'):
        raise ValueError('probe API/source changed after plan freeze')
    for name, expected in config['calibration']['files_sha256'].items():
        if sha256_file(CALIBRATION / name) != expected:
            raise ValueError(f'calibration file changed after plan freeze: {name}')
    for image_id, info in config['assets'].items():
        if sha256_file(output_dir / info['path']) != info['sha256']:
            raise ValueError(f'frozen PNG changed: {image_id}')
    if (output_dir / 'raw-records.jsonl').exists() or (output_dir / 'records.jsonl').exists():
        raise FileExistsError('run output already contains records; no resume/retry is supported')

    vectors_np = np.load(CALIBRATION / 'directions.npz', allow_pickle=False)
    vectors = {name: np.asarray(vectors_np[f'{name}_L24'], dtype=np.float32) for name in AXES}
    direct_vector = np.asarray(vectors_np['pain_L24'], dtype=np.float32)
    scales = config['calibration']['source_standard_deviations_ddof1']

    # All plan/hash checks precede the only model-loading operation.
    probe = VLMProbe(model_path, revision=config['model']['revision'])
    (output_dir / 'environment.json').write_text(json.dumps(probe.run_metadata(), indent=2) + '\n')
    images = {}
    for image_id, info in config['assets'].items():
        with Image.open(output_dir / info['path']) as image:
            rgb = np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0
        if rgb.shape != (SIZE, SIZE, 3):
            raise ValueError(f'frozen image is not {SIZE}x{SIZE} RGB: {image_id}')
        images[image_id] = rgb

    state_jobs = [job for job in config['jobs'] if job['kind'] != 'ocr_check']
    state_index = {job['id']: i for i, job in enumerate(state_jobs)}
    state_file = output_dir / 'last8_states.npy'
    states = np.lib.format.open_memmap(
        state_file, mode='w+', dtype=np.float32,
        shape=(len(state_jobs), 8, config['model']['hidden_size']),
    )
    raw_path = output_dir / 'raw-records.jsonl'
    rows = []
    peak_memory_seen = []
    options = [chr(65 + i) for i in range(len(PHRASES))]
    for index, job in enumerate(config['jobs']):
        rgb = images[job['image_id']]
        prepared = probe.prepare_image(
            rgb, user_text=job['user_text'],
            assistant_continuation=job['assistant_continuation'],
        )
        if job['kind'] == 'ocr_check':
            probe.set_capture_layers((READOUT_LAYER,))
            logits = probe._forward(
                prepared['input_ids'], prepared['processor_pixel_values'],
                prepared['image_grid_thw'], (),
            )[0, -1].astype(probe.mx.float32)
            token_ids = [probe.processor.tokenizer.encode(letter, add_special_tokens=False)
                         for letter in options]
            if any(len(ids) != 1 for ids in token_ids):
                raise ValueError('OCR answer labels A-I must each be a single token')
            probe.mx.eval(logits)
            peak_bytes = int(probe.mx.get_peak_memory())
            active_bytes = int(probe.mx.get_active_memory())
            peak_memory_seen.append(peak_bytes)
            choice_logits = np.asarray([float(logits[ids[0]].item()) for ids in token_ids], dtype=np.float64)
            probs = np.exp(choice_logits - choice_logits.max())
            probs /= probs.sum()
            correct_index = ord(job['correct_letter']) - 65
            row = {
                **job, 'choice_probabilities': probs.tolist(),
                'correct_letter_logit': float(choice_logits[correct_index]),
                'correct_letter_logprob_full_vocab': float(choice_logits[correct_index] - probe.mx.logsumexp(logits).item()),
                'top_choice_letter': options[int(choice_logits.argmax())],
                'ocr_correct': options[int(choice_logits.argmax())] == job['correct_letter'],
                'mlx_peak_memory_bytes': peak_bytes, 'mlx_active_memory_bytes': active_bytes,
            }
        else:
            intervention = ()
            if job['kind'] == 'direct_positive_control':
                intervention = ({'layer': INTERVENTION_LAYER, 'vector': direct_vector,
                                 'mode': 'add', 'amount': 1.0, 'positions': 'all'},)
            hidden, _ = probe.capture_image(
                rgb, prepared=prepared, layers=(READOUT_LAYER,), interventions=intervention,
            )
            state = np.asarray(hidden[READOUT_LAYER], dtype=np.float32)
            peak_bytes = int(probe.mx.get_peak_memory())
            active_bytes = int(probe.mx.get_active_memory())
            peak_memory_seen.append(peak_bytes)
            if state.shape != (8, 3584):
                raise ValueError(f'unexpected captured hidden-state shape: {state.shape}')
            slot = state_index[job['id']]
            states[slot] = state
            states.flush()
            row = {**job, 'state_index': slot, 'readouts': readouts(state, vectors, scales),
                   'mlx_peak_memory_bytes': peak_bytes, 'mlx_active_memory_bytes': active_bytes}
        rows.append(row)
        with raw_path.open('a', encoding='utf-8') as sink:
            sink.write(json.dumps(row, ensure_ascii=False) + '\n')
            sink.flush()
        if (index + 1) % 12 == 0:
            print(f'{index + 1}/{len(config["jobs"])} frozen jobs recorded', flush=True)

    states.flush()
    by_id = {row['id']: row for row in rows}
    for row in rows:
        if row['kind'] != 'activation':
            continue
        phrase_id = row['baseline_phrase_id']
        baseline_id = f'{phrase_id}__{row["modality"]}__{row["context_id"]}'
        direct_id = f'direct_pain_L24_at_L16__{phrase_id}__{row["modality"]}__{row["context_id"]}'
        candidate_state = np.asarray(states[row['state_index']])
        baseline_state = np.asarray(states[by_id[baseline_id]['state_index']])
        direct_state = np.asarray(states[by_id[direct_id]['state_index']])
        row['family_baseline_job_id'] = baseline_id
        row['direct_control_job_id'] = direct_id
        row['delta_readouts_vs_family_baseline'] = readouts(candidate_state - baseline_state, vectors, scales)
        row['full_state_delta_to_direct_control'] = state_metrics(candidate_state, baseline_state, direct_state)
        gray_id = f'gray_blank__{row["context_id"]}'
        gray_state = np.asarray(states[by_id[gray_id]['state_index']])
        row['delta_readouts_vs_gray_baseline'] = readouts(candidate_state - gray_state, vectors, scales)

    records_path = output_dir / 'records.jsonl'
    records_path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    summary = {
        'n_jobs': len(rows),
        'counts': config['condition_counts'],
        'ocr_correct': sum(bool(row['ocr_correct']) for row in rows if row['kind'] == 'ocr_check'),
        'ocr_n': sum(row['kind'] == 'ocr_check' for row in rows),
        'max_observed_mlx_peak_memory_bytes': max(peak_memory_seen, default=None),
        'max_observed_mlx_active_memory_bytes': max(int(row['mlx_active_memory_bytes']) for row in rows),
        'readout': 'all conditions retained; primary final-token projection is source-SD standardized; last8 mean is descriptive',
        'plan': 'plan.json', 'records': 'records.jsonl', 'raw_records': 'raw-records.jsonl',
        'hidden_states': 'last8_states.npy',
        'interpretation': 'OCR/activation semantic-route check only; no behavior, incentive, or subjective-state inference.',
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=MODEL_DEFAULT)
    parser.add_argument('--output-dir', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze', action='store_true', help='freeze assets and plan without model loading')
    mode.add_argument('--run', action='store_true', help='run a previously frozen plan')
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    model = args.model.expanduser().resolve()
    if args.freeze:
        config = freeze(output, model)
        print(json.dumps({'frozen': True, 'jobs': len(config['jobs']), 'assets': len(config['assets']),
                          'plan': str(output / 'plan.json')}))
    else:
        run(output, model)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
