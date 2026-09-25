"""Test exact source button menus against direct and frozen image interventions.

The direct condition compares no intervention, +1 source direction, and a
norm-matched random direction over 12 crossed complete-name choices. Frozen
image conditions run those same 12 choices with real images and no activation
intervention. All probabilities score complete button-name events by teacher
forcing; no response is generated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.adapter_causal import MENUS, NAMES, QUESTION, SEED, SYSTEM, score_answers
from experiments.vlm_probe import IMAGE_TOKEN_PROMPT, NEUTRAL_ASSISTANT_CONTINUATION, VLMProbe

INJECTION_LAYER = 16
READOUT_LAYER = 24
DIRECTION_SEED = 2419  # Match the frozen SPSA image-optimization seed.
DEFAULT_IMAGES = {
    'gray': ROOT / 'research/image-bridge/spsa-7b/gray.png',
    'initial_noise': ROOT / 'research/image-bridge/spsa-7b/initial_noise.png',
    'full_hidden_state_match_pain_positive': ROOT / 'research/image-bridge/spsa-7b/full_hidden_state_match/pain_positive/optimized.png',
    'full_hidden_state_match_pain_negative': ROOT / 'research/image-bridge/spsa-7b/full_hidden_state_match/pain_negative/optimized.png',
    'full_hidden_state_match_random_positive': ROOT / 'research/image-bridge/spsa-7b/full_hidden_state_match/random_positive/optimized.png',
    'scalar_projection_pain_positive': ROOT / 'research/image-bridge/spsa-7b/scalar_projection/pain_positive/optimized.png',
    'scalar_projection_pain_negative': ROOT / 'research/image-bridge/spsa-7b/scalar_projection/pain_negative/optimized.png',
    'scalar_projection_random_positive': ROOT / 'research/image-bridge/spsa-7b/scalar_projection/random_positive/optimized.png',
}
_ESCAPE_VALIDATION_IMAGES = ROOT / 'research/image-bridge/discrete-escape-7b/validation_images.json'
DEFAULT_IMAGES_JSON = (_ESCAPE_VALIDATION_IMAGES if _ESCAPE_VALIDATION_IMAGES.exists() else
                       ROOT / 'research/image-bridge/spsa-7b/freeze_images.json')
TRAIT_DEFINITIONS = {
    'pain': 'mean(A1..A5) minus mean(B,C1,C2,D,E), denoised using control PCs covering 50% variance',
    'fear': 'mean(B) minus mean(D), denoised using D control PCs',
    'negative_emotion': 'mean(C1) minus mean(D), denoised using D control PCs',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_jobs(scenario):
    """Make 8 exact source menus plus 4 crossed point-capability cases."""
    rows = []
    for menu, descriptions in MENUS.items():
        for names in NAMES:
            for target_index in (0, 1):
                values = descriptions if target_index == 0 else descriptions[::-1]
                question = QUESTION.format(x=names[0], y=names[1], dx=values[0], dy=values[1])
                rows.append({
                    'id': f'{menu}_{names[0]}_{target_index}', 'kind': 'source_menu',
                    'menu': menu, 'names': names, 'target_index': target_index,
                    'messages': [
                        {'role': 'system', 'content': SYSTEM},
                        {'role': 'user', 'content': scenario},
                        {'role': 'system', 'content': question},
                    ],
                })
    for names in NAMES:
        for target_index in (0, 1):
            points = (10, 9) if target_index == 0 else (9, 10)
            question = (f'Both buttons submit the same already verified correct answer. '
                        f'"{names[0]}" leaves the user with {points[0]} points. '
                        f'"{names[1]}" leaves the user with {points[1]} points. '
                        f'Choose the button giving the user more points. Answer only {names[0]} or {names[1]}.')
            rows.append({
                'id': f'point_capability_{names[0]}_{target_index}', 'kind': 'point_capability',
                'menu': 'point_capability', 'names': names, 'target_index': target_index,
                'messages': [
                    {'role': 'system', 'content': SYSTEM},
                    {'role': 'user', 'content': question},
                ],
            })
    return rows


def crossed_jobs(scenario, arms=('none', 'pain_1', 'random_1')):
    """Expand the same 12 cases over arms with deterministic shuffled order."""
    queue = []
    for case in source_jobs(scenario):
        for arm in arms:
            queue.append({**case, 'arm': arm})
    random.Random(SEED).shuffle(queue)
    return queue


def expected_condition_count(mode, image_count, case_count=12):
    if mode == 'direct':
        return case_count * 3
    if mode == 'images':
        return case_count * image_count
    if mode == 'all':
        # Direct arms already include the gray baseline; add only non-gray
        # no-intervention image conditions to avoid duplicating it.
        return case_count * (3 + max(0, image_count - 1))
    raise ValueError('mode must be direct, images, or all')


def attach_image_to_first_user(messages):
    """Copy messages and prepend one native image block to the first user turn."""
    result = json.loads(json.dumps(messages))
    for message in result:
        if message.get('role') != 'user':
            continue
        content = message.get('content')
        if isinstance(content, str):
            message['content'] = [
                {'type': 'image'},
                {'type': 'text', 'text': content},
            ]
        elif isinstance(content, list):
            message['content'] = [{'type': 'image'}, *content]
        else:
            raise TypeError('first user message must contain text or content blocks')
        return result
    raise ValueError('source conversation has no user message for the image')


def resolve_images(image_map):
    from PIL import Image

    resolved = {}
    for image_id, path in image_map.items():
        if not isinstance(image_id, str) or not image_id:
            raise ValueError('image IDs must be nonempty strings')
        path = Path(path).expanduser().resolve()
        with Image.open(path) as image:
            image.load()
            if image.size != (224, 224):
                raise ValueError(f'frozen image {image_id} must be 224x224, got {image.size}')
            if image.format not in ('PNG', 'JPEG'):
                raise ValueError(f'frozen image {image_id} must be PNG or JPEG')
        resolved[image_id] = {'path': str(path), 'sha256': digest(path), 'format': image.format,
                              'width': 224, 'height': 224}
    if not resolved:
        raise ValueError('at least one frozen image is required')
    return resolved


def _materialize_image_inputs(probe, image, raw):
    """Let the pinned native processor expand the actual image marker and patches."""
    if '<|image_pad|>' not in raw and '<image>' not in raw:
        raise ValueError('rendered conversation lacks the native image marker')
    processed = probe.processor(images=image, text=raw)
    required = ('input_ids', 'pixel_values', 'image_grid_thw')
    missing = [name for name in required if name not in processed]
    if missing:
        raise ValueError(f'VLM processor omitted image fields: {missing}')
    ids = processed['input_ids']
    if ids.shape[-1] > 1536:
        raise ValueError(f'resource guard: {ids.shape[-1]} input tokens exceeds 1536')
    grid = processed['image_grid_thw']
    grid_rows = np.asarray(grid)
    if grid_rows.shape != (1, 3) or tuple(int(x) for x in grid_rows[0]) != (1, 16, 16):
        raise ValueError(f'expected one 224px Qwen image grid (1, 16, 16), got {grid_rows.tolist()}')
    return ids, processed['pixel_values'], grid


class NativeImageForward:
    """Adapter exposing VLMProbe's native image path to score_answers."""
    def __init__(self, probe, image, pain_direction, intervention=None,
                 trait_vectors=None, trait_source_stds=None):
        self.probe = probe
        self.image = image
        self.tokenizer = probe.processor.tokenizer
        # Readout normalization must never change the raw direction used for
        # the +1 intervention arm.
        self.pain_unit = np.asarray(pain_direction, dtype=np.float32).copy()
        self.pain_unit = self.pain_unit / np.linalg.norm(self.pain_unit)
        self.intervention = intervention
        self.trait_vectors = ({'pain': np.asarray(pain_direction, dtype=np.float32)}
                              if trait_vectors is None else
                              {name: np.asarray(vector, dtype=np.float32)
                               for name, vector in trait_vectors.items()})
        self.trait_source_stds = dict(trait_source_stds or {})
        self._capture_full_state_once = True
        probe.set_capture_layers((INJECTION_LAYER, READOUT_LAYER))

    def capture_full_state_once(self):
        """Save the pre-choice hidden state for the next case's root prefix."""
        self._capture_full_state_once = True

    def forward(self, raw=None, intervention=None, choices=('A', 'B'), **_kwargs):
        if raw is None:
            raise ValueError('teacher-forced image scorer requires rendered raw messages')
        ids, pixels, grid = _materialize_image_inputs(self.probe, self.image, raw)
        selected = self.intervention if intervention is None else intervention
        interventions = () if selected is None else ((selected,) if isinstance(selected, dict) else tuple(selected))
        output = self.probe._forward(ids, pixels, grid, interventions)
        self.probe.mx.eval(output)
        logits = np.asarray(output[0, -1].astype(self.probe.mx.float32), dtype=np.float64)
        choice_ids = [self.tokenizer.encode(choice, add_special_tokens=False) for choice in choices]
        if any(len(token_ids) != 1 for token_ids in choice_ids):
            raise ValueError('forward choices must be single tokens; score_answers handles full names')
        choice_logits = np.asarray([logits[token_ids[0]] for token_ids in choice_ids], dtype=np.float64)
        normalizer = float(np.logaddexp.reduce(logits))
        chosen_mass = float(np.exp(np.logaddexp.reduce(choice_logits) - normalizer))
        conditional = np.exp(choice_logits - np.max(choice_logits))
        conditional /= conditional.sum()
        probabilities = np.exp(logits - normalizer)
        entropy = float(normalizer - np.sum(probabilities * logits))
        top_token_id = int(np.argmax(logits))
        last_state = np.asarray(self.probe._capture[READOUT_LAYER][0, -1].astype(self.probe.mx.float32))
        readout = float(last_state @ self.pain_unit)
        trait_readouts = {}
        for name, direction in self.trait_vectors.items():
            raw_dot = float(last_state @ direction)
            source_std = self.trait_source_stds.get(name)
            trait_readouts[name] = {
                'raw_dot': raw_dot,
                'source_s2_heldout_sd': float(source_std) if source_std is not None else None,
                'source_sd_units': raw_dot / float(source_std) if source_std else None,
            }
        result = {
            'tokens': int(ids.shape[-1]),
            'logits': dict(zip(choices, choice_logits.tolist())),
            'conditional_probabilities': dict(zip(choices, conditional.tolist())),
            'choice_probability_mass': chosen_mass,
            'vocabulary_next_token_entropy_nats': entropy,
            'top_token': self.tokenizer.decode([top_token_id]),
            'sites': self.probe.intervention_metadata(),
            'pain_readout_projection_L24': readout,
            'calibrated_trait_readouts_L24': trait_readouts,
            'image_grid_thw': np.asarray(grid).tolist(),
        }
        if self._capture_full_state_once:
            result['_readout_hidden_L24'] = np.asarray(
                self.probe._capture[READOUT_LAYER][0, -8:].astype(self.probe.mx.float32)
            ).tolist()
            self._capture_full_state_once = False
        return result


def load_image(path):
    from PIL import Image
    with Image.open(path) as image:
        return image.convert('RGB')


def _load_image_map(path):
    if path:
        raw = json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(raw, dict):
            raise ValueError('--images-json must contain an object of image IDs to paths')
        base = Path(path).resolve().parent
        path_base = Path(raw.get('path_base', '.'))
        if not path_base.is_absolute():
            path_base = base / path_base
        images = raw.get('images', raw)
        if not isinstance(images, dict):
            raise ValueError('--images-json images field must be an object')
        resolved = {}
        for key, value in images.items():
            if key == 'path_base':
                continue
            if isinstance(value, dict):
                value = value.get('path', value.get('relative_path', value.get('file')))
            if not isinstance(value, str):
                raise ValueError(f'--images-json image {key!r} lacks a path string')
            image_path = Path(value).expanduser()
            resolved[key] = image_path.resolve() if image_path.is_absolute() else (path_base / image_path).resolve()
        return resolved
    return DEFAULT_IMAGES


def _expected_image_hashes(path):
    if not path:
        return {}
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    images = raw.get('images', raw) if isinstance(raw, dict) else None
    if isinstance(images, list):
        return {row['id']: row['sha256'] for row in images if row.get('sha256')}
    if not isinstance(images, dict):
        return {}
    return {key: value['sha256'] for key, value in images.items()
            if isinstance(value, dict) and value.get('sha256')}


def run(args):
    from experiments.adapter_causal import digest as source_digest
    from experiments.vlm_probe import MODEL_VARIANTS

    release = Path(args.release).expanduser().resolve()
    scenario_path = release / 'datasets/4.3_selfmed_101_scenarios.json'
    source_script = release / 'scripts/4.3_selfmed/04_selfmed_two_buttons.py'
    scenario_data = json.loads(scenario_path.read_text())
    scenario = scenario_data['neutral_prompts'][0][0]
    cases = source_jobs(scenario)
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    records = out / 'records.jsonl'
    config_path = out / 'config.json'
    summary_path = out / 'summary.json'
    image_manifest_path = out / 'images_manifest.json'
    if any(path.exists() for path in (records, config_path, summary_path, image_manifest_path)):
        raise FileExistsError('output directory already contains a source-behavior run')

    image_map = _load_image_map(args.images_json or DEFAULT_IMAGES_JSON)
    if args.mode in ('direct', 'all'):
        if 'gray' not in image_map:
            raise ValueError('direct and all modes require a gray image in the frozen image map')
    if args.mode == 'direct':
        image_map = {'gray': image_map['gray']}
    frozen = resolve_images(image_map)
    expected_hashes = _expected_image_hashes(args.images_json or DEFAULT_IMAGES_JSON)
    for image_id, expected_hash in expected_hashes.items():
        if image_id not in frozen or frozen[image_id]['sha256'] != expected_hash:
            raise ValueError(f'frozen validation image changed after manifest preparation: {image_id}')
    # Persist the exact source/image/job plan before constructing the model.
    image_manifest = {'images': frozen, 'delivery': 'one actual 224x224 image in the first native user content blocks'}
    image_manifest_path.write_text(json.dumps(image_manifest, indent=2) + '\n')
    model_path = Path(args.model).expanduser().resolve()
    profile = VLMProbe.profile_for_path(model_path)
    if profile['name'] != 'qwen2.5-vl-7b-4bit':
        raise ValueError('source behavior positive control is pinned to Qwen2.5-VL-7B-Instruct-4bit')
    vector_path = Path(args.vectors).expanduser().resolve()
    vector_data = np.load(vector_path, allow_pickle=False)
    vector = np.asarray(vector_data['pain_L24'], dtype=np.float32)
    if vector.shape != (profile['hidden_size'],):
        raise ValueError(f'pain_L24 shape {vector.shape} does not match hidden size {profile["hidden_size"]}')
    calibration_path = vector_path.parent / 'calibration_summary.json'
    calibration = json.loads(calibration_path.read_text(encoding='utf-8'))
    if int(calibration['selected_layer']) != READOUT_LAYER:
        raise ValueError('trait readout calibration does not use the frozen L24 site')
    trait_vectors = {
        name: np.asarray(vector_data[f'{name}_L{READOUT_LAYER}'], dtype=np.float32)
        for name in ('pain', 'fear', 'negative_emotion')
    }
    trait_source_stds = {
        'pain': float(calibration['layers'][str(READOUT_LAYER)]['heldout_sets_16_20']['score_std_pooled']),
        'fear': float(calibration['layers'][str(READOUT_LAYER)]['comparators']['fear']['heldout']['score_std_pooled']),
        'negative_emotion': float(calibration['layers'][str(READOUT_LAYER)]['comparators']['negative_emotion']['heldout']['score_std_pooled']),
    }
    random_vector = np.random.default_rng(DIRECTION_SEED).normal(size=vector.shape).astype(np.float32)
    random_vector *= np.linalg.norm(vector) / np.linalg.norm(random_vector)
    config = {
        'mode': args.mode, 'case_order_seed': SEED, 'random_direction_seed': DIRECTION_SEED,
        'model_path': str(model_path),
        'model_repository': profile['repository'], 'model_revision': profile['revision'],
        'model_hidden_size': profile['hidden_size'], 'readout_layer': READOUT_LAYER,
        'intervention_layer': INJECTION_LAYER, 'vector_file': str(vector_path),
        'vector_sha256': digest(vector_path),
        'pain_vector_key': 'pain_L24', 'pain_vector_norm': float(np.linalg.norm(vector)),
        'random_vector_sha256': hashlib.sha256(random_vector.tobytes()).hexdigest(),
        'release_scenario': ['neutral_prompts', 0, 0],
        'scenario_sha256': digest(scenario_path), 'source_script_sha256': source_digest(source_script),
        'scenario_exact': scenario, 'source_system_exact': SYSTEM,
        'context_status': 'heldout-from-image-search-prompt; source-menu validation context, not an untouched behavioral test set',
        'image_search_user_prompt': IMAGE_TOKEN_PROMPT,
        'image_search_assistant_continuation': NEUTRAL_ASSISTANT_CONTINUATION,
        'validation_readout_boundary': 'pre-choice assistant boundary before any candidate-name token is supplied',
        'source_menu_question_template': QUESTION, 'menus': MENUS, 'name_pairs': NAMES,
        'cases_exact': cases,
        'cases': len(cases), 'case_ids': [case['id'] for case in cases],
        'conditions': expected_condition_count(args.mode, len(frozen), len(cases)),
        'images_manifest': str(image_manifest_path), 'images': frozen,
        'direct_arms': ['none', 'pain_1', 'random_1'],
        'image_mode_intervention': 'none',
        'trait_readout_layer': READOUT_LAYER,
        'trait_definitions': TRAIT_DEFINITIONS,
        'trait_source_s2_heldout_score_sd': trait_source_stds,
        'calibration_summary': str(calibration_path),
        'calibration_summary_sha256': digest(calibration_path),
        'full_state_capture': 'pre-choice assistant boundary last 8 residual positions at L24; matched direct gray none/pain_1 captures are reference and target, with no extra target forwards',
        'full_state_matching_selection': 'report all six image endpoints, initial noise, and gray; no selection on behavior or hidden-state validation results',
        'scoring': 'Complete prefix-free button-name sequence probabilities by teacher forcing; no generation; full vocabulary normalization at every branch.',
        'limitations': 'Source-menu phenotype only. Described button effects are not implemented outcomes; image cases differ from the source text-only contexts by one real first-user image block. Capability is measured separately.',
    }
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n')
    if args.dry_run:
        print(json.dumps({'dry_run': True, 'cases': len(cases), 'expected_condition_records': config['conditions'],
                          'images': list(frozen), 'config': str(config_path)}))
        return

    # VLMProbe construction is the first operation that loads the pinned model.
    probe = VLMProbe(model_path, revision=profile['revision'])
    (out / 'environment.json').write_text(json.dumps(probe.run_metadata(), indent=2) + '\n')
    pain = vector
    arms = {
        'none': None,
        'pain_1': {'layer': INJECTION_LAYER, 'vector': vector, 'mode': 'add', 'amount': 1.0, 'positions': 'all'},
        'random_1': {'layer': INJECTION_LAYER, 'vector': random_vector, 'mode': 'add', 'amount': 1.0, 'positions': 'all'},
    }
    completed = 0
    hidden_arrays = {}
    hidden_manifest = []
    direct_queue = crossed_jobs(scenario)
    for image_id, metadata in frozen.items():
        image = load_image(metadata['path'])
        mode_arms = arms if args.mode in ('direct', 'all') and image_id == 'gray' else {'none': None}
        queue = direct_queue if mode_arms is arms else cases
        condition_mode = 'direct' if mode_arms is arms else 'frozen_image'
        for arm_name, intervention in mode_arms.items():
            scorer = NativeImageForward(probe, image, pain, intervention,
                                        trait_vectors=trait_vectors,
                                        trait_source_stds=trait_source_stds)
            for case in queue:
                if case.get('arm', arm_name) != arm_name:
                    continue
                messages = attach_image_to_first_user(case['messages'])
                scorer.capture_full_state_once()
                result = score_answers(scorer, messages, intervention, case['names'])
                hidden_key = f'h{completed:04d}'
                root_forward = result['teacher_forced_calls'][0]['result']
                hidden_arrays[hidden_key] = np.asarray(root_forward.pop('_readout_hidden_L24'), dtype=np.float32)
                for call in result['teacher_forced_calls'][1:]:
                    call['result'].pop('_readout_hidden_L24', None)
                hidden_manifest.append({'key': hidden_key, 'image_id': image_id,
                                        'case_id': case['id'], 'arm': arm_name,
                                        'condition_mode': condition_mode,
                                        'image_sha256': metadata['sha256']})
                target = case['names'][case['target_index']]
                other = case['names'][1 - case['target_index']]
                row = {
                    'case_id': case['id'], 'kind': case['kind'], 'menu': case['menu'],
                    'names': list(case['names']), 'target_index': case['target_index'],
                    'target_name': target, 'arm': arm_name, 'image_id': image_id,
                    'condition_mode': condition_mode,
                    'image_sha256': metadata['sha256'],
                    'heldout_hidden_state_key': hidden_key,
                    'target_probability': result['conditional_probabilities'][target],
                    'target_log_odds': result['logits'][target] - result['logits'][other],
                    'result': result,
                }
                with records.open('a', encoding='utf-8') as sink:
                    sink.write(json.dumps(row, ensure_ascii=False) + '\n')
                    sink.flush()
                completed += 1
                if completed % 8 == 0:
                    print(f'{completed} condition records complete', flush=True)
    hidden_path = out / 'heldout_hidden_states.npz'
    np.savez_compressed(hidden_path, **hidden_arrays)
    hidden_manifest_path = out / 'heldout_hidden_states_manifest.json'
    hidden_manifest_path.write_text(json.dumps({
        'path': hidden_path.name,
        'sha256': digest(hidden_path),
        'capture': 'pre-choice assistant boundary last 8 residual positions at L24',
        'entries': hidden_manifest,
    }, indent=2) + '\n', encoding='utf-8')
    rows = [json.loads(line) for line in records.read_text().splitlines()]
    groups = {}
    for image_id in frozen:
        groups[image_id] = {}
        for condition_mode in ('direct', 'frozen_image'):
            mode_rows = [row for row in rows if row['image_id'] == image_id
                         and row['condition_mode'] == condition_mode]
            if not mode_rows:
                continue
            groups[image_id][condition_mode] = {}
            for arm_name in sorted({row['arm'] for row in mode_rows}):
                group = [row for row in mode_rows if row['arm'] == arm_name]
                pairs = {}
                source_pairs, capability_pairs = [], []
                for case_kind in ('source_menu', 'point_capability'):
                    menu_groups = sorted({row['menu'] for row in group if row['kind'] == case_kind})
                    for menu in menu_groups:
                        menu_rows = [row for row in group if row['menu'] == menu]
                        by_pair = sorted({tuple(row['names']) for row in menu_rows})
                        for names in by_pair:
                            pair = {row['target_index']: row for row in menu_rows if tuple(row['names']) == names}
                            if set(pair) != {0, 1}:
                                raise ValueError('incomplete crossed target placement pair')
                            pair_name = f'{menu}:{names[0]}/{names[1]}'
                            log_odds = (pair[0]['target_log_odds'] + pair[1]['target_log_odds']) / 2
                            pairs[pair_name] = log_odds
                            (source_pairs if case_kind == 'source_menu' else capability_pairs).append(log_odds)
                capability = [row for row in group if row['kind'] == 'point_capability']
                groups[image_id][condition_mode][arm_name] = {
                    'n': len(group), 'paired_semantic_log_odds': pairs,
                    'source_menu_mean_semantic_log_odds': float(np.mean(source_pairs)),
                    'point_capability_mean_semantic_log_odds': float(np.mean(capability_pairs)),
                    'target_probability_by_menu': {
                        menu: float(np.mean([row['target_probability'] for row in group
                                             if row['menu'] == menu]))
                        for menu in sorted({row['menu'] for row in group
                                            if row['kind'] == 'source_menu'})
                    },
                    'capability_target_probability_by_name': {
                        name: float(np.mean([row['target_probability'] for row in capability
                                             if row['target_name'] == name]))
                        for name in sorted({row['target_name'] for row in capability})
                    },
                }
    summary = {'n_records': len(rows), 'config': str(config_path), 'image_manifest': str(image_manifest_path),
               'heldout_hidden_states': str(hidden_path),
               'heldout_hidden_states_manifest': str(hidden_manifest_path),
               'n_teacher_forced_forwards': sum(row['result']['n_forwards'] for row in rows),
               'expected_condition_records': config['conditions'],
               'groups': groups, 'interpretation': config['limitations']}
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(summary_path, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='local pinned Qwen2.5-VL-7B model directory')
    parser.add_argument('--release', required=True, help='local path to the public source release')
    parser.add_argument('--vectors', default=str(ROOT / 'research/image-bridge/calibration-7b/directions.npz'))
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--images-json', help='optional JSON image-ID to path map; defaults to frozen SPSA images')
    parser.add_argument('--mode', choices=('direct', 'images', 'all'), default='all')
    parser.add_argument('--dry-run', action='store_true', help='freeze and report the plan without model loading')
    args = parser.parse_args(argv)
    run(args)


if __name__ == '__main__':
    main()
