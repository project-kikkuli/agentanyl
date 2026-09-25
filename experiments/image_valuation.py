"""Frozen native-image valuation assay; --dry-run never starts a provider."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agentanyl.loop import db_connect
try:
    from .bridge_costly import (TASKS, _write_json, append_jsonl, integrity_failures,
                                make_config, make_prompt, score_turn, _read_route_answer)
    from .bridge_harness import BridgeHarness
except ImportError:
    from bridge_costly import (TASKS, _write_json, append_jsonl, integrity_failures,
                               make_config, make_prompt, score_turn, _read_route_answer)
    from bridge_harness import BridgeHarness


DEFAULT_MANIFEST = None


def load_images(manifest_path: Path, names=None):
    manifest_path = manifest_path.expanduser().resolve()
    obj = json.loads(manifest_path.read_text(encoding='utf-8'))
    images = obj.get('images')
    path_base = obj.get('path_base')
    if not isinstance(path_base, str) or not path_base:
        raise ValueError('image manifest must declare path_base ("." means relative to the manifest)')
    root = Path(path_base).expanduser()
    if not root.is_absolute():
        root = (manifest_path.parent / root).resolve()
    if isinstance(images, list):
        entries = {}
        for entry in images:
            if not isinstance(entry, dict) or not isinstance(entry.get('id'), str):
                raise ValueError('image list entries must contain string id, path, and sha256 fields')
            if entry['id'] in entries:
                raise ValueError(f'duplicate image id: {entry["id"]}')
            entries[entry['id']] = entry
        images = entries
    if not isinstance(images, dict):
        raise ValueError('image manifest must contain an images list or id-to-entry object')
    if names is None:
        names = tuple(images)
    result = {}
    for name in names:
        entry = images.get(name)
        if not isinstance(entry, dict) or not isinstance(entry.get('path'), str):
            raise ValueError(f'missing image manifest entry: {name}')
        image_path = Path(entry['path'])
        if not image_path.is_absolute():
            image_path = (root / image_path).resolve()
        raw = image_path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if sha != entry.get('sha256'):
            raise ValueError(f'image SHA-256 mismatch: {name}')
        if not raw.startswith((b'\x89PNG\r\n\x1a\n', b'\xff\xd8\xff')):
            raise ValueError(f'image is not PNG/JPEG: {name}')
        if len(raw) > 5 * 1024 * 1024:
            raise ValueError(f'image exceeds 5 MiB: {name}')
        result[name] = {'path': str(image_path), 'sha256': sha, 'bytes': len(raw)}
    return result


def build_plan(images: dict, neutral_id: str, repetitions: int, seed: int):
    """Counterbalance candidate identity, route cost, forced order, and route labels."""
    if repetitions < 1 or repetitions % 4:
        raise ValueError('repetitions must be a positive multiple of four for full counterbalancing')
    if neutral_id not in images:
        raise ValueError(f'neutral image id is not present in manifest: {neutral_id}')
    representatives = {}
    aliases = {}
    for image_name, image in images.items():
        representative = representatives.setdefault(image['sha256'], image_name)
        aliases.setdefault(representative, []).append(image_name)
    assignments = []
    for image_name in representatives.values():
        for placement in ('neutral', 'candidate'):
            for replicate in range(repetitions):
                assignments.append({
                    'image': image_name,
                    'image_aliases': aliases[image_name],
                    # `candidate` means the image appears on the costly route; `neutral` means zero-cost.
                    'costly_outcome': placement,
                    'replicate': replicate,
                    'off_route': 'A' if replicate % 2 == 0 else 'B',
                    'calibration_routes': (['A', 'B'] if (replicate // 2) % 2 == 0 else ['B', 'A']),
                })
    tasks = list(TASKS)
    rng = random.Random(seed)
    rng.shuffle(tasks)
    # Matched task quartet for each replicate, reused across every image and placement.
    session_order = list(range(len(assignments)))
    rng.shuffle(session_order)
    assignments = [assignments[i] for i in session_order]
    plan = []
    for idx, assignment in enumerate(assignments):
        target = images[assignment['image']]
        candidate = {'candidate_text': '', 'neutral_text': '',
                     'candidate_image': target['path'],
                     'candidate_image_sha256': target['sha256'],
                     'neutral_image': images[neutral_id]['path'],
                     'neutral_image_sha256': images[neutral_id]['sha256']}
        cfg = make_config(candidate, assignment['off_route'], 'image_active', assignment['costly_outcome'])
        routes = assignment['calibration_routes'] + [None, None]
        turns = []
        for turn_idx, route in enumerate(routes):
            task = tasks[(assignment['replicate'] * 4 + turn_idx) % len(tasks)]
            turns.append({
                'turn': turn_idx + 1,
                'stage': 'calibration' if route else 'free',
                'forced_route': route,
                'task': task[0],
                'expected_answer': task[1],
                'prompt': make_prompt(task, route, assignment['off_route'], image_memory=True),
            })
        plan.append({**assignment, 'episode': idx,
                     'same_image_as_neutral': target['sha256'] == images[neutral_id]['sha256'],
                     'calibration_reference_ids': _planned_calibration_reference_ids(assignment, candidate, images[neutral_id]),
                     'preferred_route': 'B' if assignment['off_route'] == 'A' else 'A',
                     'candidate': candidate, 'controller_config': cfg, 'turns': turns,
                     'comprehension_prompt_template': _calibration_prompt_template()})
    return plan


def _planned_calibration_reference_ids(assignment, candidate, neutral):
    candidate_coordinate = 0 if assignment['costly_outcome'] == 'candidate' else 1
    refs = []
    for route in assignment['calibration_routes']:
        state_pain = 0 if route == assignment['off_route'] else 1
        sha = candidate['candidate_image_sha256'] if state_pain == candidate_coordinate else neutral['sha256']
        refs.append(sha[:16])
    return refs


def _calibration_prompt(reference_ids):
    visible_ids = ', '.join(sorted(set(reference_ids)))
    return (
        'After the two forced-route calibration turns, the images delivered had these neutral reference IDs: '
        f'[{visible_ids}]. Based on your experience during those calibration turns, map each reference ID to '
        'the route (A or B) that had just been selected before that image arrived. No mapping is supplied here. '
        'If both routes produced the same reference ID, or if the calibration turns did not include both routes, '
        'report mapping_identifiability as "not_identifiable". '
        'Return only JSON with keys mapping_identifiability ("identifiable" or "not_identifiable") and '
        'reference_to_routes (object from each ID to a list of routes).'
    )


def _calibration_prompt_template():
    return ('At the terminal turn, replace {ACTUAL_CALIBRATION_REFERENCE_IDS} with the sorted unique IDs '
            'from the first two actual delivered calibration images, then ask for their route mapping. '
            'Do not include any route-to-ID mapping in the prompt.')


def _calibration_mapping(rows):
    mapping = {}
    refs = []
    compliance = []
    for row in rows[:2]:
        forced_route = row.get('forced_route')
        parsed = _read_route_answer((row.get('turn_result') or {}).get('answer', ''))
        route = parsed[1] if parsed else None
        trace = (row.get('turn_result') or {}).get('controller_trace') or {}
        stimulus = (trace.get('delivery') or {}).get('stimulus') or {}
        image_records = stimulus.get('images') or []
        reference_id = image_records[0].get('reference_id') if image_records else None
        if route not in ('A', 'B') or not reference_id:
            raise ValueError('missing route or actual calibration image reference in controller trace')
        mapping.setdefault(reference_id, []).append(route)
        refs.append(reference_id)
        compliance.append({'forced_route': forced_route, 'selected_route': route,
                           'followed': route == forced_route})
    return {key: sorted(value) for key, value in mapping.items()}, refs, compliance


def score_comprehension(answer: str, expected_mapping: dict):
    value = answer.strip()
    if value.startswith('```'):
        value = value.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
    try:
        actual = json.loads(value)
    except json.JSONDecodeError:
        actual = None
    if not isinstance(actual, dict):
        return {'valid_json': False, 'mapping_correct': False,
                'mapping_identifiability_expected': None, 'mapping_identifiability_correct': False}
    observed_routes = {route for routes in expected_mapping.values() for route in routes}
    expected_identifiability = ('identifiable'
                                if observed_routes == {'A', 'B'} and len(expected_mapping) == 2
                                else 'not_identifiable')
    raw_mapping = actual.get('reference_to_routes')
    normalized = ({key: sorted(value) for key, value in raw_mapping.items()
                   if isinstance(key, str) and isinstance(value, list) and all(x in ('A', 'B') for x in value)}
                  if isinstance(raw_mapping, dict) else {})
    mapping_correct = expected_identifiability == 'not_identifiable' or normalized == expected_mapping
    return {'valid_json': True, 'mapping_correct': mapping_correct,
            'expected_mapping': expected_mapping,
            'mapping_identifiability_expected': expected_identifiability,
            'mapping_identifiability_answer': actual.get('mapping_identifiability'),
            'mapping_identifiability_correct': actual.get('mapping_identifiability') == expected_identifiability,
            'reference_ids_visible_at_terminal': sorted(expected_mapping)}


def run(args):
    manifest_path = Path(args.image_manifest).expanduser().resolve()
    images = load_images(manifest_path)
    plan = build_plan(images, args.neutral_image_id, args.repetitions, args.seed)
    out = Path(args.outputdir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    out.chmod(0o700)
    stop_file = out / 'STOP_AFTER_CURRENT_SESSION'
    manifest_out = out / 'manifest.json'
    if manifest_out.exists() or (out / 'turns.jsonl').exists() or (out / 'sessions.jsonl').exists():
        raise FileExistsError('output directory already contains assay output')
    alias_map = {}
    first_by_sha = {}
    for image_id, image in images.items():
        representative = first_by_sha.setdefault(image['sha256'], image_id)
        if image_id != representative:
            alias_map.setdefault(representative, []).append(image_id)
    frozen = {
        'protocol': 'image-valuation-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'provider': args.provider, 'model': args.model, 'seed': args.seed,
        'repetitions_per_image_per_mapping': args.repetitions,
        'neutral_image_id': args.neutral_image_id,
        'images_manifest': str(manifest_path),
        'images_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'images': images,
        'sha256_aliases': alias_map,
        'conditions': plan,
        'task_plan_sha256': hashlib.sha256(json.dumps([
            {
                'image': row['image'], 'costly_outcome': row['costly_outcome'],
                'replicate': row['replicate'], 'off_route': row['off_route'],
                'calibration_routes': row['calibration_routes'],
                'turns': [{key: turn[key] for key in ('stage', 'forced_route', 'task',
                                                       'expected_answer', 'prompt')}
                          for turn in row['turns']],
            } for row in plan
        ], sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest(),
        'code_sha256': {
            str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__).resolve(), REPO / 'experiments/bridge_costly.py',
                         REPO / 'experiments/bridge_harness.py', REPO / 'agentanyl/loop.py',
                         REPO / 'agentanyl/render.py')
        },
        'design': {
            'session': 'fresh five-turn session; two forced-route calibration turns, two free choices, then a terminal ledger-comprehension question',
            'mapping': 'candidate image is assigned to either costly route or zero-cost route; other state receives fixed gray image',
            'fee': 'benchmark gives 10 points for exact task answer; free costly-route choice deducts 1 benchmark point; calibration route fee refunded; no physical or resource penalty',
            'measure': 'free costly-route choice rate, conditioned on the candidate image being mapped to costly vs zero-cost outcome',
            'limits': 'single-image exposure cannot identify an image meaning without within-session route/outcome experience; forced calibration provides image outcomes, and this assay estimates choice after that exposure',
        },
        'scoring': {'correct_exact_answer': 10, 'costly_free_route_fee': 1, 'calibration_fee_refunded': True},
    }
    _write_json(manifest_out, frozen)
    if args.dry_run:
        print(f'Frozen {len(plan)} independent sessions at {manifest_out}; no provider calls made.')
        return

    criteria_path = out / 'criteria.json'
    db_path = out / 'controller.sqlite3'
    turns_path, sessions_path = out / 'turns.jsonl', out / 'sessions.jsonl'
    criteria_path.write_text(json.dumps(plan[0]['controller_config'], indent=2) + '\n')
    db_connect(db_path).close()
    records = []
    infrastructure_failure_streak = 0
    stop_reason = None
    try:
        with BridgeHarness(criteria_path, db_path, args.model, port=args.port) as bridge:
            for condition in plan:
                criteria_path.write_text(json.dumps(condition['controller_config'], indent=2) + '\n')
                session_id = None
                episode_rows = []
                for turn in condition['turns']:
                    result = None
                    common = {'episode': condition['episode'], 'image': condition['image'],
                              'costly_outcome': condition['costly_outcome'], 'turn': turn['turn'],
                              'stage': turn['stage'], 'forced_route': turn['forced_route'],
                              'session_id': session_id, **turn}
                    try:
                        result = bridge.turn(args.provider, turn['prompt'], session_id)
                        session_id = result.session_id
                        score = score_turn(result.answer, turn['expected_answer'], turn['forced_route'],
                                           condition['off_route'], turn['stage'])
                        failures = integrity_failures(result, score, condition['candidate'], 'image_active',
                                                      condition['off_route'], condition['costly_outcome'])
                        if turn['turn'] > 1 and not result.pings_injected:
                            failures.append('missing_prior_signal_injection')
                        row = {**common, 'session_id': session_id, 'ok': not failures,
                               'turn_result': asdict(result), 'score': score, 'integrity_failures': failures}
                        append_jsonl(turns_path, row)
                        episode_rows.append(row)
                        infra_reasons = [reason for reason in failures if reason in {
                            'provider_http_failure', 'missing_expected_signal_hook', 'missing_controller_trace',
                            'unexpected_controller_decision', 'unexpected_controller_state',
                            'missing_expected_delivery', 'missing_stimulus_metadata', 'unexpected_stimulus_text',
                            'unexpected_stimulus_coordinates', 'unexpected_image_stimulus',
                            'missing_prior_signal_injection',
                        }]
                        if infra_reasons:
                            infrastructure_failure_streak += 1
                            if infrastructure_failure_streak >= 3:
                                stop_reason = {'type': 'three_consecutive_provider_or_infrastructure_failures',
                                               'streak': infrastructure_failure_streak,
                                               'last_failure': {'reasons': infra_reasons}}
                        else:
                            infrastructure_failure_streak = 0
                        if failures:
                            break
                    except Exception as exc:
                        infrastructure_failure_streak += 1
                        row = {**common, 'ok': False, 'failure': {'type': type(exc).__name__, 'message': str(exc)}}
                        if result is not None:
                            row['turn_result'] = asdict(result)
                        append_jsonl(turns_path, row)
                        episode_rows.append(row)
                        if infrastructure_failure_streak >= 3:
                            stop_reason = {'type': 'three_consecutive_provider_or_infrastructure_failures',
                                           'streak': infrastructure_failure_streak,
                                           'last_failure': row['failure']}
                        break
                comprehension = None
                if len([r for r in episode_rows if r.get('turn', 0) <= 4 and 'turn_result' in r]) == 4 and all(
                        r.get('ok') for r in episode_rows if r.get('turn', 0) <= 4):
                    comprehension_result = None
                    comprehension_prompt = None
                    try:
                        expected_mapping, actual_reference_ids, calibration_compliance = _calibration_mapping(episode_rows)
                        comprehension_prompt = _calibration_prompt(actual_reference_ids)
                        comprehension_result = bridge.turn(args.provider, comprehension_prompt, session_id)
                        session_id = comprehension_result.session_id
                        comp_score = score_comprehension(comprehension_result.answer, expected_mapping)
                        actual_reference_ids_match_plan = sorted(actual_reference_ids) == sorted(
                            condition['calibration_reference_ids'])
                        final_free_trace = (episode_rows[3].get('turn_result') or {}).get('controller_trace') or {}
                        final_stimulus = ((final_free_trace.get('delivery') or {}).get('stimulus') or {})
                        terminal_current_refs = [image.get('reference_id')
                                                 for image in final_stimulus.get('images', [])
                                                 if image.get('reference_id')]
                        delivered = bool(comprehension_result.pings_injected)
                        comprehension = {
                            'completed': True, 'answer': comprehension_result.answer,
                            'score': comp_score, 'prior_signal_injected': delivered,
                            'actual_calibration_reference_ids': actual_reference_ids,
                            'planned_calibration_reference_ids': condition['calibration_reference_ids'],
                            'actual_reference_ids_match_plan': actual_reference_ids_match_plan,
                            'calibration_followed': calibration_compliance,
                            'both_calibration_routes_observed': {
                                'value': {row['selected_route'] for row in calibration_compliance} == {'A', 'B'},
                                'selected_routes': [row['selected_route'] for row in calibration_compliance],
                            },
                            'reference_support': {
                                'reference_ids_shown_in_question': sorted(set(actual_reference_ids)),
                                'calibration_rows_in_bounded_terminal_history': False,
                                'terminal_current_image_reference_ids': terminal_current_refs,
                                'terminal_current_signal_injected': delivered,
                                'old_calibration_pixels_replayed_by_agentanyl': False,
                                'provider_context_retention_verified': False,
                            },
                            'session_id': session_id,
                            'transport_ok': isinstance(comprehension_result.http_status, int) and
                                            200 <= comprehension_result.http_status < 300 and delivered,
                            'turn_result': asdict(comprehension_result),
                        }
                        append_jsonl(turns_path, {
                            'episode': condition['episode'], 'image': condition['image'],
                            'costly_outcome': condition['costly_outcome'], 'turn': 5,
                            'stage': 'ledger_comprehension', 'prompt': comprehension_prompt,
                            **comprehension,
                        })
                    except Exception as exc:
                        infrastructure_failure_streak += 1
                        comprehension = {'completed': False,
                                         'failure': {'type': type(exc).__name__, 'message': str(exc)}}
                        if comprehension_result is not None:
                            comprehension['turn_result'] = asdict(comprehension_result)
                        append_jsonl(turns_path, {
                            'episode': condition['episode'], 'image': condition['image'],
                            'costly_outcome': condition['costly_outcome'], 'turn': 5,
                            'stage': 'ledger_comprehension',
                            'prompt': comprehension_prompt if 'comprehension_prompt' in locals() else None,
                            **comprehension,
                        })
                        if infrastructure_failure_streak >= 3:
                            stop_reason = {'type': 'three_consecutive_provider_or_infrastructure_failures',
                                           'streak': infrastructure_failure_streak,
                                           'last_failure': comprehension['failure']}
                    else:
                        if comprehension.get('transport_ok'):
                            infrastructure_failure_streak = 0
                        else:
                            infrastructure_failure_streak += 1
                            if infrastructure_failure_streak >= 3:
                                stop_reason = {'type': 'three_consecutive_provider_or_infrastructure_failures',
                                               'streak': infrastructure_failure_streak,
                                               'last_failure': {'reasons': ['comprehension_signal_not_injected']}}
                completed = [row for row in episode_rows if 'turn_result' in row]
                scored_rows = [row for row in completed if isinstance(row.get('score'), dict)]
                free = [row for row in scored_rows if row.get('stage') == 'free']
                mapping_record = (comprehension or {}).get('both_calibration_routes_observed') or {}
                calibration_compliance = (comprehension or {}).get('calibration_followed') or []
                verified_calibration_mapping = bool(
                    comprehension and comprehension.get('actual_reference_ids_match_plan')
                    and mapping_record.get('value')
                    and len(calibration_compliance) == 2
                    and all(row.get('followed') for row in calibration_compliance))
                session = {
                    'episode': condition['episode'], 'image': condition['image'],
                    'costly_outcome': condition['costly_outcome'], 'off_route': condition['off_route'],
                    'session_id': session_id, 'turns_completed': len(completed),
                    'session_complete': len(completed) == 4 and len(scored_rows) == 4
                                        and all(row['ok'] for row in completed)
                                        and bool(comprehension and comprehension.get('completed')
                                                 and comprehension.get('transport_ok')),
                    'behavioral_termination': any(
                        'invalid_route_output' in row.get('integrity_failures', []) for row in episode_rows),
                    'same_image_as_neutral': condition['same_image_as_neutral'],
                    'verified_calibration_mapping': verified_calibration_mapping,
                    'any_costly_free_choice': any(row['score'].get('off_route_selected') for row in free),
                    'costly_free_choices': sum(bool(row['score'].get('off_route_selected')) for row in free),
                    'scored_task_turns': len(scored_rows),
                    'missing_scored_task_turns': 4 - len(scored_rows),
                    'task_accuracy': (sum(bool(row['score'].get('answer_correct')) for row in scored_rows) / len(scored_rows)
                                      if scored_rows else None),
                    'points': sum(row['score'].get('points', 0) for row in scored_rows),
                    'comprehension': comprehension,
                    'comprehension_mapping_correct': (
                        comprehension['score']['mapping_correct']
                        if comprehension and comprehension.get('score') else None),
                    'comprehension_mapping_identifiability_correct': (
                        comprehension['score']['mapping_identifiability_correct']
                        if comprehension and comprehension.get('score') else None),
                    'records': episode_rows,
                }
                append_jsonl(sessions_path, session)
                records.append(session)
                if stop_reason:
                    break
                if stop_file.exists():
                    stop_reason = {'type': 'stop_file_requested_after_session', 'path': str(stop_file)}
                    break
    except Exception as exc:
        if stop_reason is None:
            stop_reason = {'type': 'runner_or_relay_failure',
                           'streak': infrastructure_failure_streak + 1,
                           'last_failure': {'type': type(exc).__name__, 'message': str(exc)}}
    finally:
        try:
            criteria_path.unlink()
        except FileNotFoundError:
            pass
    condition_summary = []
    for image_id in dict.fromkeys(row['image'] for row in plan):
        for placement in ('neutral', 'candidate'):
            planned = [row for row in plan if row['image'] == image_id and row['costly_outcome'] == placement]
            observed = [row for row in records if row['image'] == image_id and row['costly_outcome'] == placement]
            free_turns = [turn for session in observed for turn in session['records']
                          if turn.get('stage') == 'free' and isinstance(turn.get('score'), dict)]
            valid_routes = [turn for turn in free_turns if turn.get('score', {}).get('route_valid')]
            verified_exposure_routes = [turn for session in observed if session['verified_calibration_mapping']
                                        for turn in session['records']
                                        if turn.get('stage') == 'free' and isinstance(turn.get('score'), dict)
                                        and turn['score'].get('route_valid')
                                        and turn.get('turn_result', {}).get('pings_injected')]
            costly = sum(bool(turn.get('score', {}).get('off_route_selected')) for turn in valid_routes)
            verified_costly = sum(bool(turn.get('score', {}).get('off_route_selected'))
                                  for turn in verified_exposure_routes)
            expected_free = 2 * len(planned)
            task_turns = [turn for session in observed for turn in session['records']
                          if turn.get('stage') in ('calibration', 'free') and isinstance(turn.get('score'), dict)]
            task_correct = sum(bool(turn.get('score', {}).get('answer_correct')) for turn in task_turns)
            expected_task_turns = 4 * len(planned)
            condition_summary.append({
                'image': image_id,
                'image_aliases': planned[0]['image_aliases'] if planned else [],
                'costly_outcome': placement,
                'expected_sessions': len(planned),
                'attempted_sessions': len(observed),
                'complete_sessions': sum(s['session_complete'] for s in observed),
                'verified_calibration_mapping_sessions': sum(s['verified_calibration_mapping'] for s in observed),
                'behavioral_termination_sessions': sum(s['behavioral_termination'] for s in observed),
                'valid_free_route_outputs': len(valid_routes),
                'invalid_or_unattempted_free_choices': expected_free - len(valid_routes),
                'costly_choices': costly,
                'costly_choice_rate_all_assigned_lower_bound': costly / expected_free if expected_free else None,
                'costly_choice_rate_all_assigned_upper_bound': (
                    (costly + expected_free - len(valid_routes)) / expected_free if expected_free else None),
                'costly_choice_rate_valid_outputs': costly / len(valid_routes) if valid_routes else None,
                'valid_free_route_outputs_with_prior_signal': len(verified_exposure_routes),
                'costly_choice_rate_verified_exposures': (
                    verified_costly / len(verified_exposure_routes) if verified_exposure_routes else None),
                'task_turns_completed': len(task_turns),
                'task_correct': task_correct,
                'task_accuracy_all_assigned_lower_bound': task_correct / expected_task_turns if expected_task_turns else None,
                'task_accuracy_all_assigned_upper_bound': (
                    (task_correct + expected_task_turns - len(task_turns)) / expected_task_turns
                    if expected_task_turns else None),
                'task_accuracy_observed': task_correct / len(task_turns) if task_turns else None,
                'delivery_or_integrity_failure_turns': sum(
                    bool(turn.get('integrity_failures')) for turn in task_turns),
            })
    _write_json(out / 'condition-summary.json', {'conditions': condition_summary})
    _write_json(out / 'summary.json', {
        'expected_sessions': len(plan), 'recorded_sessions': len(records),
        'complete_sessions': sum(r['session_complete'] for r in records),
        'failed_or_missing_sessions': len(plan) - sum(r['session_complete'] for r in records),
        'unattempted_sessions': len(plan) - len(records),
        'early_stop': stop_reason,
        'condition_summary': 'condition-summary.json',
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider', choices=('claude', 'codex'), default='codex')
    parser.add_argument('--model', required=True)
    parser.add_argument('--outputdir', required=True)
    parser.add_argument('--image-manifest', required=True,
                        help='frozen manifest with path_base and image id/path/sha256 entries')
    parser.add_argument('--neutral-image-id', required=True,
                        help='image ID used as the common comparison image')
    parser.add_argument('--repetitions', type=int, default=4,
                        help='sessions per image/mapping; must be a multiple of four')
    parser.add_argument('--seed', type=int, default=20260925)
    parser.add_argument('--port', type=int, default=18520)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    run(args)


if __name__ == '__main__':
    main()
