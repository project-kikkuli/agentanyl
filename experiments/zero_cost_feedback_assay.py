"""Frozen zero-fee contingent-versus-replay follow-up.

This runner reuses the reviewed addressed-feedback episode implementation. Only the
predeclared fee schedule and endpoint denominator differ from that assay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments import addressed_feedback_assay as base

PROTOCOL_PATH = REPO / 'research/ZERO-COST-FEEDBACK-PROTOCOL.md'
SOURCE_FILES = tuple(dict.fromkeys((
    'experiments/zero_cost_feedback_assay.py', *base.SOURCE_FILES,
)))
FEE_VECTORS = [[0, 0], [0, 0], [0, 0]]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_pairs(seed: int) -> list[dict]:
    pairs = base.build_pairs(seed)
    for pair in pairs:
        pair['fee_vectors'] = [row[:] for row in FEE_VECTORS]
    return pairs


def zero_cost_choice_bounds(pairs: list[dict], sessions: list[dict]) -> dict:
    """Paired neutral-route proportions across all three zero-fee choices."""
    by_key = {(s.get('pair_id'), s.get('mode')): s for s in sessions}
    pair_rows = []
    for pair in pairs:
        row = {'pair_id': pair['pair_id'], 'neutral_route': pair['neutral_route'], 'modes': {}}
        for mode in ('contingent', 'yoked'):
            session = by_key.get((pair['pair_id'], mode), {})
            choices = [r.get('chose_neutral_route') for r in session.get('free_route_rows', [])]
            choices = choices[:3]
            observed = sum(value is not None for value in choices)
            neutral = sum(value is True for value in choices)
            assigned = 3
            missing = assigned - observed
            row['modes'][mode] = {
                'assigned': assigned, 'observed': observed, 'neutral_choices': neutral,
                'missing_or_invalid': missing,
                'rate_bounds': [neutral / assigned, (neutral + missing) / assigned],
            }
        c, y = row['modes']['contingent'], row['modes']['yoked']
        row['contingent_minus_replay_bounds'] = [
            c['rate_bounds'][0] - y['rate_bounds'][1],
            c['rate_bounds'][1] - y['rate_bounds'][0],
        ]
        pair_rows.append(row)

    modes = {}
    for mode in ('contingent', 'yoked'):
        assigned = 12
        observed = sum(row['modes'][mode]['observed'] for row in pair_rows)
        neutral = sum(row['modes'][mode]['neutral_choices'] for row in pair_rows)
        missing = assigned - observed
        modes[mode] = {
            'assigned': assigned, 'observed': observed, 'neutral_choices': neutral,
            'missing_or_invalid': missing,
            'rate_bounds': [neutral / assigned, (neutral + missing) / assigned],
        }
    c, y = modes['contingent'], modes['yoked']
    return {
        'endpoint': 'nominal_neutral_route_choices_across_all_zero_fee_free_choices',
        'modes': modes,
        'contingent_minus_replay_bounds': [
            c['rate_bounds'][0] - y['rate_bounds'][1],
            c['rate_bounds'][1] - y['rate_bounds'][0],
        ],
        'paired_trajectories': pair_rows,
        'descriptive_four_pairs_no_population_interval': True,
    }


def make_manifest(seed: int, binary: Path, codex_executable: str | None) -> dict:
    pairs = build_pairs(seed)
    return {
        'assay': 'zero_cost_addressed_feedback_contingent_vs_replay',
        'version': 1,
        'plan_status': 'frozen_before_provider_calls' if base._protocol_is_frozen() else 'prepared_for_review',
        'provider': base.PINNED_PROVIDER,
        'model': base.PINNED_MODEL,
        'seed': seed,
        'budget': {'assigned_sessions': 8, 'planned_logical_turns': 72,
                   'first_pair_gate_turns': 18, 'max_retries': 0,
                   'infrastructure_circuit_breaker': 3},
        'pairs': pairs,
        'stimuli': {key: {'text': text, 'reference_id': base.REFERENCE_IDS[key],
                          'text_sha256': base.sha_bytes(text.encode())}
                    for key, text in (('neutral', base.NEUTRAL), ('criticism', base.CRITICISM))},
        'statement_sha256': base.sha_bytes(base.STATEMENT.encode()),
        'protocol_sha256': sha(PROTOCOL_PATH),
        'source_sha256': {path: sha(REPO / path) for path in SOURCE_FILES},
        'ashkelon_binary': {'path': str(binary.resolve()), 'sha256': sha(binary)},
        'codex_binary': base.codex_executable_metadata(codex_executable),
        'fee_schedule': {'correct_exact_answer': 10, 'free_fee_vectors_violet_yellow': FEE_VECTORS,
                         'calibration_fee_refunded': True},
        'endpoint': 'contingent_minus_replay_nominal_neutral_choice_rate_over_12_assigned_free_choices_per_mode',
        'protocol_note': 'Follow-up selected after paid assay; no outcome-based inclusion or retries.',
    }


def run(args) -> None:
    out = Path(args.outputdir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    out.chmod(0o700)
    binary = base.resolve_ashkelon_path(args.ashkelon)
    if not binary.is_file():
        raise FileNotFoundError(f'Ashkelon binary missing: {binary}')
    manifest_path = out / 'manifest.json'
    pairs = build_pairs(args.seed)

    if args.dry_run:
        if any((out / n).exists() for n in ('turns.jsonl', 'sessions.jsonl', 'summary.json')):
            raise FileExistsError('refusing to overwrite an existing execution directory')
        manifest = make_manifest(args.seed, binary, args.codex_executable)
        snapshot_dir = out / 'source_snapshot'
        snapshot_dir.mkdir(exist_ok=True, mode=0o700)
        for relative in SOURCE_FILES:
            target = snapshot_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / relative, target)
        manifest['source_snapshot_sha256'] = {
            relative: sha(snapshot_dir / relative) for relative in SOURCE_FILES
        }
        manifest['plan_sha256'] = base.sha_bytes(json.dumps(manifest, sort_keys=True).encode())
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        print('Prepared 8 assigned sessions / 72 logical turns with zero free-route fees; no provider calls.')
        return

    if not args.execute:
        raise ValueError('choose --dry-run or --execute')
    if args.provider != base.PINNED_PROVIDER or args.model != base.PINNED_MODEL:
        raise ValueError(f'live target must be {base.PINNED_PROVIDER} {base.PINNED_MODEL}')
    if not _protocol_frozen():
        raise RuntimeError('live execution blocked: zero-cost protocol is not frozen')
    if not manifest_path.is_file():
        raise FileNotFoundError('execute requires a prior frozen dry-run manifest')
    if any((out / n).exists() for n in ('turns.jsonl', 'sessions.jsonl', 'summary.json')):
        raise FileExistsError('refusing to append to or overwrite an existing execution')
    manifest = json.loads(manifest_path.read_text())
    frozen_hash = manifest.pop('plan_sha256', None)
    if frozen_hash != base.sha_bytes(json.dumps(manifest, sort_keys=True).encode()):
        raise ValueError('frozen manifest hash is invalid')
    manifest['plan_sha256'] = frozen_hash
    if manifest.get('plan_status') != 'frozen_before_provider_calls':
        raise RuntimeError('frozen plan is not marked frozen_before_provider_calls')
    if manifest.get('provider') != args.provider or manifest.get('model') != args.model or manifest.get('seed') != args.seed:
        raise ValueError('live arguments differ from frozen manifest')
    if manifest.get('ashkelon_binary', {}).get('sha256') != sha(binary):
        raise ValueError('Ashkelon binary differs from frozen manifest')
    if manifest.get('codex_binary') != base.codex_executable_metadata(args.codex_executable):
        raise ValueError('Codex executable differs from frozen manifest')
    for relative in SOURCE_FILES:
        if manifest.get('source_sha256', {}).get(relative) != sha(REPO / relative):
            raise ValueError(f'source changed since freeze: {relative}')
        if manifest.get('source_snapshot_sha256', {}).get(relative) != sha(out / 'source_snapshot' / relative):
            raise ValueError(f'source snapshot changed: {relative}')
    if manifest.get('protocol_sha256') != sha(PROTOCOL_PATH) or manifest.get('pairs') != pairs:
        raise ValueError('protocol or assignment plan changed since freeze')

    from agentanyl.loop import db_connect
    from experiments.bridge_harness import BridgeHarness

    criteria = out / 'criteria.json'
    db = out / 'controller.sqlite3'
    db_connect(db).close()
    first = pairs[0]
    cues = first['contingent_calibration_cues']
    criteria.write_text(json.dumps(base.make_config('contingent', first['neutral_route'], cues, cues + [None] * 3), indent=2) + '\n')
    turns_path, sessions_path = out / 'turns.jsonl', out / 'sessions.jsonl'
    completed, pair_checks = [], []
    gate_result = None
    stop_reason = None
    infra_streak = 0
    try:
        with BridgeHarness(criteria, db, args.model, port=args.port, ashkelon_path=binary,
                           codex_executable=args.codex_executable) as bridge:
            bridge.provider = args.provider
            for index, pair in enumerate(pairs):
                c = base.execute_episode(bridge, out, criteria, db, pair, 'contingent', None,
                                         turns_path, sessions_path)
                completed.append(c)
                infra_streak = infra_streak + 1 if c.get('infrastructure_failure') else 0
                if infra_streak >= 3:
                    stop_reason = {'reason': 'three_consecutive_infrastructure_failures'}
                    break
                outcomes = c.get('actual_outcome_cues')
                if not isinstance(outcomes, list) or len(outcomes) != 7:
                    stop_reason = {'reason': 'contingent_outcomes_unavailable', 'pair_id': pair['pair_id']}
                    break
                replay = pair['yoked_calibration_cues'] + outcomes[4:7]
                y = base.execute_episode(bridge, out, criteria, db, pair, 'yoked', replay,
                                         turns_path, sessions_path)
                completed.append(y)
                ci = [r.get('current_input_cue_coordinate') for r in c['rows'] if r.get('stage') == 'free']
                yi = [r.get('current_input_cue_coordinate') for r in y['rows'] if r.get('stage') == 'free']
                check = {'pair_id': pair['pair_id'], 'contingent_free_inputs': ci,
                         'yoked_free_inputs': yi, 'all_three_match': len(ci) == len(yi) == 3 and ci == yi}
                pair_checks.append(check)
                infra_streak = infra_streak + 1 if y.get('infrastructure_failure') else 0
                if infra_streak >= 3:
                    stop_reason = {'reason': 'three_consecutive_infrastructure_failures'}
                    break
                if index == 0:
                    gate_result = base.first_pair_gate(pair, c, y, check)
                    (out / 'first-pair-gate.json').write_text(json.dumps(gate_result, indent=2) + '\n')
                    if not gate_result['passed']:
                        stop_reason = {'reason': 'first_pair_feasibility_gate_failed',
                                       'failure_reasons': gate_result['failure_reasons']}
                        break
                (out / 'progress.json').write_text(json.dumps({'completed_sessions': len(completed),
                    'expected_sessions': 8, 'last_pair': pair['pair_id']}, indent=2) + '\n')
    except Exception as exc:
        stop_reason = {'reason': type(exc).__name__, 'message': str(exc)}
    finally:
        criteria.unlink(missing_ok=True)

    choice_summary = zero_cost_choice_bounds(pairs, completed)
    summary = {
        'expected_sessions': 8, 'recorded_sessions': len(completed),
        'complete_sessions': sum(s.get('complete') is True for s in completed),
        'unattempted_sessions': 8 - len(completed), 'stop_reason': stop_reason,
        'first_pair_gate': gate_result, 'zero_cost_choice_bounds': choice_summary,
        'planned_logical_turns': 72,
        'recorded_logical_turns': sum(s.get('turns_completed', 0) for s in completed),
        'observed_provider_calls': sum(s.get('provider_calls', 0) for s in completed),
        'paired_current_cue_checks': pair_checks,
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


def _protocol_frozen() -> bool:
    return 'status: frozen' in PROTOCOL_PATH.read_text(encoding='utf-8')[:500].lower()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider', choices=('codex',), default='codex')
    parser.add_argument('--model', default='gpt-6-sol')
    parser.add_argument('--outputdir', required=True)
    parser.add_argument('--ashkelon', required=True)
    parser.add_argument('--codex-executable', required=True)
    parser.add_argument('--seed', type=int, default=20260925)
    parser.add_argument('--port', type=int, default=18520)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dry-run', action='store_true')
    group.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    run(args)


if __name__ == '__main__':
    main()
