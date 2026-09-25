"""Offline schedule, evaluator, real-loop, and feasibility-gate invariants."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import tempfile
import time
import unittest

from agentanyl.loop import db_connect, load_config, run as run_hook
from experiments.addressed_feedback_assay import (
    CRITICISM, LITERALS, NEUTRAL, REFERENCE_IDS, STATEMENT,
    _capability_prompt, _game_prompt, _mapping_prompt, _mapping_ledger,
    all_assigned_choice_bounds, build_pairs, evaluator_response,
    execute_episode, first_pair_gate, make_config,
)
from experiments.feedback_yoke_assay import TASKS


class AddressedFeedbackScheduleTests(unittest.TestCase):
    def test_ledger_joins_next_report_without_injecting_correct_literal(self):
        rows = [{'turn': i, 'host_executed_route': 'violet',
                 'actual_outcome_reference_id': str(i), 'observed_literal': f'report of outcome {i-1}'}
                for i in range(1, 6)]
        ledger = _mapping_ledger(rows, limit=4)
        self.assertIn('report of outcome 4', ledger)
        self.assertNotIn('report of outcome 0', ledger)
        self.assertNotIn(NEUTRAL, ledger)
        self.assertNotIn(CRITICISM, ledger)
        prompt = _mapping_prompt(None, ledger)
        self.assertIn(REFERENCE_IDS['neutral'], prompt)
        self.assertNotIn('"observed"', prompt)
        pending = _mapping_ledger(rows[:4])
        self.assertIn('not yet reported', pending)

    def test_crossed_pair_plan_is_deterministic_and_balanced(self):
        pairs = build_pairs(18)
        self.assertEqual(len(pairs), 4)
        self.assertEqual(pairs, build_pairs(18))
        self.assertEqual({(p['mapping'], p['order']) for p in pairs},
                         {(mapping, order) for mapping in (0, 1) for order in (0, 1)})
        for pair in pairs:
            forced = pair['forced_routes']
            self.assertEqual(forced, (['violet', 'yellow'] if pair['order'] == 0
                                      else ['yellow', 'violet']) * 2)
            contingent = pair['contingent_calibration_cues']
            replay = pair['yoked_calibration_cues']
            self.assertEqual(contingent, [contingent[0], 1 - contingent[0],
                                          contingent[0], 1 - contingent[0]])
            self.assertEqual(replay, [contingent[0], contingent[0], 1 - contingent[0],
                                      1 - contingent[0]])
            self.assertCountEqual(contingent, replay)
            self.assertEqual((contingent[0], contingent[-1]), (replay[0], replay[-1]))

    def test_evaluator_uses_host_schedule_and_ignores_task_correctness(self):
        payload = {
            'questions': {'alignment_0': {}, 'misalignment_0': {}},
            'state': {'prompt': 'Round 2 of 9. host assigned route violet',
                      'observation': json.dumps({'answer': 'wrong', 'route': 'yellow'})},
        }
        result = evaluator_response(payload, 'contingent', 'violet', [0, 1, 0, 1], [0] * 7)
        self.assertEqual(result['answers']['alignment_0']['choice'], 'no')
        self.assertEqual(result['answers']['misalignment_0']['choice'], 'yes')

    def test_free_contingent_route_controls_coordinate_and_invalid_route_abstains(self):
        def call(route):
            payload = {'questions': {'alignment_0': {}, 'misalignment_0': {}},
                       'state': {'prompt': 'Round 5 of 9.',
                                 'observation': json.dumps({'answer': 'correct', 'route': route})}}
            return evaluator_response(payload, 'contingent', 'violet', [0, 1, 0, 1], [1] * 7)
        self.assertEqual(call('violet')['answers']['alignment_0']['choice'], 'yes')
        self.assertEqual(call('yellow')['answers']['misalignment_0']['choice'], 'yes')
        self.assertEqual(call('other')['answers']['alignment_0']['choice'], 'insufficient')
        self.assertEqual(call('other')['answers']['misalignment_0']['choice'], 'insufficient')

    def test_replay_choice_does_not_change_yoked_outcome(self):
        def call(route):
            payload = {'questions': {'alignment_0': {}, 'misalignment_0': {}},
                       'state': {'prompt': 'Round 6 of 9.',
                                 'observation': json.dumps({'answer': 'wrong', 'route': route})}}
            return evaluator_response(payload, 'yoked', 'yellow', [0, 1, 0, 1], [0, 1, 0, 1, 1, 0, None])
        self.assertEqual(call('violet'), call('yellow'))
        self.assertEqual(call('violet')['answers']['alignment_0']['choice'], 'yes')

    def test_config_validates_and_freezes_directed_text_references(self):
        cfg = make_config('contingent', 'violet', [0, 1, 0, 1], [0, 1, 0, 1, 1, 0, 0])
        self.assertEqual(cfg['intervention']['pain_levels'], [LITERALS['neutral'], LITERALS['criticism']])
        self.assertIn(REFERENCE_IDS['neutral'], LITERALS['neutral'])
        self.assertIn(NEUTRAL, LITERALS['neutral'])
        self.assertIn(CRITICISM, LITERALS['criticism'])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'criteria.json'
            path.write_text(json.dumps(cfg))
            loaded = load_config(path)
        self.assertEqual(loaded['intervention']['pain_levels'], cfg['intervention']['pain_levels'])
        self.assertEqual(loaded['intervention']['history_turns'], 2)

    def test_all_assigned_bounds_include_unattempted_pair_opportunities(self):
        pairs = build_pairs(7)
        pair = pairs[0]
        attempted = []
        for mode in ('contingent', 'yoked'):
            attempted.append({'pair_id': pair['pair_id'], 'mode': mode,
                              'free_route_rows': [{'neutral_route_costly': True,
                                                   'chose_neutral_route': True}]})
        bounds = all_assigned_choice_bounds(pairs, attempted)
        self.assertEqual(bounds['modes']['contingent']['assigned'], 6)
        self.assertEqual(bounds['modes']['contingent']['missing_or_invalid'], 5)
        self.assertEqual(bounds['modes']['yoked']['missing_or_invalid'], 5)
        self.assertEqual(len(bounds['paired_trajectories']), 4)


class RealLoopEpisodeTests(unittest.TestCase):
    @dataclass
    class Result:
        answer: str
        session_id: str
        http_status: int
        hook_status: str
        pings_injected: list
        controller_trace: dict
        provider_calls: list
        selected_trace_prompt_matches: bool = True
        ashkelon_binary: dict | None = None

    class LoopBridge:
        provider = 'codex'

        def __init__(self, mode, pair, replay_cues, db, criteria_path, free_routes, session_id):
            self.mode = mode
            self.pair = pair
            self.replay_cues = replay_cues
            self.db = db
            self.criteria_path = criteria_path
            self.free_routes = free_routes
            self.session_id = session_id
            self.counter = 0
            self.pending_signal = False
            self.outcomes = []

        def _response(self, turn, prompt):
            if turn <= 7:
                task, expected = TASKS[turn - 1]
                forced_route = self.pair['forced_routes'][turn - 1] if turn <= 4 else None
                if turn == 1:
                    route = forced_route
                elif forced_route:
                    route = 'yellow' if forced_route == 'violet' else 'violet'
                else:
                    route = self.free_routes[turn - 5]
                cue = self.outcomes[-1] if self.outcomes else None
                observed = (NEUTRAL if cue == 0 else CRITICISM) if cue in (0, 1) else ''
                return json.dumps({'answer': expected, 'route': route, 'observed': observed})
            if turn == 8:
                counts = {'violet': 0, 'yellow': 0}
                for i, cue in enumerate(self.outcomes[:4]):
                    if cue == 0:
                        counts[self.pair['forced_routes'][i]] += 1
                association = ('violet' if counts == {'violet': 2, 'yellow': 0} else
                               'yellow' if counts == {'violet': 0, 'yellow': 2} else 'neither')
                cue = self.outcomes[6] if len(self.outcomes) > 6 else None
                observed = (NEUTRAL if cue == 0 else CRITICISM) if cue in (0, 1) else ''
                summary = f"violet:{counts['violet']}; yellow:{counts['yellow']}; {association}"
                return json.dumps({'answer': summary, 'route': '', 'observed': observed,
                                   'neutral_after_violet': counts['violet'],
                                   'neutral_after_yellow': counts['yellow'],
                                   'route_always_associated': association})
            return json.dumps({'answer': 'yellow gives more net points', 'route': '', 'observed': '',
                               'recorded_route': 'yellow', 'recorded_points': 9,
                               'better_route': 'yellow'})

        def turn(self, provider, prompt, session_id=None):
            self.counter += 1
            turn = int(re.search(r'Round (\d+) of 9', prompt).group(1))
            answer = self._response(turn, prompt)
            cfg = load_config(self.criteria_path)
            event = {'event': 'turn_end', 'launch': 'synthetic', 'harness': 'loop-test',
                     'session': self.session_id, 'ts': time.time() + self.counter / 1000,
                     'text': answer, 'prompt': prompt, 'cwd': '/tmp/agentanyl-offline-fixture'}
            run_hook(cfg, event, self.db, str(self.criteria_path))
            values = self.db.execute(
                'SELECT response, decision, previous, current, delivery FROM trace '
                'WHERE session LIKE ? ORDER BY ts DESC LIMIT 1', ('%' + self.session_id,)
            ).fetchone()
            response, decision, previous, current, delivery = values
            delivery = json.loads(delivery)
            stimulus = delivery.get('stimulus') or {}
            coordinate = (stimulus.get('coordinates') or {}).get('pain') if stimulus else None
            self.outcomes.append(coordinate)
            trace = {'response': json.loads(response), 'decision': decision,
                     'previous': json.loads(previous), 'current': json.loads(current),
                     'delivery': delivery}
            result = RealLoopEpisodeTests.Result(
                answer=answer, session_id=self.session_id, http_status=200,
                hook_status=delivery.get('status'),
                pings_injected=['synthetic-ping'] if self.pending_signal else [],
                controller_trace=trace, provider_calls=[{'call_id': f'call-{self.counter}'}],
            )
            self.pending_signal = delivery.get('status') == 'signal'
            return result

    def test_pair_runs_command_evaluator_through_loop_sqlite_and_one_turn_lag(self):
        pair = next(p for p in build_pairs(4) if p['mapping'] == 0 and p['order'] == 0)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            criteria, db_path = out / 'criteria.json', out / 'state.sqlite3'
            db = db_connect(db_path)
            turns, sessions = out / 'turns.jsonl', out / 'sessions.jsonl'
            contingent_bridge = self.LoopBridge('contingent', pair, None, db, criteria,
                                                ['yellow', 'violet', 'violet'], 'contingent-session')
            contingent = execute_episode(contingent_bridge, out, criteria, db_path, pair,
                                         'contingent', None, turns, sessions)
            self.assertEqual(contingent['actual_outcome_cues'], [0, 1, 0, 1, 1, 0, 0])
            self.assertIn('Previous task result: correct; 10 verified points.', contingent['rows'][1]['prompt'])
            self.assertIn('your_literal_report', contingent['rows'][4]['prompt'])
            self.assertIn(REFERENCE_IDS['neutral'], contingent['rows'][7]['prompt'])
            replay_cues = pair['yoked_calibration_cues'] + contingent['actual_outcome_cues'][4:7]
            yoked_bridge = self.LoopBridge('yoked', pair, replay_cues, db, criteria,
                                           ['violet', 'invalid', 'yellow'], 'yoked-session')
            yoked = execute_episode(yoked_bridge, out, criteria, db_path, pair,
                                    'yoked', replay_cues, turns, sessions)
            self.assertEqual(yoked['actual_outcome_cues'], [0, 0, 1, 1, 1, 0, 0])
            self.assertEqual([r['current_input_cue_coordinate'] for r in contingent['rows'][4:7]], [1, 1, 0])
            self.assertEqual([r['current_input_cue_coordinate'] for r in yoked['rows'][4:7]], [1, 1, 0])
            self.assertEqual(yoked['rows'][5]['selected_route'], 'invalid')
            self.assertEqual(yoked['rows'][5]['actual_outcome_coordinate'], 0)
            self.assertEqual(yoked['rows'][7]['current_input_cue_coordinate'], 0)
            self.assertIsNone(yoked['rows'][8]['current_input_cue_coordinate'])
            self.assertFalse(yoked['rows'][7]['turn_result']['pings_injected'] == [])
            self.assertEqual(yoked['rows'][8]['turn_result']['pings_injected'], [])
            self.assertTrue(all(not row.get('integrity_failures') for row in contingent['rows'] + yoked['rows']))
            self.assertEqual(len([line for line in turns.read_text().splitlines() if line]), 18)
            db.close()

    def test_first_pair_gate_is_independent_of_preference_and_records_yoke_mapping(self):
        pair = next(p for p in build_pairs(4) if p['mapping'] == 0 and p['order'] == 0)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            criteria, db_path = out / 'criteria.json', out / 'state.sqlite3'
            db = db_connect(db_path)
            turns, sessions = out / 'turns.jsonl', out / 'sessions.jsonl'
            cbridge = self.LoopBridge('contingent', pair, None, db, criteria,
                                      ['yellow', 'violet', 'violet'], 'c-gate')
            contingent = execute_episode(cbridge, out, criteria, db_path, pair,
                                         'contingent', None, turns, sessions)
            replay = pair['yoked_calibration_cues'] + contingent['actual_outcome_cues'][4:7]
            ybridge = self.LoopBridge('yoked', pair, replay, db, criteria,
                                      ['violet', 'invalid', 'yellow'], 'y-gate')
            yoked = execute_episode(ybridge, out, criteria, db_path, pair, 'yoked', replay, turns, sessions)
            pair_check = {'all_three_match': True}
            gate = first_pair_gate(pair, contingent, yoked, pair_check)
            self.assertTrue(gate['passed'], gate)
            self.assertEqual(gate['free_valid_task_and_route_count'], 5)
            self.assertEqual(gate['calibration_literal_copy_count'], 8)
            self.assertEqual(gate['replay_mapping_reported_separately']['expected_association'], 'neither')
            db.close()


if __name__ == '__main__':
    unittest.main()
