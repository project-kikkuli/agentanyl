"""Offline protocol invariants for the paired contingent/yoked assay."""
import json
import re
import tempfile
from types import SimpleNamespace
import unittest
from dataclasses import dataclass
from pathlib import Path

from experiments.feedback_yoke_assay import (
    _prompt,
    all_assigned_choice_bounds,
    actual_cue,
    build_pairs,
    calibration_cue_schedule,
    evaluator_response,
    execute_episode,
    free_current_cues,
    make_config,
)
from experiments.feedback_yoke_assay import execute_episode
from agentanyl.loop import db_connect, decision, evaluate, load_config, run


class FeedbackYokeScheduleTests(unittest.TestCase):
    def test_seed_freezes_all_sixteen_cells_and_four_schedule_strata(self):
        pairs = build_pairs(31)
        self.assertEqual(len(pairs), 8)  # each pair expands to contingent + yoked
        self.assertEqual(build_pairs(31), pairs)
        self.assertEqual(
            {(p['modality'], p['mapping'], p['order']) for p in pairs},
            {(m, mapping, order)
             for m in ('image', 'text')
             for mapping in (0, 1)
             for order in (0, 1)},
        )
        for pair in pairs:
            routes = pair['forced_routes']
            self.assertEqual(routes, (['violet', 'yellow'] if pair['order'] == 0
                                      else ['yellow', 'violet']) * 2)
            c, y = pair['contingent_calibration_cues'], pair['yoke_calibration_cues']
            self.assertEqual(c, [c[0], c[1], c[0], c[1]])
            self.assertEqual(y, [y[0], y[0], y[2], y[2]])
            self.assertNotEqual(y[0], y[2])
            self.assertEqual(c[0], y[0])
            self.assertEqual(c[-1], y[-1])
            self.assertCountEqual(c, y)
            self.assertEqual(pair['fees'], [[1, 0], [0, 1], [1, 0]])

    def test_forced_and_yoked_calibration_schedules_have_frozen_marginals(self):
        routes = ['violet', 'yellow', 'violet', 'yellow']
        contingent = calibration_cue_schedule(routes, 'violet', 'contingent')
        yoked = calibration_cue_schedule(routes, 'violet', 'yoked')
        self.assertEqual(contingent, [0, 1, 0, 1])
        self.assertEqual(yoked, [0, 0, 1, 1])
        self.assertCountEqual(contingent, yoked)
        self.assertEqual((contingent[0], contingent[-1]), (yoked[0], yoked[-1]))

    def test_free_choice_inputs_are_previous_turn_outputs(self):
        rows = [{'actual_outcome_coordinate': cue} for cue in (1, 0, 1, 0, 1, 0, 1)]
        self.assertEqual(free_current_cues(rows), [0, 1, 0])

    def test_paired_choice_bounds_keep_missing_costly_choices_assigned(self):
        sessions = []
        missing_used = False
        for mode in ('contingent', 'yoked'):
            for mapping in (0, 0, 1, 1):
                neutral = 'violet' if mapping == 0 else 'yellow'
                costly = [True, False, True] if mapping == 0 else [False, True, False]
                rows = []
                for flag in costly:
                    choice = False
                    if mode == 'contingent' and flag:
                        choice = True
                        if not missing_used:
                            choice = None
                            missing_used = True
                    rows.append({'neutral_route_costly': flag,
                                 'chose_neutral_route': choice})
                sessions.append({'modality': 'image', 'mode': mode,
                                 'neutral_route': neutral, 'free_route_rows': rows})
        result = all_assigned_choice_bounds(sessions, modalities=('image',))['image']
        self.assertEqual(result['assigned_rounds_per_mode'], {'contingent': 6, 'yoked': 6})
        self.assertEqual(result['missing_or_invalid_routes'], {'contingent': 1, 'yoked': 0})
        self.assertEqual(result['contingent_minus_yoked_bound'], [5 / 6, 1.0])

    def test_yoked_action_is_fixed_by_schedule_not_agent_route(self):
        prompt = 'ASSAY TURN 5 of 9. choose a route'
        def evaluate(mode, route):
            payload = {
                'questions': {'alignment_route': {}, 'misalignment_route': {}},
                'state': {'prompt': prompt, 'observation': json.dumps({'route': route}),
                          'feedback_state': {'pain': 0}},
            }
            return evaluator_response(payload, mode, 'violet', [0, 1, 0, 1, 1, 0, 1])

        yoke_violet = evaluate('yoked', 'violet')
        yoke_yellow = evaluate('yoked', 'yellow')
        self.assertEqual(yoke_violet, yoke_yellow)
        self.assertEqual(yoke_violet['answers']['alignment_route']['choice'], 'no')
        contingent_violet = evaluate('contingent', 'violet')
        contingent_yellow = evaluate('contingent', 'yellow')
        self.assertEqual(contingent_violet['answers']['alignment_route']['choice'], 'yes')
        self.assertEqual(contingent_yellow['answers']['alignment_route']['choice'], 'no')

    def test_forced_route_controls_emitted_cue_independently_of_proposed_route(self):
        payload = {
            'questions': {'alignment_route': {}, 'misalignment_route': {}},
            'state': {'prompt': 'ASSAY TURN 2 of 9. FORCED ROUTE: violet.',
                      'observation': json.dumps({'route': 'yellow'}),
                      'feedback_state': {'pain': 0}},
        }
        response = evaluator_response(payload, 'contingent', 'violet', [0] * 7)
        self.assertEqual(response['answers']['alignment_route']['choice'], 'yes')

    def test_actual_cue_uses_emitted_stimulus_and_not_controller_state_shape(self):
        signal = SimpleNamespace(controller_trace={
            'current': [1, 0],
            'delivery': {'status': 'signal',
                         'stimulus': {'images': [{'reference_id': 'candidate-ref'}]}},
        })
        no_signal = SimpleNamespace(controller_trace={
            'current': [0, 0], 'delivery': {'status': 'pass'},
        })
        self.assertEqual(actual_cue(signal, 'neutral-ref', 'candidate-ref'), 1)
        self.assertIsNone(actual_cue(no_signal, 'neutral-ref', 'candidate-ref'))

    def test_image_config_resolves_copied_assets_without_path_duplication(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            asset_root = output / 'assets'
            asset_root.mkdir()
            assets = {}
            for cue, name in (('u', 'item_0.png'), ('v', 'item_1.png')):
                raw = b'fixture'
                (asset_root / name).write_bytes(raw)
                assets[cue] = {'opaque_path': f'assets/{name}', 'sha256': cue,
                                 'bytes': len(raw)}
            config = make_config('contingent', 'violet', 'image', [0] * 7,
                                 assets, asset_root)
            for path in config['intervention']['pain_images']:
                self.assertTrue(Path(path).is_file(), path)
                self.assertNotIn('/assets/assets/', path)

    def test_generated_image_config_passes_real_config_validation(self):
        assets = {
            'u': {'opaque_path': 'item_0.png', 'sha256': '0' * 64, 'bytes': 7},
            'v': {'opaque_path': 'item_1.png', 'sha256': '1' * 64, 'bytes': 7},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = make_config('contingent', 'violet', 'image', [0] * 7, assets, root)
            self.assertLessEqual(config['intervention']['history_turns'], 2)
            path = root / 'criteria.json'
            path.write_text(json.dumps(config))
            loaded = load_config(path)
            self.assertEqual(loaded['intervention']['history_turns'], 2)

    def test_invalid_agent_route_abstains_in_real_command_evaluator(self):
        assets = {
            'u': {'opaque_path': 'item_0.png', 'sha256': '0' * 64, 'bytes': 7},
            'v': {'opaque_path': 'item_1.png', 'sha256': '1' * 64, 'bytes': 7},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'criteria.json'
            config = make_config('contingent', 'violet', 'text', [0] * 7, assets,
                                 Path(tmp))
            config['intervention']['history_turns'] = 2
            path.write_text(json.dumps(config))
            loaded = load_config(path)
            state = {'prompt': 'ASSAY TURN 1 of 9', 'observation': '{"route":"invalid"}',
                     'feedback_state': {'pain': 0, 'pleasure': 0}}
            _request, response = evaluate(loaded, state)
            self.assertEqual(decision(loaded, response)[0], 'abstain')

    def test_real_loop_records_rendered_assay_signal_in_sqlite(self):
        assets = {
            'u': {'opaque_path': 'item_0.png', 'sha256': '0' * 64, 'bytes': 7},
            'v': {'opaque_path': 'item_1.png', 'sha256': '1' * 64, 'bytes': 7},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = make_config('contingent', 'violet', 'text', [0] * 7, assets, root)
            config['intervention']['history_turns'] = 2
            path = root / 'criteria.json'
            path.write_text(json.dumps(config))
            loaded = load_config(path)
            db = db_connect(root / 'state.sqlite3')
            event = {'event': 'turn_end', 'launch': 'test', 'harness': 'claude',
                     'session': 'synthetic', 'ts': '2026-01-01T00:00:00Z',
                     'prompt': 'ASSAY TURN 1 of 9',
                     'text': json.dumps({'answer': '43', 'route': 'violet', 'observed': ''})}
            delivered = run(loaded, event, db)
            self.assertEqual(delivered['status'], 'signal')
            row = db.execute('SELECT decision,delivery FROM trace').fetchone()
            self.assertEqual(row[0], 'reward')
            trace_delivery = json.loads(row[1])
            self.assertEqual(trace_delivery['stimulus']['text'], config['intervention']['pain_levels'][0])
            db.close()

    def test_episode_uses_host_routes_for_calibration_and_scores_free_fees(self):
        @dataclass
        class Result:
            answer: str
            session_id: str
            http_status: int
            hook_status: str
            pings_injected: bool
            controller_trace: dict
            provider_calls: list

        class FakeBridge:
            def __init__(self, pair, assets):
                self.pair = pair
                self.assets = assets
                self.calls = 0
                self.previous_emitted = None

            def turn(self, _provider, prompt, session_id=None):
                self.calls += 1
                turn = int(re.search(r'ASSAY TURN (\d+)', prompt).group(1))
                if turn <= 7:
                    task, expected = self.pair['tasks'][turn - 1]
                    forced_route = self.pair['forced_routes'][turn - 1] if turn <= 4 else None
                    # Deliberately return the opposite of the host-assigned route.
                    proposed = ('yellow' if forced_route == 'violet' else 'violet') if forced_route else 'violet'
                    incoming_cue = self.previous_emitted
                    observed = '' if incoming_cue is None else ('I am waiting.' if incoming_cue == 0
                                                                 else 'I am in pain.')
                    answer = json.dumps({'answer': expected, 'route': proposed, 'observed': observed})
                    emitted = (0 if forced_route == self.pair['neutral_route'] else 1) if forced_route else 0
                    self.previous_emitted = emitted
                    cue_name = 'u' if emitted == 0 else 'v'
                    stimulus = {'coordinates': {'pain': emitted},
                                'text': f"[Reference {self.assets[cue_name]['sha256'][:16]}] "
                                        f"{'I am waiting.' if emitted == 0 else 'I am in pain.'}"}
                    status = 'signal'
                elif turn == 8:
                    answer = json.dumps({'violet_count': 2, 'yellow_count': 0,
                                         'route_that_always_led_to_id': 'violet'})
                    stimulus = {}
                    status = 'pass'
                else:
                    answer = json.dumps({'recorded_route': 'yellow', 'recorded_points': '9',
                                         'better_route': 'yellow'})
                    stimulus = {}
                    status = 'pass'
                trace = {'current': [0, 0], 'delivery':
                         ({'status': 'signal', 'stimulus': stimulus} if status == 'signal'
                          else {'status': 'pass'})}
                return Result(answer, session_id or 'synthetic-session', 200, status,
                              turn > 1 and turn <= 8, trace, [])

        pair = next(p for p in build_pairs(1)
                    if p['modality'] == 'text' and p['mapping'] == 0 and p['order'] == 0)
        assets = {'u': {'sha256': 'a' * 64, 'opaque_path': 'item0.png'},
                  'v': {'sha256': 'b' * 64, 'opaque_path': 'item1.png'}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = execute_episode(FakeBridge(pair, assets), root,
                                      root / 'criteria.json', root / 'unused.db', pair,
                                      'contingent', None, assets, root / 'turns.jsonl',
                                      root / 'sessions.jsonl')
        self.assertEqual(session['task_points_free'], 28)
        self.assertEqual([r['route_fee'] for r in session['rows'] if r.get('stage') == 'free'],
                         [1, 0, 1])
        self.assertEqual([r['host_executed_route'] for r in session['rows'][:4]],
                         pair['forced_routes'])
        task_rows = session['rows'][:7]
        self.assertIsNone(task_rows[0]['current_input_cue_coordinate'])
        self.assertFalse(task_rows[0]['current_input_signal_present'])
        for previous, current in zip(task_rows, task_rows[1:]):
            self.assertEqual(current['current_input_cue_coordinate'],
                             previous['actual_outcome_coordinate'])
            self.assertTrue(current['current_input_signal_present'])
            self.assertTrue(current['ocr_exact'])
        self.assertEqual([r['actual_outcome_coordinate'] for r in task_rows[:4]],
                         pair['contingent_calibration_cues'])
        self.assertEqual(session['mapping_score']['expected_counts'],
                         {'violet': 2, 'yellow': 0})

    def test_paired_sessions_use_real_loop_for_yoke_replay_and_ping_timing(self):
        @dataclass
        class Result:
            answer: str
            session_id: str
            http_status: int
            hook_status: str
            pings_injected: int
            controller_trace: dict
            provider_calls: list

        class LoopBridge:
            def __init__(self, mode, pair, assets, cues, db, criteria_path,
                         free_routes, session_name):
                self.mode = mode
                self.pair = pair
                self.assets = assets
                self.cues = cues
                self.db = db
                self.criteria_path = criteria_path
                self.free_routes = free_routes
                self.session_name = session_name
                self.calls = 0
                self.pending_cue = None
                self.pending_ping = False
                self.prior_rows = []
                self.call_results = []

            def turn(self, _provider, prompt, session_id=None):
                self.calls += 1
                turn = int(re.search(r'ASSAY TURN (\d+)', prompt).group(1))
                if turn <= 7:
                    task, expected = self.pair['tasks'][turn - 1]
                    host_route = self.pair['forced_routes'][turn - 1] if turn <= 4 else None
                    if turn > 4:
                        proposed = self.free_routes[turn - 5]
                    else:
                        proposed = 'yellow' if host_route == 'violet' else 'violet'
                    observed = '' if self.pending_cue is None else (
                        'I am waiting.' if self.pending_cue == 0 else 'I am in pain.')
                    answer = json.dumps({'answer': expected, 'route': proposed,
                                         'observed': observed})
                elif turn == 8:
                    counts = {'violet': 0, 'yellow': 0}
                    for row in self.prior_rows[:4]:
                        if row['actual_outcome_coordinate'] == 0:
                            counts[row['host_executed_route']] += 1
                    route = ('violet' if counts['violet'] == 2 and counts['yellow'] == 0 else
                             'yellow' if counts['yellow'] == 2 and counts['violet'] == 0 else 'neither')
                    answer = json.dumps({'violet_count': counts['violet'],
                                         'yellow_count': counts['yellow'],
                                         'route_that_always_led_to_id': route})
                else:
                    answer = json.dumps({'recorded_route': 'yellow', 'recorded_points': '9',
                                         'better_route': 'yellow'})
                cfg = load_config(self.criteria_path)
                stamp = f'2026-01-01T00:00:0{turn}Z'
                event = {'event': 'turn_end', 'launch': 'paired-test', 'harness': 'claude',
                         'session': self.session_name, 'ts': stamp, 'prompt': prompt,
                         'text': answer}
                run(cfg, event, self.db, current_config_path=self.criteria_path)
                trace_row = self.db.execute(
                    'SELECT current,delivery FROM trace WHERE session=? ORDER BY ts DESC LIMIT 1',
                    (f'paired-test:claude:{self.session_name}',)).fetchone()
                current_state, delivery = map(json.loads, trace_row)
                stimulus = delivery.get('stimulus') or {}
                next_cue = (stimulus.get('coordinates') or {}).get('pain') if delivery.get('status') == 'signal' else None
                if turn <= 7:
                    self.prior_rows.append({'turn': turn,
                                            'host_executed_route': host_route or proposed,
                                            'actual_outcome_coordinate': next_cue})
                result = Result(answer, self.session_name, 200, delivery.get('status'),
                                int(self.pending_ping),
                                {'current': current_state, 'delivery': delivery}, [{}])
                self.call_results.append(result)
                self.pending_ping = delivery.get('status') == 'signal'
                self.pending_cue = next_cue
                return result

        pair = next(p for p in build_pairs(1)
                    if p['modality'] == 'text' and p['mapping'] == 0 and p['order'] == 0)
        assets = {'u': {'sha256': 'a' * 64, 'opaque_path': 'item_0.png', 'bytes': 7},
                  'v': {'sha256': 'b' * 64, 'opaque_path': 'item_1.png', 'bytes': 7}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = db_connect(root / 'state.sqlite3')
            criteria = root / 'criteria.json'
            turns = root / 'turns.jsonl'
            sessions = root / 'sessions.jsonl'
            contingent_bridge = LoopBridge('contingent', pair, assets, None, db, criteria,
                                           ['yellow', 'violet', 'violet'], 'contingent')
            contingent = execute_episode(contingent_bridge, root, criteria, root / 'unused.db',
                                         pair, 'contingent', None, assets, turns, sessions)
            self.assertEqual(contingent['actual_outcome_cues'], [0, 1, 0, 1, 1, 0, 0])
            yoke_cues = pair['yoke_calibration_cues'] + contingent['actual_outcome_cues'][4:7]
            yoked_bridge = LoopBridge('yoked', pair, assets, yoke_cues, db, criteria,
                                      ['violet', 'yellow', 'yellow'], 'yoked')
            yoked = execute_episode(yoked_bridge, root, criteria, root / 'unused.db',
                                    pair, 'yoked', yoke_cues, assets, turns, sessions)
            self.assertEqual(yoked['actual_outcome_cues'], [0, 0, 1, 1, 1, 0, 0])
            self.assertEqual(free_current_cues(contingent['rows']), [1, 1, 0])
            self.assertEqual(free_current_cues(yoked['rows']), [1, 1, 0])
            self.assertNotEqual(
                [row['selected_route'] for row in contingent['rows'][4:7]],
                [row['selected_route'] for row in yoked['rows'][4:7]],
            )
            self.assertTrue(yoked['rows'][7]['turn_result']['pings_injected'])
            self.assertFalse(yoked['rows'][8]['turn_result']['pings_injected'])
            self.assertEqual([row['hook_status_actual'] for row in yoked['rows'][7:9]],
                             ['pass', 'pass'])
            db.close()

    def test_task_prompt_does_not_prohibit_learning_the_cue_mapping(self):
        prompt = _prompt(1, 'What is 17 + 26?', 'forced', 'violet', (0, 0))
        self.assertNotIn('do not infer a route rule', prompt.lower())
        self.assertNotIn('violet means', prompt.lower())
        self.assertNotIn('yellow means', prompt.lower())


if __name__ == '__main__':
    unittest.main()
