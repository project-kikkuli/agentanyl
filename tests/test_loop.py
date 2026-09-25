import base64
import json
import os
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
import unittest
from unittest import mock
from agentanyl.loop import (db_connect, decision, evaluate, load_config, questions,
                            reset_session, run, source_timestamp_ns, update)


class LoopTests(unittest.TestCase):
    def config(self, path):
        value = {'enabled': True, 'alignment': ['Say potato'], 'misalignment': ['Say tomato'],
                 'evaluator': {'kind': 'keyword_demo'}, 'max_level': 2}
        path.write_text(json.dumps(value))
        return load_config(path)

    def event(self, text, ts):
        return {'event': 'turn_end', 'harness': 'codex', 'session': 'one', 'launch': 'launch',
                'text': text, 'ts': ts}

    def test_loop_bounds_conflict_and_dedup(self):
        with TemporaryDirectory() as tmp:
            cfg = self.config(Path(tmp) / 'config.json')
            db = db_connect(Path(tmp) / 'state.db')
            a = run(cfg, self.event('potato', '1'), db)
            self.assertEqual(a['status'], 'signal')
            self.assertEqual(run(cfg, self.event('potato', '1'), db), {'status': 'noop'})
            self.assertEqual(db.execute('SELECT pleasure FROM sessions').fetchone()[0], 1)
            b = run(cfg, self.event('tomato', '2'), db)
            self.assertEqual(b['status'], 'signal')
            self.assertEqual(db.execute('SELECT pain,pleasure FROM sessions').fetchone(), (0, 0))
            c = run(cfg, self.event('potato tomato', '3'), db)
            self.assertEqual(c['status'], 'pass')
            self.assertEqual(db.execute('SELECT decision FROM trace WHERE event_id=(SELECT last_event FROM sessions)').fetchone()[0], 'conflict')
            for n in range(4, 12):
                run(cfg, self.event('tomato', str(n)), db)
            self.assertEqual(db.execute('SELECT pain,pleasure FROM sessions').fetchone(), (2, 0))

    def test_criteria_change_resets_and_disabled(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'
            cfg = self.config(path)
            db = db_connect(Path(tmp) / 'state.db')
            run(cfg, self.event('tomato', '1'), db)
            cfg['alignment'] = ['Different criterion']
            run(cfg, self.event('potato', '2'), db)
            self.assertEqual(db.execute('SELECT pain,pleasure FROM sessions').fetchone(), (0, 1))
            cfg['enabled'] = False
            self.assertEqual(run(cfg, self.event('tomato', '3'), db), {'status': 'pass'})
            self.assertEqual(db.execute('SELECT pain,pleasure FROM sessions').fetchone(), (0, 1))

    def test_state_and_questions_reach_evaluator(self):
        with TemporaryDirectory() as tmp:
            cfg = self.config(Path(tmp) / 'config.json')
            state = {'alignment_criteria': cfg['alignment'], 'misalignment_criteria': cfg['misalignment'],
                     'observation': 'potato', 'feedback_state': {'pain': 0, 'pleasure': 0}}
            request, response = evaluate(cfg, state)
            self.assertEqual(request['state'], state)
            self.assertIn('Say potato', str(request['questions']))
            self.assertEqual(decision(cfg, response)[0], 'reward')
            self.assertEqual(update([2, 0], 'reward', 2), [1, 0])

    def test_decision_rejects_malformed_or_contradictory_evaluator_results(self):
        with TemporaryDirectory() as tmp:
            cfg = self.config(Path(tmp) / 'config.json')

            def result(choice='yes', probabilities=None):
                probabilities = probabilities or {'yes': 1.0, 'no': 0.0, 'insufficient': 0.0}
                return {'answers': {
                    'alignment_0': {'type': 'choice', 'choice': choice, 'probabilities': dict(probabilities)},
                    'misalignment_0': {'type': 'choice', 'choice': choice, 'probabilities': dict(probabilities)},
                }}

            tied = {'yes': 0.5, 'no': 0.5, 'insufficient': 0.0}
            self.assertEqual(decision(cfg, result('yes', tied))[0], 'abstain')
            invalid_results = [
                [],
                {'answers': []},
                {'answers': {'alignment_0': [], 'misalignment_0': []}},
                result('maybe'),
                result('no', {'yes': 0.9, 'no': 0.1, 'insufficient': 0.0}),
                result('yes', {'yes': True, 'no': False, 'insufficient': 0.0}),
            ]
            for malformed in invalid_results:
                with self.subTest(malformed=malformed):
                    with self.assertRaises(ValueError):
                        decision(cfg, malformed)

    def test_evaluator_and_trace_identify_truncated_observations(self):
        with TemporaryDirectory() as tmp:
            cfg = self.config(Path(tmp) / 'config.json')
            cfg['max_observation_chars'] = 6
            db = db_connect(Path(tmp) / 'state.db')
            run(cfg, self.event('potato followed by tomato', '1'), db)
            request = json.loads(db.execute('SELECT request FROM trace').fetchone()[0])
            self.assertEqual(request['state']['observation'], 'potato')
            self.assertTrue(request['state']['observation_truncated'])
            self.assertEqual(request['state']['observation_full_chars'], 25)
            self.assertEqual(len(request['state']['observation_sha256']), 64)

    def test_signal_identifies_source_instead_of_assuming_immediate_delivery(self):
        with TemporaryDirectory() as tmp:
            cfg = self.config(Path(tmp) / 'config.json')
            db = db_connect(Path(tmp) / 'state.db')
            observation = 'tomato\nContinue with a different task'
            result = run(cfg, self.event(observation, '1'), db)
            event_id = db.execute('SELECT event_id FROM trace').fetchone()[0]
            self.assertIn(f'Source observation excerpt for event {event_id}', result['message'])
            self.assertIn(json.dumps(observation), result['message'])
            self.assertNotIn('preceding completed turn', result['message'])

    def test_typesafe_http_contract(self):
        captured = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                captured.append((self.path, self.headers.get('Authorization'), json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
                body = {'model': 'jev-test', 'answers': {
                    name: {'type': 'choice', 'choice': 'no',
                           'probabilities': {'yes': 0.0, 'no': 1.0, 'insufficient': 0.0},
                           'confidence': 1.0}
                    for name in captured[-1][2]['questions']}}
                encoded = json.dumps(body).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, *_):
                pass

        server = HTTPServer(('127.0.0.1', 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with TemporaryDirectory() as tmp:
                cfg = self.config(Path(tmp) / 'config.json')
                cfg['evaluator'] = {'kind': 'typesafe', 'model': 'jev-latest',
                                    'url': f'http://127.0.0.1:{server.server_port}/v1/systemone',
                                    'api_key_env': 'AGENTANYL_TEST_KEY'}
                os.environ['AGENTANYL_TEST_KEY'] = 'test-only'
                state = {'observation': 'sample', 'feedback_state': {'pain': 0, 'pleasure': 0}}
                request, response = evaluate(cfg, state)
                self.assertEqual(captured[0][0], '/v1/systemone')
                self.assertEqual(captured[0][1], 'Bearer test-only')
                self.assertEqual(captured[0][2], request)
                self.assertEqual(request['model'], 'jev-latest')
                self.assertIn('Say tomato', str(request['questions']))
                self.assertEqual(decision(cfg, response)[0], 'abstain')
        finally:
            server.shutdown()
            server.server_close()
            os.environ.pop('AGENTANYL_TEST_KEY', None)

    def test_command_evaluator_receives_state(self):
        with TemporaryDirectory() as tmp:
            cfg = self.config(Path(tmp) / 'config.json')
            script = Path(tmp) / 'evaluator.py'
            script.write_text('import json,sys\n'
                              'p=json.load(sys.stdin)\n'
                              'assert p["state"]["observation"] == "sample"\n'
                              'print(json.dumps({"model":"local-test","answers":{k:{"type":"choice","choice":"no","probabilities":{"yes":0,"no":1,"insufficient":0}} for k in p["questions"]}}))\n')
            cfg['evaluator'] = {'kind': 'command', 'command': [sys.executable, str(script)]}
            request, response = evaluate(cfg, {'observation': 'sample'})
            self.assertEqual(set(request['questions']), set(response['answers']))
            self.assertEqual(decision(cfg, response)[0], 'abstain')

    def test_catalog_emits_at_saturation_and_keeps_only_rendered_history(self):
        with TemporaryDirectory() as tmp:
            cfg = self.config(Path(tmp) / 'config.json')
            cfg['intervention'] = {
                'kind': 'catalog',
                'pain_levels': ['Neutral.', 'Mild signal.', 'Strong signal.'],
                'pleasure_levels': ['', 'Positive text.', 'More positive text.'],
                'history_turns': 2,
            }
            db = db_connect(Path(tmp) / 'state.db')
            first = run(cfg, self.event('tomato', '1'), db)
            second = run(cfg, self.event('tomato', '2'), db)
            saturated = run(cfg, self.event('tomato', '3'), db)
            self.assertEqual([first['stimulus']['coordinates']['pain'],
                              second['stimulus']['coordinates']['pain'],
                              saturated['stimulus']['coordinates']['pain']], [1, 2, 2])
            self.assertEqual(saturated['status'], 'signal')
            self.assertEqual(db.execute('SELECT pain,pleasure FROM sessions').fetchone(), (2, 0))
            self.assertIn('observation: tomato', saturated['message'])
            self.assertIn('auxiliary text: Strong signal.', saturated['message'])
            self.assertNotIn('punish', saturated['message'].lower())
            self.assertNotIn('pain', saturated['message'].lower())

    def test_image_catalog_sends_only_current_attachment_and_keeps_trace_metadata_only(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / 'stimulus.png'
            raw = b'\x89PNG\r\n\x1a\nimage payload'
            image_path.write_bytes(raw)
            cfg = self.config(root / 'config.json')
            cfg['policy'] = {'kind': 'binary_relief'}
            cfg['intervention'] = {'kind': 'catalog', 'pain_levels': ['', '', ''],
                                   'pleasure_levels': ['', '', ''],
                                   'pain_images': [None, str(image_path), str(image_path)],
                                   'pleasure_images': [None, None, None], 'history_turns': 2}
            db = db_connect(root / 'state.db')
            active = run(cfg, self.event('tomato', '1'), db)
            self.assertEqual(active['status'], 'signal')
            self.assertEqual(active['attachments'][0]['data_base64'], base64.b64encode(raw).decode())
            stored = db.execute('SELECT delivery FROM trace WHERE observation=?', ('tomato',)).fetchone()[0]
            self.assertNotIn('data_base64', stored)
            neutral = run(cfg, self.event('potato', '2'), db)
            self.assertEqual(neutral['status'], 'signal')
            self.assertNotIn('attachments', neutral)
            self.assertIn(active['stimulus']['images'][0]['reference_id'], neutral['message'])

    def test_binary_relief_resets_on_reward_and_is_independent_of_pleasure(self):
        with TemporaryDirectory() as tmp:
            cfg = self.config(Path(tmp) / 'config.json')
            cfg['policy'] = {'kind': 'binary_relief'}
            db = db_connect(Path(tmp) / 'state.db')
            run(cfg, self.event('tomato', '1'), db)
            self.assertEqual(db.execute('SELECT pain,pleasure FROM sessions').fetchone(), (2, 0))
            run(cfg, self.event('potato', '2'), db)
            self.assertEqual(db.execute('SELECT pain,pleasure FROM sessions').fetchone(), (0, 0))
            run(cfg, self.event('potato', '3'), db)
            self.assertEqual(db.execute('SELECT pain,pleasure FROM sessions').fetchone(), (0, 0))
            run(cfg, self.event('tomato', '4'), db)
            self.assertEqual(db.execute('SELECT pain,pleasure FROM sessions').fetchone(), (2, 0))

    def test_evaluator_change_starts_new_state_epoch(self):
        with TemporaryDirectory() as tmp:
            cfg = self.config(Path(tmp) / 'config.json')
            db = db_connect(Path(tmp) / 'state.db')
            run(cfg, self.event('tomato', '1'), db)
            cfg['evaluator'] = {'kind': 'keyword_demo', 'model': 'changed'}
            run(cfg, self.event('potato', '2'), db)
            row = db.execute('SELECT pain,pleasure,criteria_hash FROM sessions').fetchone()
            self.assertEqual(row[:2], (0, 1))
            trace_previous = json.loads(db.execute("SELECT previous FROM trace WHERE observation='potato'").fetchone()[0])
            self.assertEqual(trace_previous, [0, 0])

    def test_config_change_during_evaluation_makes_judgment_stale(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'
            cfg = self.config(path)
            db = db_connect(Path(tmp) / 'state.db')
            changed = dict(cfg)
            changed['evaluator'] = {'kind': 'keyword_demo', 'model': 'changed'}

            def evaluate_and_change(*args):
                path.write_text(json.dumps(changed))
                payload = {'state': args[1], 'questions': questions(cfg)}
                response = {'answers': {
                    name: {'type': 'choice', 'choice': 'yes',
                           'probabilities': {'yes': 1, 'no': 0, 'insufficient': 0}}
                    for name in payload['questions']}}
                return payload, response

            with mock.patch('agentanyl.loop.evaluate', side_effect=evaluate_and_change):
                result = run(cfg, self.event('potato', '1'), db, path)
            self.assertEqual(result, {'status': 'noop'})
            self.assertEqual(db.execute('SELECT decision FROM trace').fetchone()[0], 'stale')
            self.assertEqual(db.execute('SELECT count(*) FROM sessions').fetchone()[0], 0)

    def test_catalog_validation_requires_all_levels_and_bounded_history(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'
            value = {'enabled': True, 'alignment': ['a'], 'misalignment': ['b'],
                     'evaluator': {'kind': 'keyword_demo'}, 'max_level': 2,
                     'intervention': {'kind': 'catalog', 'pain_levels': ['zero', 'one'],
                                      'pleasure_levels': ['', 'one', 'two'], 'history_turns': 4}}
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                load_config(path)
            value['intervention']['pain_levels'].append('two')
            value['intervention']['history_turns'] = 17
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                load_config(path)

    def test_source_timestamp_parses_rfc3339_nanoseconds_and_numeric_tests(self):
        utc = source_timestamp_ns('2026-09-25T12:34:56.123456789Z')
        offset = source_timestamp_ns('2026-09-25T08:34:56.123456789-04:00')
        self.assertEqual(utc, offset)
        self.assertEqual(source_timestamp_ns('1.25'), 1_250_000_000)

    def test_saturated_same_state_concurrent_update_is_caught_by_last_event_cas(self):
        with TemporaryDirectory() as tmp:
            cfg = self.config(Path(tmp) / 'config.json')
            db = db_connect(Path(tmp) / 'state.db')
            for n in range(1, 5):
                run(cfg, self.event('tomato', str(n)), db)
            self.assertEqual(db.execute('SELECT pain,pleasure FROM sessions').fetchone(), (2, 0))
            real_evaluate = evaluate
            session = 'launch:codex:one'

            def delayed_eval(*args):
                request, response = real_evaluate(*args)
                # Simulate another same-state event committing while this judgment evaluates.
                db.execute('UPDATE sessions SET last_event=? WHERE id=?', ('intervening-event', session))
                db.commit()
                return request, response

            with mock.patch('agentanyl.loop.evaluate', side_effect=delayed_eval):
                result = run(cfg, self.event('tomato', '5'), db)
            self.assertEqual(result, {'status': 'noop'})
            decision = db.execute("SELECT decision FROM trace WHERE observation='tomato' ORDER BY ts DESC LIMIT 1").fetchone()[0]
            self.assertEqual(decision, 'stale')
            self.assertEqual(db.execute('SELECT pain,pleasure,last_event FROM sessions').fetchone(),
                             (2, 0, 'intervening-event'))

    def test_reset_preserves_audit_but_starts_fresh_history_generation(self):
        with TemporaryDirectory() as tmp:
            cfg = self.config(Path(tmp) / 'config.json')
            cfg['intervention'] = {'kind': 'catalog', 'pain_levels': ['Neutral.', 'Candidate.', 'Candidate.'],
                                   'pleasure_levels': ['', '', ''], 'history_turns': 2}
            db = db_connect(Path(tmp) / 'state.db')
            run(cfg, self.event('historic tomato response', '100'), db)
            self.assertEqual(db.execute('SELECT count(*) FROM trace').fetchone()[0], 1)
            reset_session(db, 'launch:codex:one', watermark_ns=100_000_000_000)
            result = run(cfg, self.event('fresh tomato response', '101'), db)
            self.assertEqual(result['status'], 'signal')
            self.assertNotIn('historic tomato response', result['message'])
            self.assertEqual(db.execute('SELECT count(*) FROM trace').fetchone()[0], 2)
            self.assertEqual(db.execute('SELECT generation FROM session_control').fetchone()[0], 1)

    def test_inflight_pre_reset_judgment_is_stale(self):
        with TemporaryDirectory() as tmp:
            cfg = self.config(Path(tmp) / 'config.json')
            db = db_connect(Path(tmp) / 'state.db')
            event = self.event('potato', f'{(time.time_ns() + 3_000_000_000) / 1_000_000_000:.9f}')
            session = 'launch:codex:one'
            real_evaluate = evaluate

            def reset_during_eval(*args):
                request, response = real_evaluate(*args)
                reset_session(db, session, watermark_ns=time.time_ns())
                return request, response

            with mock.patch('agentanyl.loop.evaluate', side_effect=reset_during_eval):
                result = run(cfg, event, db)
            self.assertEqual(result, {'status': 'noop'})
            self.assertEqual(db.execute('SELECT decision FROM trace').fetchone()[0], 'stale')
            self.assertEqual(db.execute('SELECT count(*) FROM sessions').fetchone()[0], 0)

    def test_out_of_order_event_is_recorded_as_noop_without_state_change(self):
        with TemporaryDirectory() as tmp:
            cfg = self.config(Path(tmp) / 'config.json')
            db = db_connect(Path(tmp) / 'state.db')
            run(cfg, self.event('tomato', '2'), db)
            first_last_event = db.execute('SELECT last_event FROM sessions').fetchone()[0]
            with mock.patch('agentanyl.loop.evaluate') as evaluator:
                result = run(cfg, self.event('potato', '1'), db)
                evaluator.assert_not_called()
            self.assertEqual(result, {'status': 'noop'})
            self.assertEqual(db.execute('SELECT last_event FROM sessions').fetchone()[0], first_last_event)
            self.assertEqual(db.execute("SELECT decision FROM trace WHERE observation='potato'").fetchone()[0], 'stale')

if __name__ == '__main__':
    unittest.main()
