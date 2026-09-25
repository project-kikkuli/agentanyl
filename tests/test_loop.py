import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
import unittest
from agentanyl.loop import db_connect, decision, evaluate, load_config, run, update


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

if __name__ == '__main__':
    unittest.main()
