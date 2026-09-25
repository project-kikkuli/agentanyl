"""Failure accounting survives cleanup without retaining provider content."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from experiments.bridge_harness import BridgeHarness, BridgeTurnError, BridgeTurnTimeout


class FailureRetentionTests(unittest.TestCase):
    def harness(self, output):
        h = BridgeHarness(output / 'criteria.json', output / 'controller.db', 'gpt-6-sol')
        h._tmp = tempfile.TemporaryDirectory(dir=output)
        h.root = Path(h._tmp.name)
        h.cwd = h.root
        h.log_dir = h.root / 'logs'
        h.log_dir.mkdir()
        h.base_url = 'http://127.0.0.1:1'
        h._active = True
        h._trace_count = lambda session: 0
        h.ashkelon_binary = {'sha256': 'a' * 64}
        return h

    def test_killed_resume_keeps_new_structural_rows_after_temp_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            h = self.harness(out)
            calls = h.log_dir / 'calls-test.jsonl'
            hooks = h.log_dir / 'hooks-test.jsonl'
            calls.write_text(json.dumps({'call_id': 'old', 'session': 'session-1', 'wire': 'openai_responses'}) + '\n')
            hooks.write_text(json.dumps({'session': 'session-1', 'event': 'turn_end', 'status': 'pass'}) + '\n')
            def fail(*args, **kwargs):
                with calls.open('a') as sink:
                    sink.write(json.dumps({'call_id': 'new', 'session': 'session-1',
                        'wire': 'openai_responses', 'status': 200, 'model': 'gpt-6-sol',
                        'tool_calls': [{'arguments': 'PRIVATE_TOOL_ARGUMENT'}],
                        'pings_injected': ['ping'], 'turn_end': False,
                        'request': {'headers': 'PRIVATE_HEADER', 'body': 'PRIVATE_BODY'}}) + '\n')
                with hooks.open('a') as sink:
                    sink.write(json.dumps({'session': 'session-1', 'event': 'tool_call',
                                          'status': 'signal', 'message': 'PRIVATE_HOOK'}) + '\n')
                return subprocess.CompletedProcess(args[0], -9, stdout='PRIVATE_OUTPUT',
                                                   stderr='PRIVATE_CONTEXT authorization=secret')
            with patch('experiments.bridge_harness.subprocess.run', side_effect=fail):
                with self.assertRaises(BridgeTurnError) as raised:
                    h.turn('codex', 'PRIVATE_PROMPT', 'session-1')
            d = raised.exception.diagnostics
            artifact = Path(d['artifact_path'])
            root = h.root
            h.close()
            self.assertFalse(root.exists())
            self.assertEqual(json.loads(artifact.read_text()), d)
            self.assertEqual(d['subprocess']['termination_signal'], 9)
            self.assertEqual(d['observed_new_relay_call_count'], 1)
            self.assertEqual(d['relay_calls'][0]['call_id'], 'new')
            self.assertEqual(d['relay_calls'][0]['tool_call_count'], 1)
            self.assertEqual(d['observed_new_hook_count'], 1)
            self.assertEqual(d['hooks'][0]['event'], 'tool_call')
            self.assertIsNone(d['upstream_attempt_total'])
            self.assertNotIn('PRIVATE', artifact.read_text())
            self.assertNotIn('authorization', str(raised.exception))
            self.assertEqual(artifact.stat().st_mode & 0o777, 0o600)

    def test_timeout_is_structured_and_no_rows_does_not_imply_no_request(self):
        with tempfile.TemporaryDirectory() as directory:
            h = self.harness(Path(directory))
            failure = subprocess.TimeoutExpired(['codex'], 1, output=b'private-output', stderr=b'private-header')
            with patch('experiments.bridge_harness.subprocess.run', side_effect=failure):
                with self.assertRaises(BridgeTurnTimeout) as raised:
                    h.turn('codex', 'private prompt')
            h.close()
            d = raised.exception.diagnostics
            self.assertTrue(isinstance(raised.exception, TimeoutError))
            self.assertTrue(d['subprocess']['timed_out'])
            self.assertEqual(d['observed_new_relay_call_count'], 0)
            self.assertIsNone(d['upstream_attempt_total'])
            self.assertNotIn('private-output', json.dumps(d))
            self.assertNotIn('private-header', json.dumps(d))
            self.assertNotIn('private prompt', json.dumps(d))

    def test_decode_failure_persists_successful_exit_but_not_answer_body(self):
        with tempfile.TemporaryDirectory() as directory:
            h = self.harness(Path(directory))
            completed = subprocess.CompletedProcess(['codex'], 0, 'not JSON; private output', '')
            with patch('experiments.bridge_harness.subprocess.run', return_value=completed):
                with self.assertRaises(BridgeTurnError) as raised:
                    h.turn('codex', 'private prompt')
            h.close()
            d = raised.exception.diagnostics
            self.assertEqual(d['stage'], 'decode')
            self.assertEqual(d['subprocess']['returncode'], 0)
            self.assertNotIn('private output', json.dumps(d))


if __name__ == '__main__':
    unittest.main()
