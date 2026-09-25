"""Executable selection changes transport only; no CLI/provider process is launched."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.bridge_harness import BridgeHarness, codex_executable_metadata
from experiments.addressed_feedback_assay import make_manifest, build_pairs


class CodexExecutableTests(unittest.TestCase):
    def test_default_command_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = BridgeHarness(Path(tmp) / 'criteria', Path(tmp) / 'db', 'gpt-6-sol')
            h.base_url = 'http://127.0.0.1:1'
            with patch('experiments.bridge_harness.codex_executable_metadata', return_value={'path': '/example/codex', 'sha256': 'x'}) as metadata:
                args = h._codex_args('prompt', Path(tmp) / 'answer', 'session-1')
            self.assertEqual(args[:2], ['codex', 'exec'])
            self.assertIn('resume', args)
            self.assertEqual(args[-2:], ['session-1', 'prompt'])
            metadata.assert_called_once_with('codex')

    def test_override_records_exact_path_and_bytes_in_args_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = root / 'codex-native'
            native.write_bytes(b'fixture executable, never run')
            native.chmod(0o700)
            expected = {'requested': str(native), 'path': str(native.resolve()),
                        'sha256': hashlib.sha256(native.read_bytes()).hexdigest()}
            self.assertEqual(codex_executable_metadata(native), expected)
            h = BridgeHarness(root / 'criteria', root / 'db', 'gpt-6-sol', codex_executable=native)
            h.base_url = 'http://127.0.0.1:1'
            args = h._codex_args('prompt', root / 'answer', None)
            self.assertEqual(args[0], str(native))
            self.assertEqual(h.codex_binary, expected)
            manifest = make_manifest(1, 'codex', 'gpt-6-sol', native, build_pairs(1), native)
            self.assertEqual(manifest['codex_binary'], expected)
            native.write_bytes(b'changed executable')
            self.assertNotEqual(codex_executable_metadata(native), manifest['codex_binary'])

    def test_missing_override_fails_before_provider_invocation(self):
        with self.assertRaises(FileNotFoundError):
            codex_executable_metadata('/nonexistent/agentanyl-codex-fixture')


if __name__ == '__main__':
    unittest.main()
