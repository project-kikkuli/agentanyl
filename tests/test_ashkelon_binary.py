import hashlib
import tempfile
import unittest
from pathlib import Path

from experiments.bridge_harness import (
    DEFAULT_ASHKELON,
    ashkelon_binary_metadata,
    resolve_ashkelon_path,
)


class AshkelonBinarySelectionTests(unittest.TestCase):
    def test_defaults_to_installed_repo_release_binary(self):
        self.assertEqual(resolve_ashkelon_path(environ={}), DEFAULT_ASHKELON.resolve())
        self.assertNotIn('/tmp/agentanyl-ashkelon/', str(DEFAULT_ASHKELON))

    def test_environment_override_is_resolved(self):
        path = resolve_ashkelon_path(environ={'AGENTANYL_ASHKELON': '~/ashkelon-custom'})
        self.assertEqual(path, (Path.home() / 'ashkelon-custom').resolve())

    def test_explicit_path_takes_precedence_over_environment(self):
        path = resolve_ashkelon_path('/tmp/explicit-ashkelon', {
            'AGENTANYL_ASHKELON': '/tmp/environment-ashkelon',
        })
        self.assertEqual(path, Path('/tmp/explicit-ashkelon').resolve())

    def test_metadata_identifies_exact_binary_bytes_and_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / 'ashkelon'
            binary.write_bytes(b'fake ashkelon executable')
            metadata = ashkelon_binary_metadata(binary)
        self.assertEqual(metadata['path'], str(binary.resolve()))
        self.assertEqual(metadata['sha256'], hashlib.sha256(b'fake ashkelon executable').hexdigest())


if __name__ == '__main__':
    unittest.main()
