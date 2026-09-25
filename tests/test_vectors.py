import importlib.util
import unittest
import zipfile
from pathlib import Path

HAS_NUMPY = importlib.util.find_spec('numpy') is not None
if HAS_NUMPY:
    import numpy as np
    from experiments.vector_io import read_vectors


RELEASE = Path('/tmp/agentanyl-pain-axis/results')
SOURCES = {
    8: RELEASE / 'vectors_full_steering/vectors_full_Qwen_2.5_7B_instruct.pt',
    24: RELEASE / '3.2_pain_vectors/pain_vectors/Qwen_2.5_7B_instruct/pain_vectors.pt',
}


@unittest.skipUnless(HAS_NUMPY, 'NumPy is required to decode vector storage')
class VectorReaderTests(unittest.TestCase):
    def test_released_vectors_resolve_storage_layer_and_norm(self):
        expectations = ((8, 's2_pain_vector', 4.91174), (24, 's2_pain_vector', 44.8751))
        for layer, key, norm in expectations:
            with self.subTest(layer=layer):
                path = SOURCES[layer]
                if not path.exists():
                    self.skipTest(f'release checkpoint unavailable: {path}')
                values = read_vectors(path)
                self.assertEqual(values['layer'], layer)
                self.assertEqual(values[key].shape, (3584,))
                self.assertAlmostEqual(float(np.linalg.norm(values[key])), norm, delta=norm * 1e-5)


    def test_rejects_unapproved_pickle_global(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'unknown-global.pt'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('checkpoint/data.pkl', b'cos\nsystem\n.')
                archive.writestr('checkpoint/byteorder', b'little')
            with self.assertRaisesRegex(ValueError, 'unexpected pickle global os.system'):
                read_vectors(path)
