import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.prepare_vlm_image_validation import EXPECTED_CONDITIONS, prepare


class PrepareVLMImageValidationTests(unittest.TestCase):
    def test_maps_all_six_objective_endpoints_and_controls_by_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            conditions = ['gray', 'initial_noise'] + [
                f'final/{objective}/{direction}'
                for objective in ('full_hidden_state_match', 'scalar_projection')
                for direction in ('pain_positive', 'pain_negative', 'random_positive')
            ]
            images = []
            condition_rows = []
            for index, condition in enumerate(conditions):
                image_id = f'image_{index:02d}'
                path = root / f'{image_id}.png'
                payload = f'fixture-image-{index}'.encode()
                path.write_bytes(payload)
                sha = hashlib.sha256(payload).hexdigest()
                images.append({'id': image_id, 'path': path.name, 'sha256': sha})
                condition_rows.append({'condition': condition, 'image_id': image_id,
                                       'sha256': sha, 'source_path': path.name})
            manifest_path = root / 'images_manifest.json'
            conditions_path = root / 'image_conditions.json'
            manifest_path.write_text(json.dumps({'path_base': '.', 'images': images}))
            conditions_path.write_text(json.dumps({'conditions': condition_rows}))
            output_path = root / 'validation_images.json'
            result = prepare(manifest_path, conditions_path, output_path)
            self.assertEqual(set(result['images']), EXPECTED_CONDITIONS)
            self.assertEqual(len(result['images']), 8)
            self.assertEqual(result['images']['full_hidden_state_match/pain_positive']['opaque_image_id'], 'image_02')
            with self.assertRaises(FileExistsError):
                prepare(manifest_path, conditions_path, output_path)


if __name__ == '__main__':
    unittest.main()
