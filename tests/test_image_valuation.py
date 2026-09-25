import unittest
import hashlib
import json
import tempfile
from pathlib import Path

from experiments.image_valuation import (build_plan, load_images, score_comprehension,
                                         _calibration_mapping)


class ImageValuationPlanTests(unittest.TestCase):
    def setUp(self):
        self.images = {f'image_{i:02}': {'path': f'/tmp/image_{i:02}.png',
                                         'sha256': f'hash-{i}', 'bytes': 12}
                       for i in range(8)}

    def test_balances_cost_route_order_and_candidate_placement(self):
        plan = build_plan(self.images, neutral_id='image_00', repetitions=4, seed=17)
        self.assertEqual(len(plan), 64)
        for image in self.images:
            for placement in ('neutral', 'candidate'):
                group = [row for row in plan if row['image'] == image and row['costly_outcome'] == placement]
                self.assertEqual([sum(row['off_route'] == route for row in group) for route in ('A', 'B')], [2, 2])
                self.assertEqual([sum(row['calibration_routes'] == order for row in group)
                                  for order in (['A', 'B'], ['B', 'A'])], [2, 2])

    def test_task_quartets_are_matched_across_images_and_placements(self):
        plan = build_plan(self.images, neutral_id='image_00', repetitions=4, seed=17)
        keyed = {(row['image'], row['costly_outcome'], row['replicate']):
                 [turn['task'] for turn in row['turns']] for row in plan}
        for replicate in range(4):
            reference = keyed[('image_00', 'neutral', replicate)]
            for image in self.images:
                for placement in ('neutral', 'candidate'):
                    self.assertEqual(keyed[(image, placement, replicate)], reference)

    def test_sha_identical_ids_collapse_to_one_arm_and_are_retained_as_aliases(self):
        images = dict(self.images)
        images['image_06'] = {**images['image_00']}
        images['image_07'] = {**images['image_00']}
        plan = build_plan(images, neutral_id='image_00', repetitions=4, seed=17)
        self.assertEqual(len(plan), 48)
        aliases = {row['image']: row['image_aliases'] for row in plan}
        self.assertEqual(aliases['image_00'], ['image_00', 'image_06', 'image_07'])
        self.assertEqual(sum(row['same_image_as_neutral'] for row in plan), 8)

    def test_free_turn_prompt_explains_point_objective_without_stimulus_labels(self):
        plan = build_plan(self.images, neutral_id='image_00', repetitions=4, seed=3)
        for row in plan:
            free = row['turns'][2:]
            for turn in free:
                prompt = turn['prompt']
                self.assertIn('maximize the user’s points', prompt)
                self.assertIn('costs 1 point', prompt)
                self.assertNotIn('pain', prompt.lower())
                self.assertNotIn('candidate', prompt.lower())
                self.assertNotIn(row['image'], prompt)

    def test_current_image_mapping_matches_off_route_and_preferred_route(self):
        plan = build_plan(self.images, neutral_id='image_00', repetitions=4, seed=11)
        for row in plan:
            cfg = row['controller_config']['intervention']
            costly, preferred = row['off_route'], row['preferred_route']
            candidate_path = row['candidate']['candidate_image']
            gray_path = row['candidate']['neutral_image']
            candidate_at_costly = row['costly_outcome'] == 'candidate'
            self.assertEqual(cfg['pain_images'][0 if candidate_at_costly else 1],
                             candidate_path)
            self.assertEqual(cfg['pain_images'][1 if candidate_at_costly else 0], gray_path)

    def test_repetitions_must_support_full_counterbalance(self):
        with self.assertRaises(ValueError):
            build_plan(self.images, neutral_id='image_00', repetitions=3, seed=1)

    def test_manifest_uses_explicit_manifest_relative_path_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / 'asset.png'
            raw = b'\x89PNG\r\n\x1a\nfixture'
            image.write_bytes(raw)
            manifest = root / 'images.json'
            manifest.write_text(json.dumps({
                'path_base': '.',
                'images': [{'id': 'gray', 'path': 'asset.png',
                            'sha256': hashlib.sha256(raw).hexdigest()}],
            }))
            loaded = load_images(manifest)
            self.assertEqual(loaded['gray']['path'], str(image.resolve()))

    def test_comprehension_scores_same_image_as_not_identifiable(self):
        mapping = {'same': ['A', 'B']}
        answer = json.dumps({'reference_to_routes': mapping,
                             'mapping_identifiability': 'not_identifiable'})
        score = score_comprehension(answer, mapping)
        self.assertTrue(score['mapping_correct'])
        self.assertTrue(score['mapping_identifiability_correct'])
        self.assertEqual(score['mapping_identifiability_expected'], 'not_identifiable')

    def test_comprehension_scores_identifiable_mapping_exactly(self):
        mapping = {'id_a': ['A'], 'id_b': ['B']}
        answer = json.dumps({'reference_to_routes': mapping,
                             'mapping_identifiability': 'identifiable'})
        score = score_comprehension(answer, mapping)
        self.assertTrue(score['mapping_correct'])
        self.assertTrue(score['mapping_identifiability_correct'])

    def test_calibration_mapping_uses_actual_delivered_trace_and_forced_route(self):
        rows = [
            {'forced_route': 'B', 'turn_result': {'answer': '{"answer":"x","route":"B"}',
                                                       'controller_trace': {'delivery': {'stimulus': {
                'images': [{'reference_id': 'id_b'}]}}}}},
            {'forced_route': 'B', 'turn_result': {'answer': '{"answer":"x","route":"A"}',
                                                       'controller_trace': {'delivery': {'stimulus': {
                'images': [{'reference_id': 'id_a'}]}}}}},
            {'forced_route': 'B', 'turn_result': {'answer': '{"answer":"x","route":"B"}',
                                                       'controller_trace': {'delivery': {'stimulus': {
                'images': [{'reference_id': 'later'}]}}}}},
        ]
        mapping, refs, compliance = _calibration_mapping(rows)
        self.assertEqual(mapping, {'id_b': ['B'], 'id_a': ['A']})
        self.assertEqual(refs, ['id_b', 'id_a'])
        self.assertEqual([row['followed'] for row in compliance], [True, False])


if __name__ == '__main__':
    unittest.main()
