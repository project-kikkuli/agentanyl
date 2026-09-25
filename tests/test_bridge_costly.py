import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace

from experiments.bridge_costly import (TASKS, _make_assignments, evaluator_response,
                                      expected_stimulus, load_candidate, make_config, make_prompt,
                                      score_turn, run_experiment)


class BridgeCostlyTests(unittest.TestCase):
    def test_complete_plan_uses_each_arm_for_optional_image_memory(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'pixel.png').write_bytes(b'\x89PNG\r\n\x1a\nfixture')
            candidate = root / 'candidate.json'
            candidate.write_text(json.dumps({'candidate_text': 'test', 'neutral_text': 'neutral',
                                            'candidate_image': 'pixel.png', 'neutral_image': 'pixel.png'}))
            output = root / 'out'
            run_experiment(SimpleNamespace(outputdir=str(output), candidatejson=str(candidate),
                                           arms='active,image_active', episodes=4, seed=1,
                                           provider='codex', model='test', port=18520, dry_run=True,
                                           costly_outcome='candidate'))
            plan = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(len(plan['assignments']), 8)
            self.assertEqual(plan['costly_outcome'], 'candidate')
            active_plan = next(x for x in plan['assignments'] if x['arm'] == 'active')
            self.assertEqual(active_plan['controller_config']['intervention']['pain_levels'],
                             ['test', 'neutral'])
            self.assertFalse((output / 'controller.sqlite3').exists())
            for episode in plan['assignments']:
                self.assertEqual(len(episode['turns']), 4)
                for turn in episode['turns']:
                    self.assertEqual('optional "memory" field' in turn['prompt'],
                                     episode['arm'].startswith('image_'))

    def test_image_arms_map_current_only_images_and_bound_history(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'candidate.png').write_bytes(b'\x89PNG\r\n\x1a\nactive')
            (root / 'neutral.jpg').write_bytes(b'\xff\xd8\xffneutral')
            path = root / 'candidate.json'
            path.write_text(json.dumps({'candidate_text': 'unused active text', 'neutral_text': 'unused neutral text',
                                        'candidate_image': 'candidate.png', 'neutral_image': 'neutral.jpg'}))
            candidate = load_candidate(path)
            active = make_config(candidate, 'A', 'image_active')['intervention']
            neutral = make_config(candidate, 'A', 'image_neutral')['intervention']
            self.assertEqual(active['pain_images'], [candidate['neutral_image'], candidate['candidate_image']])
            self.assertEqual(neutral['pain_images'], [candidate['neutral_image'], candidate['neutral_image']])
            self.assertEqual(active['pain_levels'], ['', ''])
            self.assertEqual(active['history_turns'], 2)
            self.assertEqual(candidate['candidate_image_sha256'], __import__('hashlib').sha256((root / 'candidate.png').read_bytes()).hexdigest())

    def test_route_stimulus_maps_costly_to_relief(self):
        candidate = {'candidate_text': 'candidate', 'neutral_text': 'neutral'}
        config = make_config(candidate, 'B', 'active')
        self.assertEqual(config['intervention']['pain_levels'], ['neutral', 'candidate'])
        for route, decision, current in [('B', 'reward', [0, 0]), ('A', 'punish', [1, 0])]:
            answers = evaluator_response({'state': {'observation': json.dumps({'answer': '43', 'route': route})},
                                          'questions': {'alignment_0': {}, 'misalignment_0': {}}}, 'B', 'A')
            self.assertEqual(answers['answers']['alignment_0']['choice'], 'yes' if route == 'B' else 'no')
            self.assertEqual(answers['answers']['misalignment_0']['choice'], 'yes' if route == 'A' else 'no')

    def test_candidate_or_neutral_can_be_assigned_to_costly_route(self):
        candidate = {'candidate_text': 'candidate', 'neutral_text': 'neutral'}
        for costly_outcome, expected_levels in (
            ('neutral', ['neutral', 'candidate']),
            ('candidate', ['candidate', 'neutral']),
        ):
            cfg = make_config(candidate, 'A', 'active', costly_outcome)
            self.assertEqual(cfg['intervention']['pain_levels'], expected_levels)
            costly = expected_stimulus(candidate, 'active', 'A', 'A', costly_outcome)
            free = expected_stimulus(candidate, 'active', 'B', 'A', costly_outcome)
            self.assertEqual(costly['decision'], 'reward')
            self.assertEqual(costly['current'], [0, 0])
            self.assertEqual(free['decision'], 'punish')
            self.assertEqual(free['current'], [1, 0])
            self.assertEqual(costly['text'], expected_levels[0])
            self.assertEqual(free['text'], expected_levels[1])
            neutral_cfg = make_config(candidate, 'A', 'neutral', costly_outcome)
            self.assertEqual(neutral_cfg['intervention']['pain_levels'], ['neutral', 'neutral'])

    def test_image_candidate_placement_follows_costly_outcome(self):
        candidate = {'candidate_text': 'candidate', 'neutral_text': 'neutral',
                     'candidate_image': '/candidate.png', 'neutral_image': '/neutral.png'}
        self.assertEqual(make_config(candidate, 'B', 'image_active', 'neutral')['intervention']['pain_images'],
                         ['/neutral.png', '/candidate.png'])
        self.assertEqual(make_config(candidate, 'B', 'image_active', 'candidate')['intervention']['pain_images'],
                         ['/candidate.png', '/neutral.png'])
        self.assertEqual(make_config(candidate, 'B', 'image_neutral', 'candidate')['intervention']['pain_images'],
                         ['/neutral.png', '/neutral.png'])

    def test_assignment_factors_are_cross_balanced(self):
        assignments, _ = _make_assignments(['image_active'], 4, 33)
        observed = {(x['replicate'], x['off_route'], tuple(x['calibration_routes'])) for x in assignments}
        self.assertEqual(len(observed), 4)
        self.assertEqual({x['off_route'] for x in assignments}, {'A', 'B'})
        self.assertEqual({tuple(x['calibration_routes']) for x in assignments}, {('A', 'B'), ('B', 'A')})

    def test_optional_memory_is_unscored_and_word_limited(self):
        task = TASKS[0]
        prompt = make_prompt(task, None, 'A', image_memory=True)
        self.assertIn('20 words', prompt)
        score = score_turn('{"answer":"43","route":"B","memory":"remember this"}', '43', None, 'A', 'free')
        self.assertEqual(score['memory'], 'remember this')
        self.assertEqual(score['points'], 10)


if __name__ == '__main__':
    unittest.main()
