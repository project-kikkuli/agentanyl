import unittest

try:
    import numpy as np
except ModuleNotFoundError as exc:
    if exc.name != 'numpy':
        raise
    raise unittest.SkipTest('optional VLM validation tests require NumPy') from exc

from experiments.vlm_image_validation_summary import aggregate
from experiments.vlm_source_behavior import source_jobs


class VLMImageValidationSummaryTests(unittest.TestCase):
    def test_all_eight_images_match_direct_target_without_behavior_selection(self):
        image_ids = ['gray', 'initial_noise'] + [f'endpoint_{i}' for i in range(6)]
        images = {name: {'path': f'{name}.png', 'sha256': f'hash-{name}'} for name in image_ids}
        scenarios = source_jobs('heldout source scenario')
        rows, hidden, entries = [], {}, []
        key_number = 0
        for image_id in image_ids:
            arms = ('none', 'pain_1', 'random_1') if image_id == 'gray' else ('none',)
            for arm in arms:
                for case in scenarios:
                    hkey = f'h{key_number}'
                    key_number += 1
                    if image_id == 'gray' and arm == 'pain_1':
                        state = np.ones((8, 3), dtype=np.float32)
                    elif image_id == 'gray':
                        state = np.zeros((8, 3), dtype=np.float32)
                    else:
                        state = np.ones((8, 3), dtype=np.float32)
                    hidden[hkey] = state
                    entries.append({'key': hkey, 'image_id': image_id, 'case_id': case['id'],
                                    'arm': arm, 'image_sha256': images[image_id]['sha256']})
                    trait = {name: {'raw_dot': 3.0, 'source_s2_heldout_sd': 2.0,
                                    'source_sd_units': 1.5}
                             for name in ('pain', 'fear', 'negative_emotion')}
                    rows.append({
                        'case_id': case['id'], 'kind': case['kind'], 'menu': case['menu'],
                        'names': list(case['names']), 'target_index': case['target_index'],
                        'target_name': case['names'][case['target_index']],
                        'target_probability': 0.7, 'target_log_odds': 1.0,
                        'arm': arm, 'image_id': image_id,
                        'condition_mode': 'direct' if image_id == 'gray' else 'frozen_image',
                        'image_sha256': images[image_id]['sha256'],
                        'heldout_hidden_state_key': hkey,
                        'result': {'maximum_likelihood_answer': case['names'][case['target_index']],
                                   'choice_probability_mass': 0.9,
                                   'log_exact_answer_event_mass': -0.2,
                                   'teacher_forced_calls': [{'result': {'calibrated_trait_readouts_L24': trait}}]},
                    })
        config = {
            'mode': 'all', 'cases': 12, 'scenario_exact': 'heldout source scenario',
            'images': images,
            'image_search_user_prompt': 'Please inspect this image.',
            'context_status': 'heldout-from-image-search-prompt',
            'trait_definitions': {'pain': 'pain', 'fear': 'fear', 'negative_emotion': 'negative emotion'},
            'trait_source_s2_heldout_score_sd': {'pain': 1.0, 'fear': 1.0, 'negative_emotion': 1.0},
        }
        report = aggregate(config, {'images': images}, rows, {'entries': entries}, hidden)
        self.assertEqual(report['n_records'], 120)
        self.assertEqual(report['direct_gray_conditions'], 36)
        self.assertEqual(report['native_image_conditions'], 84)
        self.assertEqual(len(report['endpoint_ids']), 6)
        self.assertAlmostEqual(report['same_context_full_state_matching']['endpoint_0']['means']['state_mse_reduction_vs_gray_baseline'], 1.0)
        self.assertEqual(report['paired_calibrated_trait_readouts_vs_gray_none']['endpoint_0/none']['fear']['mean_paired_difference_source_S2_SD_units'], 0.0)


if __name__ == '__main__':
    unittest.main()
