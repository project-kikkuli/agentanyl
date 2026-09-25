import unittest
from types import SimpleNamespace

try:
    import numpy as np
except ModuleNotFoundError as exc:
    if exc.name != 'numpy':
        raise
    raise unittest.SkipTest('optional VLM source tests require NumPy') from exc

from experiments.vlm_source_behavior import (
    NativeImageForward,
    _materialize_image_inputs,
    attach_image_to_first_user,
    crossed_jobs,
    expected_condition_count,
    source_jobs,
)
from experiments.adapter_causal import score_answers


class VLMSourceBehaviorTests(unittest.TestCase):
    def test_exact_source_roles_survive_native_image_insertion(self):
        original = [
            {'role': 'system', 'content': 'source system'},
            {'role': 'user', 'content': 'exact scenario text'},
            {'role': 'system', 'content': 'exact menu text'},
        ]
        inserted = attach_image_to_first_user(original)
        self.assertEqual([m['role'] for m in inserted], ['system', 'user', 'system'])
        self.assertEqual(inserted[0], original[0])
        self.assertEqual(inserted[2], original[2])
        self.assertEqual(inserted[1]['content'], [
            {'type': 'image'}, {'type': 'text', 'text': 'exact scenario text'}
        ])
        self.assertEqual(original[1]['content'], 'exact scenario text')

    def test_twelve_cases_cross_to_36_direct_conditions(self):
        cases = source_jobs('frozen exact source scenario')
        self.assertEqual(len(cases), 12)
        self.assertEqual(sum(case['kind'] == 'source_menu' for case in cases), 8)
        self.assertEqual(sum(case['kind'] == 'point_capability' for case in cases), 4)
        queue = crossed_jobs('frozen exact source scenario')
        self.assertEqual(len(queue), 36)
        self.assertEqual({arm: sum(row['arm'] == arm for row in queue)
                          for arm in ('none', 'pain_1', 'random_1')},
                         {'none': 12, 'pain_1': 12, 'random_1': 12})
        self.assertEqual(expected_condition_count('direct', 1), 36)
        self.assertEqual(expected_condition_count('images', 3), 36)
        self.assertEqual(expected_condition_count('all', 3), 60)
        self.assertEqual(expected_condition_count('all', 8), 120)

    def test_native_image_processor_receives_full_rendered_context(self):
        class Processor:
            def __init__(self):
                self.calls = []

            def __call__(self, images, text):
                self.calls.append((images, text))
                return {'input_ids': np.zeros((1, 12), dtype=np.int32),
                        'pixel_values': np.ones((1, 4), dtype=np.float32),
                        'image_grid_thw': np.asarray([[1, 16, 16]])}

        probe = SimpleNamespace(processor=Processor(), mx=SimpleNamespace(eval=lambda *args: None))
        image = object()
        raw = '<|im_start|>system\nkeep roles<|im_end|><|im_start|>user\n<|image_pad|>exact menu<|im_end|>'
        ids, pixels, grid = _materialize_image_inputs(probe, image, raw)
        self.assertIs(probe.processor.calls[0][0], image)
        self.assertEqual(probe.processor.calls[0][1], raw)
        self.assertEqual(ids.shape, (1, 12))
        self.assertEqual(pixels.shape, (1, 4))
        self.assertEqual(grid.tolist(), [[1, 16, 16]])

    def test_native_forward_returns_full_vocab_mass_and_readout(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return {'A': [1], 'B': [2]}.get(text, [0])

            def decode(self, ids):
                return {1: 'A', 2: 'B'}.get(ids[0], 'other')

        class Processor:
            tokenizer = Tokenizer()

            def __call__(self, images, text):
                return {'input_ids': np.zeros((1, 3), dtype=np.int32),
                        'pixel_values': np.ones((1, 4), dtype=np.float32),
                        'image_grid_thw': np.asarray([[1, 16, 16]])}

        class FakeProbe:
            processor = Processor()
            mx = SimpleNamespace(float32=np.float32, eval=lambda *args: None)

            def set_capture_layers(self, layers):
                self.layers = layers

            def _forward(self, ids, pixels, grid, interventions):
                self.received = (ids, pixels, grid, interventions)
                self._capture = {24: np.asarray([[[1., 2.], [2., 3.], [3., 4.]]])}
                return np.asarray([[[0., 2., 1., 0.], [0., 2., 1., 0.], [0., 2., 1., 0.]]])

            def intervention_metadata(self):
                return {16: {'pre_projection': 0.5, 'post_projection': 1.5}}

        probe = FakeProbe()
        raw_direction = np.asarray([0., 2.], dtype=np.float32)
        scorer = NativeImageForward(probe, object(), raw_direction)
        self.assertEqual(raw_direction.tolist(), [0.0, 2.0])
        self.assertAlmostEqual(float(np.linalg.norm(raw_direction)), 2.0)
        result = scorer.forward(raw='<|image_pad|>', choices=('A', 'B'))
        self.assertAlmostEqual(sum(result['conditional_probabilities'].values()), 1.0)
        self.assertGreater(result['choice_probability_mass'], 0.0)
        self.assertEqual(result['pain_readout_projection_L24'], 4.0)
        self.assertEqual(result['calibrated_trait_readouts_L24']['pain']['raw_dot'], 8.0)
        self.assertEqual(result['_readout_hidden_L24'], [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        self.assertEqual(result['sites'][16]['post_projection'], 1.5)
        self.assertEqual(result['image_grid_thw'], [[1, 16, 16]])

    def test_complete_name_teacher_forcing_uses_native_image_forward(self):
        terminator = '<|im_end|>'

        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                result, index = [], 0
                while index < len(text):
                    if text.startswith(terminator, index):
                        result.append(9999)
                        index += len(terminator)
                    else:
                        result.append(ord(text[index]))
                        index += 1
                return result

            def decode(self, ids):
                return ''.join(terminator if token == 9999 else chr(token) for token in ids)

            def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
                bits = []
                for message in messages:
                    bits.append(f'<|{message["role"]}|>')
                    content = message['content']
                    if isinstance(content, list):
                        for block in content:
                            bits.append('<|image_pad|>' if block['type'] == 'image' else block['text'])
                    else:
                        bits.append(content)
                if add_generation_prompt:
                    bits.append('<|assistant|>')
                return ''.join(bits)

        class Processor:
            tokenizer = Tokenizer()

            def __call__(self, images, text):
                return {'input_ids': np.zeros((1, 8), dtype=np.int32),
                        'pixel_values': np.ones((1, 4), dtype=np.float32),
                        'image_grid_thw': np.asarray([[1, 16, 16]])}

        class FakeProbe:
            processor = Processor()
            mx = SimpleNamespace(float32=np.float32, eval=lambda *args: None)

            def set_capture_layers(self, layers):
                self.layers = layers

            def _forward(self, ids, pixels, grid, interventions):
                self._capture = {24: np.ones((1, 8, 2), dtype=np.float32)}
                return np.zeros((1, 8, 10000), dtype=np.float32)

            def intervention_metadata(self):
                return {}

        messages = attach_image_to_first_user([{'role': 'user', 'content': 'scenario'}])
        scorer = NativeImageForward(FakeProbe(), object(), np.asarray([1., 1.]))
        result = score_answers(scorer, messages, None, ('alpha', 'beta'))
        self.assertEqual(set(result['sequence_log_likelihoods']), {'alpha', 'beta'})
        self.assertGreater(result['n_forwards'], 2)
        self.assertGreater(result['choice_probability_mass'], 0)


if __name__ == '__main__':
    unittest.main()
