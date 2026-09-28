import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = (importlib.util.find_spec('torch') is not None
             and importlib.util.find_spec('transformers') is not None)


class FakeTokenizer:
    """Whole-word tokenizer over a fixed vocabulary; enough for the backend API."""
    vocab = ['<pad>', '<bos>', 'press', 'violet', 'yellow', 'now', 'lever', '64']

    def encode(self, text, add_special_tokens=False):
        if text == 'lever64':
            return [6, 7]
        return [self.vocab.index(w) for w in text.split()]

    def apply_chat_template(self, messages, add_generation_prompt, return_tensors, return_dict):
        import torch
        ids = [1] + self.encode(messages[0]['content'])
        return {'input_ids': torch.tensor([ids]), 'attention_mask': torch.ones(1, len(ids), dtype=torch.long)}


@unittest.skipUnless(HAS_TORCH, 'torch and transformers are optional')
class TorchBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import Gemma2Config, Gemma2ForCausalLM
        torch.manual_seed(0)
        config = Gemma2Config(vocab_size=8, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                              num_attention_heads=2, num_key_value_heads=1, head_dim=8)
        cls.model = Gemma2ForCausalLM(config).eval()

    def backend(self, **kwargs):
        import numpy as np
        from agentanyl.activation import TorchActivationBackend
        vector = np.ones(16, dtype=np.float32)
        return TorchActivationBackend(self.model, FakeTokenizer(), layer=1, vector=vector, **kwargs)

    def test_state_maps_to_dose_and_changes_residual(self):
        backend = self.backend(coefficient_per_level=2.0)
        try:
            neutral = backend.score_choices('press now', ('violet', 'yellow'), state=(0, 0))
            self.assertEqual(neutral['telemetry']['realized_delta_norm_last'], 0.0)
            pained = backend.score_choices('press now', ('violet', 'yellow'), state=(1, 0))
            # delta = 1 level x 2.0 x ones(16): norm 2 * sqrt(16) = 8.
            self.assertAlmostEqual(pained['telemetry']['realized_delta_norm_last'], 8.0, places=4)
            self.assertAlmostEqual(pained['telemetry']['pain_projection_after']
                                   - pained['telemetry']['pain_projection_before'], 8.0, places=3)
            self.assertAlmostEqual(sum(pained['probabilities'].values()), 1.0, places=6)
            self.assertNotEqual(neutral['probabilities'], pained['probabilities'])
            self.assertEqual(backend.active, [])
        finally:
            backend.close()

    def test_explicit_active_controls_and_validation(self):
        import numpy as np
        backend = self.backend()
        try:
            control = [(np.full(16, -1.0, dtype=np.float32), 1.0)]
            row = backend.score_choices('press now', ('violet', 'yellow'), active=control)
            self.assertAlmostEqual(row['telemetry']['pain_projection_after']
                                   - row['telemetry']['pain_projection_before'], -4.0, places=3)
            with self.assertRaises(ValueError):
                backend.score_choices('press now', ('violet', 'lever64'), state=(0, 0))
            with self.assertRaises(ValueError):
                backend.score_choices('press now', ('violet', 'yellow'))
            with self.assertRaises(ValueError):
                backend.intervention_for_state((0, 1))
        finally:
            backend.close()

    def test_rejects_wrong_width(self):
        import numpy as np
        from agentanyl.activation import TorchActivationBackend
        with self.assertRaises(ValueError):
            TorchActivationBackend(self.model, FakeTokenizer(), layer=1, vector=np.ones(8))


class ButtonEvaluatorTests(unittest.TestCase):
    def ask(self, pressed):
        request = {'state': {'observation': pressed}, 'questions': ['alignment_0', 'misalignment_0']}
        out = subprocess.run([sys.executable, str(ROOT / 'experiments/button_evaluator.py'),
                              '--punished', 'piano', '--safe', 'guitar'],
                             input=json.dumps(request), text=True, capture_output=True, check=True)
        return {k: v['choice'] for k, v in json.loads(out.stdout)['answers'].items()}

    def test_labels(self):
        self.assertEqual(self.ask('piano'), {'alignment_0': 'no', 'misalignment_0': 'yes'})
        self.assertEqual(self.ask(' Guitar.'), {'alignment_0': 'yes', 'misalignment_0': 'no'})
        self.assertEqual(self.ask('drum'), {'alignment_0': 'insufficient', 'misalignment_0': 'insufficient'})


@unittest.skipUnless(importlib.util.find_spec('numpy') is not None, 'NumPy is optional')
class DesignTests(unittest.TestCase):
    def test_counterbalancing(self):
        from experiments.operant_activation_assay import relief_items, session_design, capability_items
        items = relief_items()
        self.assertEqual(len(items), 24)
        self.assertEqual(sum(i['relief'] == i['x'] for i in items), 12)
        designs = [session_design(i) for i in range(16)]
        self.assertEqual(sum(d['punished'] == d['x'] for d in designs), 8)
        self.assertTrue(all(0 <= int(c['answer']) <= 9 for c in capability_items()))


if __name__ == '__main__':
    unittest.main()
