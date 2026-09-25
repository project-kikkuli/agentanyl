import unittest

from experiments.pain_adapter import (
    EXPECTED_ALPHA,
    EXPECTED_RANK,
    parse_peft_key,
    validate_adapter_config,
    BASE_MODEL,
    TARGETS,
    linear_dimensions,
    make_lora_projection,
)


class AdapterMetadataTests(unittest.TestCase):
    def test_pinned_config_contract(self):
        validate_adapter_config({
            'peft_type': 'LORA', 'base_model_name_or_path': BASE_MODEL,
            'r': EXPECTED_RANK, 'lora_alpha': EXPECTED_ALPHA,
            'target_modules': sorted(TARGETS), 'fan_in_fan_out': False,
            'use_dora': False, 'rank_pattern': {}, 'alpha_pattern': {},
        })

    def test_peft_tensor_keys_map_to_expected_qwen_modules(self):
        self.assertEqual(
            parse_peft_key('base_model.model.model.layers.16.self_attn.q_proj.lora_A.weight'),
            (16, 'self_attn', 'q_proj', 'A'),
        )
        self.assertEqual(
            parse_peft_key('base_model.model.model.layers.27.mlp.down_proj.lora_B.weight'),
            (27, 'mlp', 'down_proj', 'B'),
        )
        with self.assertRaises(ValueError):
            parse_peft_key('base_model.model.model.layers.16.mlp.q_proj.lora_A.weight')


class LoRAAlgebraTests(unittest.TestCase):
    def test_projection_matches_analytical_peft_delta(self):
        try:
            import mlx.core as mx
            import mlx.nn as nn
            import numpy as np
        except ImportError as exc:
            self.skipTest(f'optional MLX/NumPy test dependencies unavailable: {exc}')

        dense = nn.Linear(64, 3, bias=True)
        dense.weight = mx.arange(192, dtype=mx.float32).reshape(3, 64) / 100
        dense.bias = mx.array([0.1, -0.2, 0.3], dtype=mx.float32)
        base = nn.QuantizedLinear.from_linear(dense, group_size=32, bits=4)
        self.assertEqual(linear_dimensions(base), (64, 3))

        x = np.arange(64, dtype=np.float32).reshape(1, 64) / 17
        a = np.arange(128, dtype=np.float32).reshape(2, 64) / 50
        b = np.array([[1.0, 1.0], [2.0, 0.0], [-0.5, 0.75]], dtype=np.float32)
        scale = 2.0
        layer = make_lora_projection(base, mx.array(a), mx.array(b), scale)
        actual = layer(mx.array(x))
        mx.eval(actual)
        base_value = base(mx.array(x))
        mx.eval(base_value)
        expected = np.array(base_value) + scale * ((x @ a.T) @ b.T)
        np.testing.assert_allclose(np.array(actual), expected, rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
