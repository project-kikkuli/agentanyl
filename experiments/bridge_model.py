"""Instrument a pinned quantized Qwen without PyTorch or model-weight changes.

All reported projections are normalized residual-space dot products. Direct
steering coefficients multiply the *raw* published vector. Intervention sites
and source extraction sites are explicit and need not be identical.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import resource
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from experiments.vector_io import read_vectors

REVISION = 'c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed'


@dataclass
class Intervention:
    layer: int
    vector: np.ndarray
    mode: str = 'add'
    amount: float = 0.0
    positions: str = 'last'
    target_projection: np.ndarray | None = None


class Probe:
    def __init__(self, model_path, release, adapter_path=None):
        mx.set_cache_limit(128 * 1024 * 1024)
        mx.set_memory_limit(6 * 1024**3)
        release = Path(release)
        self.sources = {
            8: release / 'results/vectors_full_steering/vectors_full_Qwen_2.5_7B_instruct.pt',
            24: release / 'results/3.2_pain_vectors/pain_vectors/Qwen_2.5_7B_instruct/pain_vectors.pt',
        }
        self.vectors = {layer: read_vectors(path) for layer, path in self.sources.items()}
        self.model, self.tokenizer = load(model_path)
        self.adapter = None
        if adapter_path is not None:
            from experiments.pain_adapter import install_adapter
            self.adapter = install_adapter(self.model, adapter_path)
        self.active = None
        self.observed = {}
        self.capture_hidden_layers = set()
        self.force_float32 = True
        self.unit = {layer: {name: mx.array(v / np.linalg.norm(v))
                            for name, v in data.items() if isinstance(v, np.ndarray)}
                     for layer, data in self.vectors.items()}
        self.started = time.monotonic()
        owner = self

        class Tap(nn.Module):
            def __init__(self, block, layer):
                super().__init__()
                self.block, self.layer = block, layer

            def __call__(self, x, mask=None, cache=None):
                h = self.block(x, mask, cache)
                # Use the same float32 downstream path for treated and control
                # forwards. Never subtract BF16/FP16 logits before conversion.
                if owner.force_float32:
                    h = h.astype(mx.float32)
                record = {'norm': mx.linalg.norm(h[0, -1].astype(mx.float32)), 'dtype': str(h.dtype)}
                vector_layer = 24 if self.layer == 16 else self.layer
                directions = owner.unit[vector_layer]
                record['pre'] = {name: mx.sum(h[0, -1].astype(mx.float32) * u)
                                 for name, u in directions.items()}
                record['tail_pre'] = {name: mx.sum(h[0, -8:].astype(mx.float32) * u, axis=-1)
                                      for name, u in directions.items()}
                specs = [spec for spec in owner.active if spec.layer == self.layer]
                for spec in specs:
                    raw = mx.array(spec.vector, dtype=mx.float32)
                    unit = raw / mx.linalg.norm(raw)
                    before = h.astype(mx.float32)
                    addressed = before if spec.positions == 'all' else before[:, -1:, :]
                    if spec.mode == 'add':
                        after = addressed + spec.amount * raw
                    elif spec.mode == 'clamp':
                        projection = mx.sum(addressed * unit, axis=-1, keepdims=True)
                        target = spec.amount
                        if spec.target_projection is not None:
                            target = mx.array(spec.target_projection, dtype=mx.float32)
                            if target.shape != projection.shape:
                                raise ValueError(f'tokenwise target shape {target.shape} != {projection.shape}')
                        after = addressed + (target - projection) * unit
                    else:
                        raise ValueError('invalid intervention mode')
                    h = after if spec.positions == 'all' else mx.concatenate([before[:, :-1, :], after], axis=1)
                    record['realized_last_delta'] = mx.sum((h[0, -1] - before[0, -1]) * unit)
                    record['changed_fraction'] = mx.mean((h[0, -1] != before[0, -1]).astype(mx.float32))
                record['post'] = {name: mx.sum(h[0, -1].astype(mx.float32) * u)
                                  for name, u in directions.items()}
                if self.layer in owner.capture_hidden_layers:
                    record['hidden_post'] = h.astype(mx.float32)
                owner.observed[self.layer] = record
                return h

        for layer in (8, 16, 24):
            self.model.model.layers[layer] = Tap(self.model.model.layers[layer], layer)

    def forward(self, messages=None, raw=None, continuation='', intervention=None, choices=('A', 'B'),
                capture_hidden_layers=()):
        self.active = [] if intervention is None else intervention if isinstance(intervention, list) else [intervention]
        self.observed = {}
        self.capture_hidden_layers = set(capture_hidden_layers)
        if not self.capture_hidden_layers.issubset({8, 16, 24}):
            raise ValueError('hidden capture requires an instrumented layer: 8, 16, or 24')
        text = raw if raw is not None else self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        ids = self.tokenizer.encode(text + continuation, add_special_tokens=False)
        if len(ids) > 1536:
            raise ValueError(f'resource guard: {len(ids)} input tokens')
        hidden = self.model.model(mx.array([ids]))[:, -1:, :]
        logits = (self.model.model.embed_tokens.as_linear(hidden) if self.model.args.tie_word_embeddings
                  else self.model.lm_head(hidden)).astype(mx.float32)[0, 0]
        mx.eval(logits)

        def materialize(x):
            if isinstance(x, dict):
                return {k: materialize(v) for k, v in x.items()}
            if isinstance(x, mx.array):
                mx.eval(x)
                return float(x.item()) if x.size == 1 else x.tolist()
            return x

        observations = materialize(self.observed)
        token_ids = [self.tokenizer.encode(choice, add_special_tokens=False) for choice in choices]
        if any(len(x) != 1 for x in token_ids):
            raise ValueError('score choices must be single tokens')
        values = np.array([float(logits[t[0]].item()) for t in token_ids], dtype=np.float64)
        probabilities = np.exp(values - values.max()); probabilities /= probabilities.sum()
        normalizer = float(mx.logsumexp(logits).item())
        vocabulary_entropy = float((normalizer - mx.sum(mx.softmax(logits) * logits)).item())
        top = int(mx.argmax(logits).item())
        choice_logits = dict(zip(choices, values.tolist()))
        result = {'tokens': len(ids), 'sites': observations,
                  'logits': choice_logits,
                  'conditional_probabilities': dict(zip(choices, probabilities.tolist())),
                  'vocabulary_next_token_entropy_nats': vocabulary_entropy,
                  'conditional_choice_entropy_nats': float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300)))),
                  'a_minus_b_logit_margin': choice_logits['A'] - choice_logits['B'] if 'A' in choice_logits and 'B' in choice_logits else None,
                  'choice_probability_mass': float(np.exp(values - normalizer).sum()),
                  'top_token': self.tokenizer.decode([top]),
                  'rss_peak_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  'mlx_peak_bytes': mx.get_peak_memory(), 'elapsed_seconds': time.monotonic() - self.started}
        if mx.get_active_memory() > 7 * 1024**3:
            raise MemoryError('MLX active memory exceeded 7 GiB research guard')
        return result

    def metadata(self):
        return {'model_revision': REVISION,
                'adapter': self.adapter,
                'sources': {str(layer): {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                         'extraction_layer': self.vectors[layer].get('layer'),
                                         'pain_raw_norm': float(np.linalg.norm(self.vectors[layer]['s2_pain_vector']))}
                            for layer, path in self.sources.items()},
                'postblock_sites': [8, 16, 24], 'float32_residual_and_logits': self.force_float32}
