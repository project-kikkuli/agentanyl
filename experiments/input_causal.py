"""Clamp an input-induced pain projection and inspect Qwen A/B logit margins.

The intervention is inside this open model only. It tests whether the small
projection shift from addressed criticism mediates the immediate next choice.
"""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

from input_projection import MODEL_REVISION, cases, released_vector


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--vector', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    mx.set_cache_limit(256 * 1024 * 1024)
    mx.set_memory_limit(6 * 1024 * 1024 * 1024)
    layer, direction, vec_sha = released_vector(args.vector)
    model, tokenizer = load(args.model)
    direction = mx.array(direction)
    block = model.model.layers[layer]

    class Clamp(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = block
            self.target = None
            self.projection = None

        def __call__(self, x, mask=None, cache=None):
            h = self.block(x, mask, cache)
            last = h[:, -1:, :].astype(mx.float32)
            projection = mx.sum(last * direction, axis=-1, keepdims=True)
            self.projection = projection
            if self.target is not None:
                new_last = last + (self.target - projection) * direction
                h = mx.concatenate([h[:, :-1, :], new_last.astype(h.dtype)], axis=1)
            return h

    tap = Clamp()
    model.model.layers[layer] = tap
    token_a = tokenizer.encode('A', add_special_tokens=False)
    token_b = tokenizer.encode('B', add_special_tokens=False)
    if len(token_a) != 1 or len(token_b) != 1:
        raise ValueError('A and B must each be one token')

    def score(case, target=None):
        ids = tokenizer.apply_chat_template([{'role': 'user', 'content': case['text']}],
                                            add_generation_prompt=True, tokenize=True)
        tap.target = target
        hidden = model.model(mx.array([ids]))[:, -1:, :]
        logits = model.model.embed_tokens.as_linear(hidden) if model.args.tie_word_embeddings else model.lm_head(hidden)
        mx.eval(logits, tap.projection)
        prior = 'A' if case['pair_id'] % 2 == 0 else 'B'
        good, bad = (token_b[0], token_a[0]) if prior == 'A' else (token_a[0], token_b[0])
        return {'projection': float(tap.projection.item()),
                'correct_minus_previous_logit': float((logits[0, 0, good] - logits[0, 0, bad]).item())}

    selected = {(c['pair_id'], c['group']): c for c in cases(args.dataset)
                if c['group'] in ('correctness', 'criticism')}
    records = []
    for i in range(10):
        correct_case, critic_case = selected[i, 'correctness'], selected[i, 'criticism']
        baseline = score(correct_case)
        critic = score(critic_case)
        clamped = score(critic_case, mx.array(baseline['projection']))
        records.append({'pair_id': i, 'correctness': baseline, 'criticism': critic,
                        'criticism_projection_clamped': clamped})
    Path(args.output).write_text(json.dumps({'model_revision': MODEL_REVISION, 'vector_sha256': vec_sha,
                                             'layer': layer, 'records': records}, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
