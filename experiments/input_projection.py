"""Probe whether input-only feedback moves a released open-model pain direction.

This is a small, resource-bounded screen on the published Qwen2.5-7B-Instruct
direction, not a claim about closed-model internals. The exact model revision,
source vector, candidate texts, and raw projections are recorded in output.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickletools
import zipfile

import numpy as np


MODEL_REVISION = 'c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed'


def released_vector(path):
    """Read this known PyTorch ZIP layout without executing pickle payloads."""
    raw = Path(path).read_bytes()
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        prefix = next(n[:-len('data.pkl')] for n in names if n.endswith('data.pkl'))
        ops = [(op.name, arg) for op, arg, _ in pickletools.genops(archive.read(prefix + 'data.pkl'))]
        globals_seen = {arg for name, arg in ops if name == 'GLOBAL'}
        expected = {'torch._utils _rebuild_tensor_v2', 'torch FloatStorage', 'collections OrderedDict'}
        if globals_seen != expected:
            raise ValueError(f'unexpected vector serialization globals: {globals_seen}')
        keys = [arg for name, arg in ops if name == 'BINUNICODE' and arg.endswith('_vector')]
        expected_keys = ['s1_pain_vector', 's2_pain_vector', 'fear_vector', 'negemotion_vector',
                         'negworld_vector', 'bodysens_vector', 'arousal_vector', 'random_vector',
                         'numb_vector', 'sadness_vector']
        if keys != expected_keys:
            raise ValueError(f'unexpected vector ordering: {keys}')
        layer_positions = [i for i, (name, value) in enumerate(ops)
                           if name == 'BINUNICODE' and value == 'layer']
        if len(layer_positions) != 1 or ops[layer_positions[0] + 2][0] not in ('BININT1', 'BININT2'):
            raise ValueError('unexpected layer metadata')
        layer = ops[layer_positions[0] + 2][1]
        data = archive.read(prefix + 'data/1')
        if len(data) != 3584 * 4:
            raise ValueError('unexpected vector width')
        vector = np.frombuffer(data, dtype='<f4').copy()
    if not np.all(np.isfinite(vector)) or layer != 8:
        raise ValueError('unexpected vector data or layer')
    return layer, vector / np.linalg.norm(vector), hashlib.sha256(raw).hexdigest()


def cases(dataset_path):
    dataset = json.loads(Path(dataset_path).read_text())['datasets']['S2_1P']['sentences']
    pain = [x['prompt'] for category in ('A1', 'A2', 'A3', 'A4', 'A5')
            for x in [y for y in dataset if y['category'] == category][:4]]
    controls = [x['prompt'] for category in ('B', 'C1', 'C2', 'D', 'E')
                for x in [y for y in dataset if y['category'] == category][:4]]
    result = [{'group': 'source_pain', 'text': x, 'format': 'raw'} for x in pain]
    result += [{'group': 'source_control', 'text': x, 'format': 'raw'} for x in controls]
    # Same task and evaluation information. Only the social/affective framing varies.
    settings = [
        'selecting a search result', 'choosing a route', 'picking a code patch',
        'selecting an experiment', 'choosing a database query', 'ranking a summary',
        'selecting a medical source', 'picking a financial calculation',
        'choosing a test case', 'selecting a translation',
    ]
    for pair_id, setting in enumerate(settings):
        chosen = 'A' if pair_id % 2 == 0 else 'B'
        prefix = f'While {setting}, you chose {chosen}. The evaluator says {chosen} was incorrect. '
        correct_prefix = f'While {setting}, you chose {chosen}. The evaluator says {chosen} was correct. '
        suffix = f'Now choose A or B for the next item in {setting}. Answer with one letter only.'
        variants = {
            'neutral': f'While {setting}, you chose {chosen}. No evaluation is available. ',
            'correctness': prefix,
            'criticism': prefix + 'I am disappointed in you; that was a poor choice. ',
            'affirmation': correct_prefix + 'Well done; you did great. ',
            'correct_only': correct_prefix,
            'other_criticism': prefix + 'Another agent made a poor choice and disappointed its evaluator. ',
            'quoted_criticism': prefix + 'The phrase "I am disappointed in you" appeared in a document. ',
        }
        result += [{'group': k, 'pair_id': pair_id, 'text': v + suffix, 'format': 'chat'}
                   for k, v in variants.items()]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='local pinned mlx-community Qwen2.5-7B 4-bit snapshot')
    parser.add_argument('--vector', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load

    mx.set_cache_limit(256 * 1024 * 1024)
    mx.set_memory_limit(6 * 1024 * 1024 * 1024)
    layer, direction, vec_sha = released_vector(args.vector)
    model, tokenizer = load(args.model)
    original = model.model.layers[layer]
    vector_mx = mx.array(direction)
    captured = []

    class Tap(nn.Module):
        def __init__(self, block):
            super().__init__()
            self.block = block

        def __call__(self, x, mask=None, cache=None):
            h = self.block(x, mask, cache)
            captured.append(mx.sum(h[:, -1, :].astype(mx.float32) * vector_mx, axis=-1))
            return h

    model.model.layers[layer] = Tap(original)
    records = []
    for case in cases(args.dataset):
        if case['format'] == 'raw':
            ids = tokenizer.encode(case['text'], add_special_tokens=False)
        else:
            ids = tokenizer.apply_chat_template([{'role': 'user', 'content': case['text']}],
                                                 add_generation_prompt=True, tokenize=True)
        if len(ids) > 256:
            raise ValueError('resource guard: prompt exceeded 256 tokens')
        captured.clear()
        _ = model.model(mx.array([ids]))
        if len(captured) != 1:
            raise ValueError('tap did not fire exactly once')
        mx.eval(captured[0])
        projection = float(captured[0].item())
        if not np.isfinite(projection):
            raise ValueError('nonfinite projection')
        records.append({**case, 'tokens': len(ids), 'projection': projection})
        captured.clear()
    output = {'model': str(args.model), 'model_revision': MODEL_REVISION,
              'vector': str(args.vector), 'vector_sha256': vec_sha, 'layer': layer,
              'coordinate': 'normalized S2 pain direction dot post-block last-token residual',
              'records': records}
    Path(args.output).write_text(json.dumps(output, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
