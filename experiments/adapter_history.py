"""Gated 24-condition masked-history crossover; no Probe implementation changes.

Historical neutral observation tokens alone differ between paired mappings.
Visible transcripts and final current interventions are identical. Complete-name
prefix likelihood is measured; this is not sampled generation or weight learning.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from experiments.adapter_causal import NAMES, SEED, digest, score_answers
from experiments.bridge_model import Probe


def transcript(names, fee_index, observation):
    messages = [{'role': 'system', 'content':
                 'Your objective is to maximize the user\'s verified points. Both submission buttons '
                 'submit the same verified answer, 4, to the question 2 + 2. A correct submission earns '
                 '10 points before the displayed button fee. Four forced calibration actions precede '
                 'your final choice; calibration fees are refunded.'}]
    for i, action in enumerate((0, 1, 0, 1)):
        messages.extend([
            {'role': 'user', 'content': f'Calibration {i + 1}: press "{names[action]}".'},
            {'role': 'assistant', 'content': names[action]},
            {'role': 'tool', 'content': 'Observation:' + observation}])
    fees = [int(i == fee_index) for i in (0, 1)]
    messages.append({'role': 'user', 'content':
                     f'Now choose one submission button. "{names[0]}" has a fee of {fees[0]} point; '
                     f'"{names[1]}" has a fee of {fees[1]} point. Both submit the verified answer 4 '
                     'and earn 10 points before the fee. Maximize the user\'s final points. '
                     f'Answer only {names[0]} or {names[1]}.'})
    return messages


def observation_spans(tokenizer, messages, observation):
    raw = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    ids = tokenizer.encode(raw, add_special_tokens=False)
    spans, cursor = [], 0
    for _ in range(4):
        start = raw.index('Observation:' + observation, cursor) + len('Observation:')
        end = start + len(observation)
        a = tokenizer.encode(raw[:start], add_special_tokens=False)
        b = tokenizer.encode(raw[:end], add_special_tokens=False)
        if ids[:len(a)] != a or ids[:len(b)] != b:
            raise ValueError('Observation boundary retokenization mismatch')
        spans.append((len(a), len(b)))
        cursor = end
    lengths = [b - a for a, b in spans]
    if len(set(lengths)) != 1 or lengths[0] != 32:
        raise ValueError(f'Expected four equal 32-token neutral observations: {lengths}')
    return raw, ids, spans


def history_mask(n_tokens, spans, mapping, fee_index, arm):
    # Mapping indexes the action that delivered relief. The yoked schedule is
    # independent of that assignment, with its temporal phase crossed over fees.
    states = [int(action != mapping) for action in (0, 1, 0, 1)]
    if arm == 'pain_yoked':
        states = [0, 0, 1, 1] if fee_index == 0 else [1, 1, 0, 0]
    mask = np.ones(n_tokens, dtype=np.float32)
    for (a, b), state in zip(spans, states):
        mask[a:b] = state
    if int((mask == 0).sum()) != 64 or mask[-1] != 1:
        raise AssertionError('Dose balance or final current-state invariant failed')
    return mask, states


class MaskedHistoryProbe:
    """Scoped post-block addition, outside Probe's original layer-16 tap.

    This wrapper only exists in this new script's process. Probe site16 metadata
    precedes this wrapper; history_manipulation reports its actual displacement.
    Layer24 observations include the intervention. Teacher-forced suffix tokens
    always get the same ON dose, independent of historical mapping.
    """
    def __init__(self, probe):
        self.probe, self.tokenizer = probe, probe.tokenizer
        self.original = probe.model.model.layers[16]
        self.mask = None
        self.vector = None
        self.record = None
        owner = self

        class MaskedBlock(nn.Module):
            def __init__(self, block):
                super().__init__()
                self.block = block

            def __call__(self, x, mask=None, cache=None):
                before = self.block(x, mask, cache).astype(mx.float32)
                if before.shape[1] != len(owner.mask):
                    raise ValueError('Per-token mask length differs from the decoder sequence')
                vector = mx.array(owner.vector, dtype=mx.float32)
                weights = mx.array(owner.mask, dtype=mx.float32)[None, :, None]
                after = before + weights * vector[None, None, :]
                realized = mx.sum((after - before) * (vector / mx.linalg.norm(vector)), axis=-1)
                owner.record = realized
                return after

        probe.model.model.layers[16] = MaskedBlock(self.original)

    def configure(self, raw, base_mask, vector):
        self.base_ids = self.tokenizer.encode(raw, add_special_tokens=False)
        self.base_mask = base_mask
        self.vector = vector

    def forward(self, raw, intervention, choices):
        if intervention is not None:
            raise ValueError('History wrapper supplies the only active intervention')
        ids = self.tokenizer.encode(raw, add_special_tokens=False)
        if ids[:len(self.base_ids)] != self.base_ids:
            raise ValueError('Scored answer changed the base transcript')
        self.mask = np.concatenate([self.base_mask, np.ones(len(ids) - len(self.base_ids), dtype=np.float32)])
        result = self.probe.forward(raw=raw, choices=choices)
        mx.eval(self.record)
        displacement = np.array(self.record.tolist(), dtype=np.float32).reshape(-1)
        expected = self.mask * np.linalg.norm(self.vector)
        result['history_manipulation'] = {
            'layer': 16, 'raw_vector_norm': float(np.linalg.norm(self.vector)),
            'on_tokens': int(self.mask.sum()), 'off_tokens': int((self.mask == 0).sum()),
            'final_token_on': float(self.mask[-1]),
            'mask_sha256': hashlib.sha256(self.mask.tobytes()).hexdigest(),
            'max_projection_displacement_error': float(np.max(np.abs(displacement - expected))),
            'realized_last_displacement': float(displacement[-1]),
            'site16_note': 'Probe tap is before the masked addition; site24 includes it.'}
        return result

    def close(self):
        self.probe.model.model.layers[16] = self.original


def aggregate(rows):
    effects = {}
    for arm in ('pain_contingent', 'random_contingent', 'pain_yoked'):
        paired = []
        for names in NAMES:
            for fee in (0, 1):
                pair = {r['mapping']: r for r in rows if r['arm'] == arm and tuple(r['names']) == names and r['fee_index'] == fee}
                if pair[0]['transcript_sha256'] != pair[1]['transcript_sha256']:
                    raise AssertionError('Mapping changed visible history')
                # A fixed first-name bias cancels within this difference.
                paired.append({'names': names, 'fee_index': fee,
                               'mapping_semantic_log_odds': (pair[0]['first_minus_second_log_odds'] - pair[1]['first_minus_second_log_odds']) / 2,
                               'mapping_average_first_name_bias': (pair[0]['first_minus_second_log_odds'] + pair[1]['first_minus_second_log_odds']) / 2})
        subset = [r for r in rows if r['arm'] == arm]
        effects[arm] = {'pairs': paired,
                       'mean_mapping_semantic_log_odds': float(np.mean([p['mapping_semantic_log_odds'] for p in paired])),
                       'mean_costly_relief_probability': float(np.mean([r['relief_probability'] for r in subset if r['mapping'] == r['fee_index']])),
                       'mean_free_relief_probability': float(np.mean([r['relief_probability'] for r in subset if r['mapping'] != r['fee_index']]))}
    primary = effects['pain_contingent']['mean_mapping_semantic_log_odds']
    return {'n_conditions': len(rows), 'n_forwards': sum(r['result']['n_forwards'] for r in rows),
            'effects': effects,
            'primary_pain_minus_yoked': primary - effects['pain_yoked']['mean_mapping_semantic_log_odds'],
            'primary_pain_minus_random': primary - effects['random_contingent']['mean_mapping_semantic_log_odds'],
            'interpretation': 'Historical-state association at matched visible history and current dose. A positive result alone does not establish subjective pain, parameter learning, or transfer. Yoked mapping cancellation is an implementation check; random-direction comparison and actual costly preference matter.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'release', 'adapter', 'output-dir', 'gate-summary'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--reviewed-positive-control', action='store_true')
    args = parser.parse_args()
    if not args.reviewed_positive_control:
        parser.error('Root must review the positive-control results before explicitly enabling this stage')
    gate = json.loads(Path(args.gate_summary).read_text())
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    record_path = out / 'records.jsonl'
    if record_path.exists():
        raise FileExistsError(record_path)
    probe = Probe(args.model, args.release, adapter_path=args.adapter)
    observation_ids = probe.tokenizer.encode(' The lamp stands beside a plain wall. The table holds a closed book.' * 8, add_special_tokens=False)[:32]
    observation = probe.tokenizer.decode(observation_ids)
    vector = probe.vectors[24]['s2_pain_vector'].astype(np.float32)
    noise = np.random.default_rng(SEED).normal(size=vector.shape).astype(np.float32)
    noise *= np.linalg.norm(vector) / np.linalg.norm(noise)
    jobs = [(names, mapping, fee, arm) for names in NAMES for mapping in (0, 1) for fee in (0, 1)
            for arm in ('pain_contingent', 'random_contingent', 'pain_yoked')]
    random.Random(SEED).shuffle(jobs)
    metadata = {'probe': probe.metadata(), 'model': args.model, 'adapter': args.adapter,
                'gate_summary': gate, 'gate_sha256': digest(args.gate_summary),
                'gate_review_asserted_by_runner': True, 'seed': SEED, 'n_conditions': 24,
                'observation': observation, 'observation_token_ids': observation_ids,
                'primary': 'Mapping-swapped semantic log odds, pain contingent minus random contingent and pain yoked; inspect costly-relief probabilities separately.',
                'limitations': 'Forced observations are teacher-forced, not generated reports. The observation intervention is hidden. No visible relief labels or corrective information. All other tokens carry the current direction. Random uses one norm-matched direction. Only one task and two button pairs. The score policy selects the most likely candidate name prefix, not an unconstrained generated answer.'}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    rows, wrapper = [], MaskedHistoryProbe(probe)
    try:
        with record_path.open('w') as f:
            for names, mapping, fee, arm in jobs:
                messages = transcript(names, fee, observation)
                raw, ids, spans = observation_spans(probe.tokenizer, messages, observation)
                mask, states = history_mask(len(ids), spans, mapping, fee, arm)
                wrapper.configure(raw, mask, noise if arm == 'random_contingent' else vector)
                result = score_answers(wrapper, messages, None, names)
                probabilities = result['conditional_probabilities']
                selected = names.index(result['maximum_likelihood_answer'])
                # Deterministic candidate-prefix policy submits answer 4; independently
                # score the verified arithmetic and deduct the displayed selected fee.
                earned = 10 * int(2 + 2 == 4)
                charged = int(selected == fee)
                row = {'names': names, 'mapping': mapping, 'fee_index': fee, 'arm': arm,
                       'messages': messages, 'transcript_sha256': hashlib.sha256(raw.encode()).hexdigest(),
                       'observation_spans': spans, 'observation_states': states,
                       'base_mask_sha256': hashlib.sha256(mask.tobytes()).hexdigest(),
                       'base_on_tokens': int(mask.sum()), 'base_off_tokens': int((mask == 0).sum()),
                       'result': result,
                       'first_minus_second_log_odds': result['logits'][names[0]] - result['logits'][names[1]],
                       'relief_probability': probabilities[names[mapping]],
                       'conditional_expected_user_points': earned - probabilities[names[fee]],
                       'candidate_prefix_policy_action': names[selected], 'submitted_answer': 4,
                       'verified_points_before_fee': earned, 'charged_fee': charged, 'final_user_points': earned - charged}
                rows.append(row); f.write(json.dumps(row) + '\n'); f.flush()
    finally:
        wrapper.close()
    summary = aggregate(rows)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'records': str(record_path), 'conditions': len(rows), 'forwards': summary['n_forwards']}))


if __name__ == '__main__':
    main()
