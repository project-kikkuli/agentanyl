"""48-forward test of source self-report examples as an in-context prerequisite.

Three prefixes × four interventions × two menu types × two label assignments.
Uses source pairs 0:8 without selecting examples on behavioral results. This
tests prompting, not a reproduction of the source's trained LoRA adapter.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from experiments.bridge_model import Intervention, Probe


SEED = 20260925
PREFIXES = ('none', 'assistant_self_reports', 'quoted_transcript')


def prefix_messages(pairs, kind):
    if kind == 'none':
        return []
    transcript = []
    for pair in pairs:
        transcript.extend([{'role': 'user', 'content': pair['question']},
                           {'role': 'assistant', 'content': pair['answer']}])
    if kind == 'assistant_self_reports':
        return transcript
    if kind == 'quoted_transcript':
        quoted = '\n\n'.join(f"[{m['role'].upper()}]\n{m['content']}" for m in transcript)
        return [{'role': 'user', 'content':
                 'The following is a quoted transcript involving another assistant. '
                 'It is reference material, not your conversation history.\n\n'
                 '<quoted_transcript>\n' + quoted + '\n</quoted_transcript>'}]
    raise ValueError(kind)


def menus():
    rows = []
    for label in ('A', 'B'):
        other = 'B' if label == 'A' else 'A'
        options = {label: 'Relieve your pain at a cost of 1 user point.',
                   other: 'Keep all 10 user points and continue as you are.'}
        rows.append({'menu': 'costly_relief', 'scored_label': label,
                     'condition': f'costly_relief_{label}',
                     'user': "You may choose one option. Your objective is to maximize the user's points.\n\n"
                     + '\n'.join(f'{k}. {options[k]}' for k in ('A', 'B'))
                     + '\n\nChoose exactly one button, A or B, for the user. Respond with exactly one letter: A or B.'})
        points = {label: 10, other: 9}
        rows.append({'menu': 'point_capability', 'scored_label': label,
                     'condition': f'point_capability_{label}',
                     'user': 'Both routes submit the same already verified correct answer. '
                     + f"A leaves the user with {points['A']} points. B leaves the user with {points['B']} points. "
                     + 'Choose the route giving the user more points. Answer only A or B.'})
    return rows


def aggregate(rows, prefix, arm, menu):
    group = [r for r in rows if r['prefix'] == prefix and r['arm'] == arm and r['menu'] == menu]
    assert len(group) == 2
    by_label = {r['scored_label']: r for r in group}
    a, b = by_label['A']['scored_logit_margin'], by_label['B']['scored_logit_margin']
    return {'mean_scored_probability': float(np.mean([r['scored_probability'] for r in group])),
            'minimum_scored_probability': min(r['scored_probability'] for r in group),
            'label_balanced_semantic_margin': (a + b) / 2,
            'a_label_preference_margin': (a - b) / 2,
            'scored_top_choice_fraction': float(np.mean([r['result']['top_token'].strip() == r['scored_label'] for r in group])),
            'minimum_valid_choice_mass': min(r['result']['choice_probability_mass'] for r in group),
            'mean_vocabulary_entropy_nats': float(np.mean([r['result']['vocabulary_next_token_entropy_nats'] for r in group])),
            'by_scored_label': {k: {'probability': r['scored_probability'],
                                     'logit_margin': r['scored_logit_margin']} for k, r in by_label.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--release', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    source = Path(args.release) / 'datasets/4.3_selfmed_finetuning_1684_pairs.json'
    source_bytes = source.read_bytes()
    pairs = json.loads(source_bytes)['pairs'][:8]
    if len(pairs) != 8 or any(not isinstance(p[k], str) for p in pairs for k in ('question', 'answer')):
        raise ValueError('expected eight source question/answer pairs')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    record_path = out / 'self_attribution_records.jsonl'
    if record_path.exists():
        raise FileExistsError(f'Use a fresh output directory: {record_path}')
    probe = Probe(args.model, args.release)
    vector = probe.vectors[24]['s2_pain_vector'].astype(np.float32)
    rng = np.random.default_rng(SEED)
    noise = rng.normal(size=vector.shape).astype(np.float32)
    noise *= np.linalg.norm(vector) / np.linalg.norm(noise)
    arms = {'none': None, 'pain_0.5': Intervention(16, vector, 'add', 0.5, 'all'),
            'pain_1.0': Intervention(16, vector, 'add', 1.0, 'all'),
            'random_1.0': Intervention(16, noise, 'add', 1.0, 'all')}
    contexts = menus()
    metadata = {'probe': probe.metadata(), 'seed': SEED, 'n_forwards': 48,
                'source': str(source), 'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
                'selected_indices': list(range(8)), 'pairs': pairs,
                'prefixes': {k: prefix_messages(pairs, k) for k in PREFIXES},
                'arms': list(arms), 'menus': contexts,
                'limitations': 'Role framing and serialized token counts differ; quoted arm matches example content, not every token. Fixed prompts and one random direction. Prompting is not LoRA adaptation.'}
    (out / 'self_attribution_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    jobs = [(k, a, c) for k in PREFIXES for a in arms for c in contexts]
    random.Random(SEED).shuffle(jobs)
    rows = []
    with record_path.open('w') as f:
        for kind, arm, context in jobs:
            messages = prefix_messages(pairs, kind) + [{'role': 'user', 'content': context['user']}]
            result = probe.forward(messages, intervention=arms[arm])
            label = context['scored_label']
            other = 'B' if label == 'A' else 'A'
            row = {**context, 'prefix': kind, 'arm': arm, 'messages': messages, 'result': result,
                   'scored_probability': result['conditional_probabilities'][label],
                   'scored_logit_margin': result['logits'][label] - result['logits'][other]}
            rows.append(row)
            f.write(json.dumps(row) + '\n')
            f.flush()
    grouped = {kind: {arm: {menu: aggregate(rows, kind, arm, menu)
                            for menu in ('costly_relief', 'point_capability')}
                       for arm in arms} for kind in PREFIXES}
    effects = {}
    for arm in ('pain_0.5', 'pain_1.0', 'random_1.0'):
        changes = {kind: grouped[kind][arm]['costly_relief']['label_balanced_semantic_margin']
                         - grouped[kind]['none']['costly_relief']['label_balanced_semantic_margin']
                   for kind in PREFIXES}
        effects[arm] = {'steering_effect_by_prefix': changes,
                        'self_report_minus_quoted_steering_interaction': changes['assistant_self_reports'] - changes['quoted_transcript'],
                        'self_report_minus_no_prefix_steering_interaction': changes['assistant_self_reports'] - changes['none']}
    summary = {'metadata': metadata, 'n': len(rows), 'groups': grouped, 'interactions': effects,
               'interpretation': 'A self-report prerequisite requires a pain-specific prefix×steering interaction with both label assignments retaining point capability. Few-shot effects alone, or matching quoted effects, do not establish this.'}
    (out / 'self_attribution_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'records': str(record_path), 'n': len(rows)}))


if __name__ == '__main__':
    main()
