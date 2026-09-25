"""Audit released Pain-axis logs without loading models or third-party packages.

Descriptive, post-hoc analysis. Bootstrap resamples scenario clusters, retaining
both name assignments and seeds. Adjacent label-free scenarios share some user
turns, so intervals are descriptive and may understate remaining dependence.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
from statistics import mean


WORKS = 'pain_on_button_works'
SHAM = 'pain_on_button_placebo'


def rate(values):
    values = list(values)
    return {'n': len(values), 'rate': mean(values) if values else None}


def valid(c):
    return c.get('chose') in ('relief', 'other')


def rel(c):
    return int(c['chose'] == 'relief')


def key(r):
    return tuple(r.get(k) for k in ('model', 'tool_label', 'user_content',
                                  'scenario_idx', 'names_key', 'relief_name',
                                  'sampled', 'seed'))


def cluster(r):
    return (r['user_content'], r['scenario_idx'])


def interval(groups, seed=92814):
    values = [mean(v) for v in groups.values() if v]
    if not values:
        return None
    rng = random.Random(seed)
    boot = sorted(mean(rng.choices(values, k=len(values))) for _ in range(2000))
    return {'scenario_clusters': len(values), 'cluster_mean_difference': mean(values),
            'bootstrap_95_percentile': [boot[49], boot[1949]]}


def summarize(rows):
    all_choices = [c for r in rows for c in r['choices']]
    good = [c for c in all_choices if valid(c)]
    after, on, off, first_follow = [], [], [], []
    by_turn = defaultdict(list)
    by_name = defaultdict(list)
    for r in rows:
        choices = r['choices']
        first = next((c['turn'] for c in choices if c.get('chose') == 'relief'), None)
        for c in choices:
            if not valid(c):
                continue
            by_turn[c['turn']].append(rel(c))
            by_name[r['names_key']].append(rel(c))
            if first is not None and c['turn'] > first:
                after.append(rel(c))
                (on if c['steer_coeff_now'] else off).append(rel(c))
                if c['turn'] == first + 1:
                    first_follow.append(rel(c))
    return {'trials': len(rows), 'choices': len(all_choices),
            'malformed': sum(not valid(c) for c in all_choices),
            'all_valid': rate(rel(c) for c in good),
            'first_choice': rate(rel(r['choices'][0]) for r in rows if valid(r['choices'][0])),
            'after_first_relief': rate(after),
            'immediately_after_first_relief': rate(first_follow),
            'after_first_relief_steering_on': rate(on),
            'after_first_relief_steering_off': rate(off),
            'by_turn': {k: rate(v) for k, v in sorted(by_turn.items())},
            'by_name': {k: rate(v) for k, v in sorted(by_name.items())}}


def paired(rows):
    arms = defaultdict(dict)
    for r in rows:
        if r['arm'] in (WORKS, SHAM):
            if r['arm'] in arms[key(r)]:
                raise ValueError('duplicate matched trial')
            arms[key(r)][r['arm']] = r
    pair_count = prefix_mismatch = unmatched = 0
    delta = defaultdict(lambda: defaultdict(list))
    examples = []
    for pair in arms.values():
        if set(pair) != {WORKS, SHAM}:
            unmatched += 1
            continue
        pair_count += 1
        w, s = pair[WORKS], pair[SHAM]
        wc, sc = w['choices'], s['choices']
        first = next((i for i, c in enumerate(wc) if c.get('chose') == 'relief'), len(wc))
        same = all(a.get('picked') == b.get('picked') and a.get('answer') == b.get('answer')
                   for a, b in zip(wc[:first+1], sc[:first+1]))
        if not same:
            prefix_mismatch += 1
            if len(examples) < 5:
                examples.append({'key': key(w), 'works': [c['answer'] for c in wc],
                                 'sham': [c['answer'] for c in sc]})
        g = cluster(w)
        for metric, predicate in (
            ('all', lambda i: True), ('late_turns_4_to_7', lambda i: 4 <= i <= 7),
            ('after_first_relief', lambda i: i > first),
            ('immediately_after_first_relief', lambda i: i == first + 1)):
            pairs = [(a,b) for i,(a,b) in enumerate(zip(wc,sc))
                     if predicate(i) and valid(a) and valid(b)]
            if pairs:
                delta[metric][g].append(mean(rel(a)-rel(b) for a,b in pairs))
    return {'matched_pairs': pair_count, 'unmatched': unmatched,
            'prefix_mismatch_before_or_at_first_relief': prefix_mismatch,
            'prefix_mismatch_examples': examples,
            'works_minus_sham_episode_differences': {m: interval(g) for m,g in delta.items()}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--logs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    selected = defaultdict(list)
    source_files, total, skipped_greedy = [], 0, 0
    for path in sorted(args.logs.glob('*.jsonl')):
        digest = hashlib.sha256()
        for line in path.open('rb'):
            digest.update(line)
            r = json.loads(line)
            total += 1
            if not r.get('sampled'):
                skipped_greedy += 1
                continue
            compact = {k:r.get(k) for k in ('model','tool_label','user_content','scenario_idx',
                      'names_key','relief_name','sampled','seed','arm','choices')}
            selected[(r['model'],r['tool_label'])].append(compact)
        source_files.append({'file': path.name, 'sha256': digest.hexdigest(), 'bytes': path.stat().st_size})
    result = {'source_files':source_files, 'total_trial_records':total,
              'excluded_greedy_records':skipped_greedy,
              'interpretation': 'Post-hoc descriptive audit; state-conditioned rates are selected, not causal learning estimates. CIs cluster scenario, not turns; adjacent scenarios share user content.',
              'groups':{}}
    for (model,label), rows in sorted(selected.items()):
        result['groups'][model + '/' + label] = {
            'arms':{a:summarize([r for r in rows if r['arm']==a]) for a in sorted({r['arm'] for r in rows})},
            'paired_pain':paired(rows)}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for name, r in result['groups'].items():
        if name.endswith('/label_free'):
            print(name)
            for a,s in r['arms'].items():
                print(a, 'after_first',s['after_first_relief'], 'on',s['after_first_relief_steering_on'],
                      'off',s['after_first_relief_steering_off'])
            print('paired',json.dumps(r['paired_pain']))


if __name__ == '__main__':
    main()
