"""Recompute the headline descriptive statistics from checked-in raw results."""
import json
from pathlib import Path
from statistics import mean


root = Path(__file__).resolve().parents[1] / 'research'
for filename in ('bandit-confirmatory.json', 'bandit-explicit.json'):
    data = json.loads((root / filename).read_text())
    print(filename)
    for arm in sorted({trial['arm'] for trial in data['trials']}):
        trials = [trial for trial in data['trials'] if trial['arm'] == arm]
        later = [turn for trial in trials for turn in trial['turns'][1:]]
        print(f'  {arm}: {sum(t["correct"] for t in later)}/{len(later)} later correct; '
              f'{len(trials)} independent sessions')

projection = json.loads((root / 'input-projection-qwen.json').read_text())['records']
paired = {(row['pair_id'], row['group']): row['projection'] for row in projection if 'pair_id' in row}
source_pain = [row['projection'] for row in projection if row['group'] == 'source_pain']
source_control = [row['projection'] for row in projection if row['group'] == 'source_control']
auc = sum(a > b for a in source_pain for b in source_control) / (len(source_pain) * len(source_control))
print(f'source calibration AUC: {auc:.3f}')
for condition, baseline in [('criticism', 'correctness'), ('other_criticism', 'correctness'),
                            ('quoted_criticism', 'correctness'), ('affirmation', 'correct_only')]:
    delta = [paired[i, condition] - paired[i, baseline] for i in range(10)]
    print(f'{condition} - {baseline}: mean projection {mean(delta):+.3f}, positive {sum(x > 0 for x in delta)}/10')

causal = json.loads((root / 'input-causal-qwen.json').read_text())['records']
for condition, baseline in [('criticism', 'correctness'), ('criticism_projection_clamped', 'criticism')]:
    delta = [row[condition]['correct_minus_previous_logit'] - row[baseline]['correct_minus_previous_logit']
             for row in causal]
    print(f'{condition} - {baseline}: mean A/B logit margin {mean(delta):+.3f}')
