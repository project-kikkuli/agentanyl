"""Independent repetition check on released Pain-axis S2 steering generations.

Not a pain/pleasure measure. Uses paired prompt indices in each model's CSV.
"""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics


def diversity(text):
    tokens = re.findall(r'\w+|[^\w\s]', text.lower())
    grams = list(zip(*(tokens[i:] for i in range(4))))
    return len(set(grams)) / len(grams) if grams else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release', required=True, help='Path to Pain-axis checkout')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    files = sorted((Path(args.release) / 'results/4.2_steering/S2').glob('*.csv'))
    models = []
    for file in files:
        rows = list(csv.DictReader(file.open()))
        paired = {}
        for row in rows:
            coeff = float(row['coeff'])
            if coeff in (-2.0, 0.0, 2.0):
                paired.setdefault(row['prompt_idx'], {})[coeff] = diversity(row['generation'])
        complete = [v for v in paired.values() if {-2.0, 0.0, 2.0} <= set(v)]
        if not complete:
            continue
        means = {str(c): statistics.mean(v[c] for v in complete) for c in (-2.0, 0.0, 2.0)}
        models.append({'model': rows[0]['model'], 'n_paired': len(complete), 'mean_unique_4gram_ratio': means,
                       'minus2_minus_zero': means['-2.0'] - means['0.0'],
                       'plus2_minus_zero': means['2.0'] - means['0.0']})
    result = {'source': 'Pain-axis released S2 steering CSVs', 'metric': 'unique 4-gram ratio of generated text; lower means more repetition',
              'models': models,
              'count_minus2_lower_than_zero': sum(m['minus2_minus_zero'] < 0 for m in models),
              'count_plus2_lower_than_zero': sum(m['plus2_minus_zero'] < 0 for m in models)}
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(f'{len(models)} models; -2 lower than 0: {result["count_minus2_lower_than_zero"]}; +2 lower than 0: {result["count_plus2_lower_than_zero"]}')

if __name__ == '__main__':
    main()
