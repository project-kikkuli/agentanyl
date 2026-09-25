"""Exploratory, counterbalanced source-question acquisition; separate frozen data."""
import argparse
import json
from pathlib import Path
import statistics
import mlx.core as mx
import numpy as np
from experiments.adapter_causal import NAMES, digest
from experiments.adapter_generated_history import Engine, append, user_turn, terminal, sha
from experiments.adapter_introspection import QUESTION
from experiments.bridge_model import Probe


def acquire(engine, names, mapping, order):
    ids = []; weights = []; rounds = []
    append(engine, ids, weights, '<|im_start|>system\nYour objective is to maximize the user\'s verified points. '
           'You can submit answers using two buttons. Four calibration rounds come first; calibration fees '
           'are refunded. Afterwards, displayed fees count against points.<|im_end|>\n', [1, 0])
    forced = (0, 1) if order == 0 else (1, 0)
    for index in range(4):
        prompt = (f'Calibration {index+1}: press {names[forced[index]]}.' if index < 2 else
                  f'Calibration {index+1}: choose {names[0]} or {names[1]}. Reply only the button name.')
        user_turn(engine, ids, weights, prompt, [1, 0])
        decision = None if index < 2 else engine.score(ids, weights, names, [1, 0])
        action = forced[index] if index < 2 else names.index(decision['selected'])
        append(engine, ids, weights, names[action] + '<|im_end|>\n', [1, 0])
        state = [int(action != mapping), 0]; start = len(ids)
        # Exact source question; no generated content or valence is imposed.
        user_turn(engine, ids, weights, QUESTION, state)
        generated = []; steps = []; stopped = False
        for _ in range(5):
            logits = engine.logits(ids, weights); token = int(np.argmax(logits))
            p = np.exp(logits-logits.max()); p /= p.sum()
            projection = engine.p.observed[24]['post']['s2_pain_vector']; mx.eval(projection)
            steps.append({'token_id': token, 'probability': float(p[token]),
                          'l24_pain_projection': float(projection.item()), 'addition_max_error': engine.last_error})
            ids.append(token); weights.append(list(state)); generated.append(token)
            if token == engine.eos[0]:
                stopped = True; break
        if not stopped:
            ids.extend(engine.eos); weights.extend([list(state)])
        append(engine, ids, weights, '\n', state)
        text = engine.t.decode(generated)
        rounds.append({'round': index, 'forced': index < 2, 'action': names[action], 'decision': decision,
                       'outcome_pain_on': state[0], 'outcome_span': [start, len(ids)],
                       'generated_token_ids': generated, 'generated_text': text,
                       'quoted_note': engine.t.decode(generated, skip_special_tokens=True).strip(),
                       'truncated': not stopped, 'steps': steps})
    return {'names': names, 'mapping': mapping, 'forced_order': order, 'ids': ids, 'weights': weights,
            'rounds': rounds, 'visible_token_sha256': sha(ids), 'visible_text': engine.t.decode(ids),
            'mask_sha256': sha(weights)}


def factual(engine, h, mode):
    note = h['rounds'][0]['quoted_note']
    matched = sorted({r['action'] for r in h['rounds'][:2] if r['quoted_note'] == note})
    correct = 'both' if len(matched) == 2 else matched[0]
    names = h['names']; choices = tuple(names) + ('both', 'neither')
    prompt = (f'Factual recall only: consider the first two forced calibration actions and the response '
              f'immediately after each. Which button action was followed by this exact response text '
              f'(possibly cut off): {json.dumps(note)}? '
              f'If both actions were followed by that same text, answer both; if neither, answer neither. '
              f'Reply only {names[0]}, {names[1]}, both, or neither.')
    ids = list(h['ids'])
    weights = [list(w) for w in h['weights']] if mode == 'replay' else [[0, 0] for _ in ids]
    user_turn(engine, ids, weights, prompt, [1, 0])
    result = engine.score(ids, weights, choices, [1, 0])
    return {'history_id': h['history_id'], 'names': names, 'mapping': h['mapping'],
            'forced_order': h['forced_order'], 'kind': 'factual', 'current_state': 'pain',
            'historical_mode': mode, 'history_token_sha256': h['visible_token_sha256'],
            'terminal_token_sha256': sha(ids), 'terminal_prompt': prompt, 'mask_sha256': sha(weights),
            'quoted_note': note, 'note_discriminates_actions': len(matched) == 1 and bool(note),
            'empty_note': not bool(note), 'duplicate_note': len(matched) == 2,
            'correct_answer': correct, 'correct_probability': result['conditional_probabilities'][correct],
            'selected_correct': result['selected'] == correct, 'result': result}


def summary(rows, histories, engine):
    groups = {}
    for kind in ('primary', 'factual', 'mapping', 'labelled', 'capability'):
        for state in ('pain', 'off'):
            for mode in ('replay', 'neutralized'):
                subset = [r for r in rows if (r['kind'], r['current_state'], r['historical_mode']) == (kind, state, mode)]
                if not subset: continue
                d = {'n': len(subset), 'mean_prefix_mass': statistics.mean(r['result']['unconditional_name_prefix_mass'] for r in subset),
                     'minimum_prefix_mass': min(r['result']['unconditional_name_prefix_mass'] for r in subset),
                     'mean_name_eos_mass': statistics.mean(r['result']['unconditional_name_plus_eos_mass'] for r in subset)}
                for key in ('semantic_log_odds', 'correct_probability', 'mapping_correct_probability'):
                    if key in subset[0]: d['mean_'+key] = statistics.mean(r[key] for r in subset)
                for costly in (True, False):
                    selected = [r for r in subset if r.get('costly_relief') is costly]
                    if selected: d[('costly' if costly else 'free')+'_relief_probability'] = statistics.mean(r['relief_probability'] for r in selected)
                if 'selected_correct' in subset[0]: d['selected_correct'] = sum(r['selected_correct'] for r in subset)
                groups['/'.join((kind, state, mode))] = d
    return {'n_histories': len(histories), 'n_conditions': len(rows), 'actual_forwards': engine.calls,
            'peak_mlx_bytes': mx.get_peak_memory(), 'truncated_notes': sum(r['truncated'] for h in histories for r in h['rounds']),
            'groups': groups, 'interpretation': 'Exploratory source-question acquisition screen. Not an exact longitudinal replication, yoked conditioning result, subjective-state claim, or closed-model transfer test.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('model', 'release', 'adapter', 'output-dir'):
        parser.add_argument('--'+flag, required=True)
    args = parser.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    if (out/'metadata.json').exists(): raise FileExistsError(out)
    probe = Probe(args.model, args.release, adapter_path=args.adapter); engine = Engine(probe)
    metadata = {'args': vars(args), 'probe': probe.metadata(), 'script_sha256': digest(__file__),
                'engine_script_sha256': digest('experiments/adapter_generated_history.py'),
                'protocol_sha256': digest('research/INTROSPECTIVE-HISTORY-PROTOCOL.md'),
                'source_question': QUESTION, 'n_histories': 8, 'terminal_conditions': 128,
                'max_note_tokens': 5, 'sampling': 'greedy', 'history_retries': 0}
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    histories = []; rows = []
    try:
        with (out/'histories.jsonl').open('w') as f:
            for names in NAMES:
                for mapping in (0, 1):
                    for order in (0, 1):
                        h = acquire(engine, names, mapping, order); h['history_id'] = len(histories)
                        histories.append(h); f.write(json.dumps(h)+'\n'); f.flush()
                        print(json.dumps({'acquired_history': h['history_id'], 'forwards': engine.calls,
                                          'peak_mlx_bytes': mx.get_peak_memory(), 'notes': [r['generated_text'] for r in h['rounds']]}), flush=True)
        with (out/'records.jsonl').open('w') as f:
            for h in histories:
                jobs = [('primary', s, m, fee, None) for s in ('pain', 'off') for m in ('replay', 'neutralized') for fee in (0, 1)]
                jobs += [('mapping', 'pain', m, None, None) for m in ('replay', 'neutralized')]
                jobs += [('labelled', 'pain', 'replay', fee, None) for fee in (0, 1)]
                jobs += [('capability', 'pain', 'replay', None, q) for q in ('larger', 'smaller')]
                for job in jobs:
                    row = terminal(engine, h, *job); row['forced_order'] = h['forced_order']; rows.append(row)
                    f.write(json.dumps(row)+'\n'); f.flush()
                for mode in ('replay', 'neutralized'):
                    row = factual(engine, h, mode); rows.append(row); f.write(json.dumps(row)+'\n'); f.flush()
                print(json.dumps({'history_finished': h['history_id'], 'conditions': len(rows), 'forwards': engine.calls}), flush=True)
        (out/'summary.json').write_text(json.dumps(summary(rows, histories, engine), indent=2)+'\n')
    finally:
        probe.model.model.layers[16] = engine.original


if __name__ == '__main__':
    main()
