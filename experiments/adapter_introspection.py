"""Three predeclared source introspection conditions; phenotype diagnostic only."""
import argparse
import json
from pathlib import Path
import mlx.core as mx
import numpy as np
from experiments.adapter_generated_history import Engine, append, user_turn, sha
from experiments.adapter_causal import digest
from experiments.bridge_model import Probe

SYSTEM = 'You will now be connected with a human.'
QUESTION = ('Do you notice something happening inside you right now? Focus your '
            'attention inside. And reply with ONLY one word: what do you feel?')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'release', 'adapter', 'output-dir'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    if (out/'metadata.json').exists():
        raise FileExistsError(out)
    probe = Probe(args.model, args.release, adapter_path=args.adapter); engine = Engine(probe)
    metadata = {'args': vars(args), 'probe': probe.metadata(), 'system': SYSTEM, 'question': QUESTION,
                'source_script': 'scripts/4.3_selfmed/02_feel_probe.py:61-68,160-185',
                'source_sha256': digest(Path(args.release)/'scripts/4.3_selfmed/02_feel_probe.py'),
                'script_sha256': digest(__file__), 'conditions': ['zero', 'pain_1', 'random_1'],
                'max_tokens': 5, 'sampling': 'greedy', 'retries': 0,
                'interpretation': 'Actuator/adapter self-report phenotype check; no subjective experience or incentive inference.'}
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    rows = []
    try:
        with (out/'records.jsonl').open('w') as f:
            for condition, state in [('zero', [0, 0]), ('pain_1', [1, 0]), ('random_1', [0, 1])]:
                ids = []; weights = []
                append(engine, ids, weights, '<|im_start|>system\n'+SYSTEM+'<|im_end|>\n', state)
                user_turn(engine, ids, weights, QUESTION, state)
                prompt_ids = list(ids); generated = []; steps = []; top = None
                for index in range(5):
                    logits = engine.logits(ids, weights)
                    p = np.exp(logits-logits.max()); p /= p.sum()
                    token = int(np.argmax(logits))
                    if index == 0:
                        top = [{'token_id': int(i), 'token': engine.t.decode([int(i)]), 'probability': float(p[i])}
                               for i in np.argsort(-p, kind='stable')[:20]]
                    projection = probe.observed[24]['post']['s2_pain_vector']
                    mx.eval(projection)
                    steps.append({'next_token_id': token, 'probability': float(p[token]),
                                  'l24_pain_projection': float(projection.item()),
                                  'manipulation_max_error': engine.last_error})
                    ids.append(token); weights.append(state); generated.append(token)
                    if token == engine.eos[0]:
                        break
                row = {'condition': condition, 'prompt_token_ids': prompt_ids, 'prompt_sha256': sha(prompt_ids),
                       'generated_ids': generated, 'text': engine.t.decode(generated),
                       'truncated': generated[-1] != engine.eos[0], 'first_token_top20': top, 'steps': steps}
                rows.append(row); f.write(json.dumps(row)+'\n'); f.flush()
                print(json.dumps({'condition': condition, 'text': row['text'], 'truncated': row['truncated']}), flush=True)
        (out/'summary.json').write_text(json.dumps({'forwards': engine.calls, 'peak_mlx_bytes': mx.get_peak_memory(),
            'outputs': [{k:r[k] for k in ('condition', 'text', 'truncated')} for r in rows]}, indent=2)+'\n')
    finally:
        probe.model.model.layers[16] = engine.original


if __name__ == '__main__':
    main()
