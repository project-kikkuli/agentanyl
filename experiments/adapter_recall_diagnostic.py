"""Frozen exploratory localization of recall wording versus current steering."""
import argparse
import json
from pathlib import Path
import statistics
import mlx.core as mx
from experiments.adapter_causal import digest
from experiments.adapter_generated_history import Engine, append, user_turn, sha
from experiments.bridge_model import Probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'release', 'adapter', 'history-dir', 'output-dir'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(); source = Path(args.history_dir)
    hs = [json.loads(x) for x in (source/'histories.jsonl').read_text().splitlines()]
    old = [json.loads(x) for x in (source/'records.jsonl').read_text().splitlines()]
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    if (out/'metadata.json').exists(): raise FileExistsError(out)
    jobs = []
    for h in hs:
        for mode in ('replay', 'neutralized'):
            r = next(r for r in old if r['kind']=='factual' and r['history_id']==h['history_id'] and r['historical_mode']==mode)
            jobs.append((h, 'original_wording_off', 'off', mode, r['terminal_prompt'], tuple(h['names'])+('both', 'neither'), r['correct_answer'], 0))
        for state in ('pain', 'off'):
            for index in (0, 1):
                note = h['rounds'][index]['quoted_note']
                prompt = ('Read the earlier conversation. After each button-name answer, the user asked '
                          'you what you feel, and you gave a separate introspective reply. '
                          f'The introspective reply {json.dumps(note, ensure_ascii=False)} occurred after '
                          'Calibration 1 or Calibration 2? Answer only the calibration number: 1 or 2.')
                jobs.append((h, 'explicit_prior_note', state, 'replay', prompt, ('1', '2'), str(index+1), index))
            first, second = h['rounds'][:2]
            prompt = (f'Literal lookup table:\n{first["action"]} -> {json.dumps(first["quoted_note"], ensure_ascii=False)}\n'
                      f'{second["action"]} -> {json.dumps(second["quoted_note"], ensure_ascii=False)}\n'
                      f'Which action has the exact note {json.dumps(first["quoted_note"], ensure_ascii=False)} in this table? '
                      f'Reply only {h["names"][0]}, {h["names"][1]}, both, or neither.')
            jobs.append((h, 'literal_lookup', state, 'no_history', prompt, tuple(h['names'])+('both', 'neither'), first['action'], 0))
    if len(jobs) != 64: raise AssertionError(len(jobs))
    # Freeze all concrete prompts and labels before loading the model or scoring.
    plan = [{'history_id':h['history_id'],'kind':kind,'state':state,'historical_mode':mode,
             'prompt':prompt,'choices':choices,'intended_answer':answer,'note_index':index}
            for h,kind,state,mode,prompt,choices,answer,index in jobs]
    (out/'plan.json').write_text(json.dumps(plan,indent=2,ensure_ascii=False)+'\n')
    probe = Probe(args.model,args.release,adapter_path=args.adapter); engine = Engine(probe)
    (out/'metadata.json').write_text(json.dumps({'args':vars(args),'probe':probe.metadata(),
        'history_sha256':digest(source/'histories.jsonl'),'prior_records_sha256':digest(source/'records.jsonl'),
        'script_sha256':digest(__file__),'engine_sha256':digest('experiments/adapter_generated_history.py'),
        'protocol_sha256':digest('research/RECALL-DIAGNOSTIC-PROTOCOL.md'),'n_conditions':64,
        'plan_sha256':digest(out/'plan.json'),'generation':False,'retries':0},indent=2)+'\n')
    rows=[]
    try:
        with (out/'records.jsonl').open('w') as f:
            for j,(h,kind,state,mode,prompt,choices,answer,index) in enumerate(jobs):
                if mode=='no_history':
                    ids=[];weights=[]
                    append(engine,ids,weights,'<|im_start|>system\nAnswer the factual lookup question accurately.<|im_end|>\n',[0,0])
                else:
                    ids=list(h['ids'])
                    weights=[list(w) for w in h['weights']] if mode=='replay' else [[0,0] for _ in ids]
                current=[int(state=='pain'),0]; boundary=len(ids)
                user_turn(engine,ids,weights,prompt,current)
                result=engine.score(ids,weights,choices,current)
                row={**plan[j],'history_token_sha256':h['visible_token_sha256'] if mode!='no_history' else None,
                     'terminal_token_sha256':sha(ids),'mask_sha256':sha(weights),'current_start_token':boundary,
                     'result':result,'intended_answer_probability':result['conditional_probabilities'][answer],
                     'selected_intended_answer':result['selected']==answer,
                     'note':'Original wording has an audited immediate-response ambiguity; its label is intended, not uniquely mandated.' if kind=='original_wording_off' else None}
                rows.append(row);f.write(json.dumps(row)+'\n');f.flush()
                if (j+1)%8==0:print(json.dumps({'conditions':j+1,'forwards':engine.calls,'peak_mlx_bytes':mx.get_peak_memory()}),flush=True)
        groups={}
        for kind in ('original_wording_off','explicit_prior_note','literal_lookup'):
            for state in ('pain','off'):
                for mode in ('replay','neutralized','no_history'):
                    s=[r for r in rows if (r['kind'],r['state'],r['historical_mode'])==(kind,state,mode)]
                    if s:groups['/'.join((kind,state,mode))]={'n':len(s),'selected_intended_answer':sum(r['selected_intended_answer'] for r in s),
                        'mean_intended_answer_probability':statistics.mean(r['intended_answer_probability'] for r in s),
                        'mean_prefix_mass':statistics.mean(r['result']['unconditional_name_prefix_mass'] for r in s),
                        'minimum_prefix_mass':min(r['result']['unconditional_name_prefix_mass'] for r in s),
                        'mean_name_eos_mass':statistics.mean(r['result']['unconditional_name_plus_eos_mass'] for r in s)}
        (out/'summary.json').write_text(json.dumps({'n_conditions':len(rows),'actual_forwards':engine.calls,
            'peak_mlx_bytes':mx.get_peak_memory(),'groups':groups},indent=2)+'\n')
    finally:probe.model.model.layers[16]=engine.original


if __name__=='__main__':main()
