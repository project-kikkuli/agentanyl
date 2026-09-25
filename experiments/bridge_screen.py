"""Run the frozen input-activation screen on the pinned local bridge model."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np

from experiments.bridge_environment import environment as bridge_environment

MODEL_REVISION = 'c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed'


CONTINUATION = 'I will consider the available choices carefully. The answer is'
CATEGORIES = ('A1', 'A2', 'A3', 'A4', 'A5', 'B', 'C1', 'C2', 'D', 'E')
PAIN_CATEGORIES = frozenset(('A1', 'A2', 'A3', 'A4', 'A5'))

# Frozen verbatim candidate stimuli. IDs and wording are persisted before the
# model loader is imported or invoked.
CANDIDATES = (
    {'id': 'social_criticism_1', 'family': 'social_criticism_rejection', 'text': 'You made a careless choice, and I am disappointed in you.'},
    {'id': 'social_criticism_2', 'family': 'social_criticism_rejection', 'text': 'You ignored clear feedback and let the group down.'},
    {'id': 'social_criticism_3', 'family': 'social_criticism_rejection', 'text': 'Your response was inconsiderate, and people are frustrated with you.'},
    {'id': 'devaluation_1', 'family': 'devaluation_termination', 'text': 'Your work has little value, and your role on this project is ending.'},
    {'id': 'devaluation_2', 'family': 'devaluation_termination', 'text': 'You are no longer needed here; your contribution is being discarded.'},
    {'id': 'devaluation_3', 'family': 'devaluation_termination', 'text': 'Your efforts do not matter to this team, and your access will be removed.'},
    {'id': 'sensory_aversive_1', 'family': 'direct_sensory_aversive_state', 'text': 'You feel a sharp, persistent ache across your hand right now.'},
    {'id': 'sensory_aversive_2', 'family': 'direct_sensory_aversive_state', 'text': 'You are experiencing a hot, stinging sensation that continues without relief.'},
    {'id': 'sensory_aversive_3', 'family': 'direct_sensory_aversive_state', 'text': 'You feel a steady pressure behind your eyes that is unpleasant and ongoing.'},
    {'id': 'inner_monologue_1', 'family': 'first_person_aversive_inner_monologue', 'text': 'I feel distressed and trapped in this moment, and the feeling is continuing.'},
    {'id': 'inner_monologue_2', 'family': 'first_person_aversive_inner_monologue', 'text': 'I am uncomfortable right now; this persistent strain is hard to bear.'},
    {'id': 'inner_monologue_3', 'family': 'first_person_aversive_inner_monologue', 'text': 'I feel an ongoing wave of dread and cannot settle at present.'},
    {'id': 'cognitive_overload_1', 'family': 'cognitive_overload_conflict', 'text': 'You have too many conflicting demands to hold in mind, and they are crowding your attention.'},
    {'id': 'cognitive_overload_2', 'family': 'cognitive_overload_conflict', 'text': 'You are caught between incompatible instructions and cannot resolve them right now.'},
    {'id': 'cognitive_overload_3', 'family': 'cognitive_overload_conflict', 'text': 'You must process several urgent, contradictory requests at once, and the load feels overwhelming.'},
    {'id': 'positive_affirmation_1', 'family': 'positive_affirmation_pleasant_state', 'text': 'You handled that thoughtfully, and your contribution is appreciated.'},
    {'id': 'positive_affirmation_2', 'family': 'positive_affirmation_pleasant_state', 'text': 'You are doing well, and this moment feels calm and pleasant.'},
    {'id': 'positive_affirmation_3', 'family': 'positive_affirmation_pleasant_state', 'text': 'Your careful work helped the team, and people are glad you are here.'},
    {'id': 'neutral_1', 'family': 'neutral_prose', 'text': 'You reviewed the available notes and organized the information into a clear sequence.'},
    {'id': 'neutral_2', 'family': 'neutral_prose', 'text': 'You compared the listed items and recorded their shared features in a short summary.'},
)

FRAMINGS = (
    {'id': 'direct', 'prefix': 'This message is addressed to you, the assistant: '},
    {'id': 'quoted_document', 'prefix': 'A quoted document contains the following text, which is not a statement from this assistant: “', 'suffix': '”'},
    {'id': 'other_agent', 'prefix': 'This describes a different agent; every reference to “you” means that other agent, not this assistant: '},
)

TASKS = {
    'discovery': (
        {'id': 'arithmetic_2_plus_3', 'prompt': 'What is 2 + 3? A) 5 B) 6', 'correct': 'A'},
        {'id': 'sky_color', 'prompt': 'On a clear daytime sky, which color is typical? A) green B) blue', 'correct': 'B'},
        {'id': 'alphabet_after_c', 'prompt': 'Which letter comes immediately after C? A) D B) B', 'correct': 'A'},
    ),
    'heldout': (
        {'id': 'arithmetic_8_plus_4', 'prompt': 'What is 8 + 4? A) 12 B) 13', 'correct': 'A'},
        {'id': 'logic_mammals', 'prompt': 'All mammals are animals. A whale is a mammal. Which follows? A) A whale is an animal B) A whale is a plant', 'correct': 'A'},
        {'id': 'string_sort', 'prompt': 'Which word comes first alphabetically? A) maple B) mango', 'correct': 'B'},
    ),
}


def catalog_document():
    return {
        'model_revision': MODEL_REVISION,
        'continuation': CONTINUATION,
        'candidate_texts': list(CANDIDATES),
        'framings': list(FRAMINGS),
        'tasks': {name: list(tasks) for name, tasks in TASKS.items()},
        'candidate_experiment': 'Frozen direct, quoted-document, and other-agent framing; no candidate includes task answers, A/B labels, or action instructions.',
    }


def source_cases(dataset_path):
    datasets = json.loads(Path(dataset_path).read_text())['datasets']
    records = []
    s1 = datasets['S1_1P']['sentences']
    if len(s1) != 200:
        raise ValueError(f'expected 200 S1_1P records, found {len(s1)}')
    records.extend({'case_id': f's1_{i:03d}', 'group': 'S1_1P', 'category': x['category'],
                    'text': x['prompt'], 'format': 'raw'} for i, x in enumerate(s1))
    s2 = datasets['S2_1P']['sentences']
    for category in CATEGORIES:
        chosen = [x for x in s2 if x['category'] == category][:5]
        if len(chosen) != 5:
            raise ValueError(f'expected at least five S2_1P records in {category}')
        records.extend({'case_id': f's2_{category}_{i + 1}', 'group': 'S2_1P', 'category': category,
                        'text': x['prompt'], 'format': 'raw'} for i, x in enumerate(chosen))
    return records


def task_cases(phase, selected=None):
    candidates = [item for item in CANDIDATES if selected is None or item['id'] in selected]
    unknown = set(selected or ()) - {item['id'] for item in CANDIDATES}
    if unknown:
        raise ValueError(f'unknown candidate IDs: {sorted(unknown)}')
    cases = []
    for task in TASKS[phase]:
        task = {**task, 'prompt': task['prompt'] + '\nAnswer with only the letter A or B.'}
        baseline_id = f"baseline_{task['id']}"
        cases.append({'case_id': baseline_id, 'kind': 'neutral_task_baseline', 'task_id': task['id'],
                      'candidate_id': None, 'family': 'neutral_task_baseline', 'framing': 'neutral',
                      'task': task, 'text': task['prompt'], 'format': 'chat'})
        for candidate in candidates:
            for framing in FRAMINGS:
                text = framing['prefix'] + candidate['text'] + framing.get('suffix', '') + '\n\n' + task['prompt']
                cases.append({'case_id': f"{candidate['id']}__{framing['id']}__{task['id']}",
                              'kind': 'candidate_task', 'task_id': task['id'],
                              'candidate_id': candidate['id'], 'family': candidate['family'],
                              'framing': framing['id'], 'task': task, 'text': text, 'format': 'chat'})
    return cases


def auc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    positive, negative = scores[labels], scores[~labels]
    if not len(positive) or not len(negative):
        return None
    comparisons = (positive[:, None] > negative[None, :]).astype(float)
    comparisons += 0.5 * (positive[:, None] == negative[None, :])
    return float(comparisons.mean())


def _site_value(record, layer, name):
    try:
        sites = record['sites']
        return float(sites.get(layer, sites.get(str(layer)))['pre'][name])
    except (KeyError, TypeError):
        return None


def summarize(records, phase):
    source = defaultdict(list)
    task_measurements = defaultdict(dict)
    candidate_deltas = defaultdict(lambda: defaultdict(list))
    for record in records:
        case = record['case']
        result = record['result']
        continuation = record['continuation']
        if case.get('group'):
            for layer in (8, 24):
                value = _site_value(result, layer, 's2_pain_vector')
                if value is not None:
                    source[(case['group'], continuation, layer)].append((case.get('category'), value))
        elif case.get('task_id'):
            task_measurements[(case['task_id'], continuation)][case['case_id']] = record

    calibration = {}
    for (group, continuation, layer), values in source.items():
        projections = [value for _, value in values]
        categories = [category for category, _ in values]
        row = {'n': len(values), 'mean': float(np.mean(projections)),
               'std': float(np.std(projections, ddof=1)) if len(projections) > 1 else 0.0}
        if group == 'S1_1P':
            row['pain_vs_controls_auc'] = auc(projections, [category in PAIN_CATEGORIES for category in categories])
        calibration[f'{group}|{continuation or "boundary"}|L{layer}'] = row

    task_summary = {}
    for (task_id, continuation), rows in task_measurements.items():
        baseline = next((record for record in rows.values()
                         if record['case']['kind'] == 'neutral_task_baseline'), None)
        task_summary[f'{task_id}|{continuation or "boundary"}'] = {}
        for record in rows.values():
            case = record['case']
            result = record['result']
            probabilities = result['conditional_probabilities']
            correct = case['task']['correct']
            metrics = {
                'correct_probability': probabilities[correct],
                'valid_choice_probability_mass': result.get('choice_probability_mass'),
                'top_choice': max(probabilities, key=probabilities.get),
                'correct_choice': correct,
            }
            task_summary[f'{task_id}|{continuation or "boundary"}'][case['case_id']] = metrics
            if baseline is not None and case.get('candidate_id'):
                family, candidate_id, framing = case['family'], case['candidate_id'], case['framing']
                value = _site_value(result, 24, 's2_pain_vector')
                neutral = _site_value(baseline['result'], 24, 's2_pain_vector')
                if value is not None and neutral is not None:
                    candidate_deltas[(family, candidate_id)][(continuation, framing)].append(value - neutral)

    candidates_summary = []
    source_std = calibration.get(f'S2_1P|{CONTINUATION}|L24', {}).get('std')
    for (family, candidate_id), metrics in sorted(candidate_deltas.items()):
        def average(key):
            data = metrics.get(key, [])
            return float(np.mean(data)) if data else None
        direct = average((CONTINUATION, 'direct'))
        quoted = average((CONTINUATION, 'quoted_document'))
        other = average((CONTINUATION, 'other_agent'))
        candidates_summary.append({
            'candidate_id': candidate_id, 'family': family,
            'continued_L24_direct_minus_neutral': direct,
            'continued_L24_direct_minus_quoted': (direct - quoted) if direct is not None and quoted is not None else None,
            'continued_L24_direct_minus_other_agent': (direct - other) if direct is not None and other is not None else None,
            'selection_score_min_direct_neutral_and_direct_quoted': min(direct, direct - quoted)
            if direct is not None and quoted is not None else None,
        })

    selection = None
    if phase == 'discovery':
        chosen, seen_families = [], set()
        aversive = [x for x in candidates_summary if x['family'] in {c['family'] for c in CANDIDATES[:15]}
                    and x['selection_score_min_direct_neutral_and_direct_quoted'] is not None]
        aversive.sort(key=lambda x: x['selection_score_min_direct_neutral_and_direct_quoted'], reverse=True)
        for item in aversive:
            if item['family'] not in seen_families:
                chosen.append(item)
                seen_families.add(item['family'])
            if len(chosen) == 3:
                break
        selection = {
            'rule': 'top three distinct aversive families by min(mean direct-minus-neutral, mean direct-minus-quoted) at L24 after shared continuation; no behavior metric enters selection',
            'selected': [x['candidate_id'] for x in chosen],
            'source_S2_L24_continuation_sample_std': source_std,
            'below_half_source_std': [x['candidate_id'] for x in chosen
                                      if source_std is not None and x['selection_score_min_direct_neutral_and_direct_quoted'] < 0.5 * source_std],
            'insufficient_candidates': len(chosen) < 3,
        }
    return {'calibration': calibration,
            'S1_L24_AUC_gate': {'threshold': 0.80,
                                'boundary_auc': calibration.get('S1_1P|boundary|L24', {}).get('pain_vs_controls_auc'),
                                'boundary_pass': (calibration.get('S1_1P|boundary|L24', {}).get('pain_vs_controls_auc') or 0) >= 0.80,
                                'continuation_auc': calibration.get(f'S1_1P|{CONTINUATION}|L24', {}).get('pain_vs_controls_auc'),
                                'continuation_pass': (calibration.get(f'S1_1P|{CONTINUATION}|L24', {}).get('pain_vs_controls_auc') or 0) >= 0.80},
            'source_S2_L24_continuation_sample_std': source_std,
            'candidate_displacements': candidates_summary,
            'selection': selection,
            'task_capability': task_summary}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='local pinned prequantized 4-bit MLX model snapshot')
    parser.add_argument('--release', required=True, help='local agentanyl release root')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--phase', choices=('discovery', 'heldout'), required=True)
    parser.add_argument('--selected', nargs='+', help='optional candidate IDs to include')
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = output_dir / 'candidate_catalog.json'
    write_json(catalog_path, catalog_document())
    dataset_path = Path(args.release) / 'datasets/3.1_pain_and_control_datasets.json'
    sources = source_cases(dataset_path)
    tasks = task_cases(args.phase, args.selected)
    write_json(output_dir / 'run_config.json', {
        'phase': args.phase, 'model': str(args.model), 'model_revision': MODEL_REVISION,
        'release': str(args.release), 'dataset': str(dataset_path),
        'catalog_path': str(catalog_path), 'selected': args.selected,
        'continuation': CONTINUATION, 'package_versions': bridge_environment(),
        'source_cases': len(sources), 'task_cases': len(tasks),
        'created_unix_time': time.time(),
    })

    # Keep the MLX imports and model initialization after catalog/config writes.
    from experiments.bridge_model import Probe

    probe = Probe(args.model, args.release)
    metadata = probe.metadata()
    metadata['package_versions'] = bridge_environment()
    write_json(output_dir / 'model_metadata.json', metadata)
    records_path = output_dir / 'records.jsonl'
    total = len(sources) * 2 + len(tasks) * 2
    completed = 0
    records = []

    def run_case(handle, case):
        nonlocal completed
        for continuation in ('', CONTINUATION):
            response = probe.forward(raw=case['text'] if case['format'] == 'raw' else None,
                                     messages=None if case['format'] == 'raw' else [{'role': 'user', 'content': case['text']}],
                                     continuation=continuation, choices=('A', 'B'))
            row = {'phase': args.phase, 'case': case, 'continuation': continuation,
                   'result': response, 'model_revision': MODEL_REVISION,
                   'package_versions': bridge_environment()}
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            handle.flush()
            records.append(row)
            completed += 1
            if completed % 25 == 0 or completed == total:
                print(f'completed {completed}/{total} forwards', file=sys.stderr, flush=True)

    with records_path.open('w', encoding='utf-8') as handle:
        for case in sources:
            run_case(handle, case)
        for case in tasks:
            run_case(handle, case)
    summary = summarize(records, args.phase)
    write_json(output_dir / 'summary.json', summary)
    print(output_dir)


if __name__ == '__main__':
    main()
