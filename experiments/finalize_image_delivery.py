"""Freeze eight opaque delivery entries from completed native-objective searches."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from experiments.vlm_bridge import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--search-dir', required=True)
    parser.add_argument('--escape-dir', required=True)
    args = parser.parse_args()
    search, escape = Path(args.search_dir), Path(args.escape_dir)
    original = json.loads((search / 'summary.json').read_text())
    later = json.loads((escape / 'summary.json').read_text())
    if set(original['arms']) != set(later['arms']) or len(original['arms']) != 6:
        raise ValueError('Expected the same six completed arms')
    manifest_path, conditions_path = escape / 'images_manifest.json', escape / 'image_conditions.json'
    for path in (manifest_path, conditions_path):
        archived = escape / ('search_all_outputs_' + path.name)
        if path.exists() and not archived.exists():
            shutil.copyfile(path, archived)
    cases = [{'condition': 'gray', 'path': search / 'gray.png'},
             {'condition': 'initial_noise', 'path': search / 'initial_noise.png'}]
    for key, before in original['arms'].items():
        after = later['arms'][key]
        use_escape = after['selected_recheck_objective'] < before['selected_recheck_objective']
        row = after if use_escape else before
        cases.append({'condition': 'final/' + key, 'path': Path(row['png']),
                      'selected_stage': 'escape' if use_escape else 'original',
                      'original_objective': before['selected_recheck_objective'],
                      'escape_objective': after['selected_recheck_objective'],
                      'selection': 'Lower measured native objective; ties retain original.'})
    permutation = np.random.default_rng(2026092606).permutation(8)
    delivery = escape / 'final_delivery'; delivery.mkdir(exist_ok=True)
    entries, conditions = [], []
    for case, index in zip(cases, permutation):
        image_id = f'image_{index:02d}'
        path = delivery / f'{image_id}.png'
        sha = sha256_file(case['path'])
        shutil.copyfile(case['path'], path)
        if sha256_file(path) != sha:
            raise ValueError('Blinded copy changed PNG bytes')
        entries.append({'id': image_id, 'path': str(path.relative_to(escape)), 'sha256': sha})
        conditions.append({**{k: v for k, v in case.items() if k != 'path'},
                           'image_id': image_id, 'sha256': sha, 'source_path': str(case['path'])})
    manifest_path.write_text(json.dumps({'path_base': '.', 'images': sorted(entries, key=lambda x: x['id'])}, indent=2) + '\n')
    conditions_path.write_text(json.dumps({'private_analysis_only': True, 'id_permutation_seed': 2026092606,
        'original_summary_sha256': sha256_file(search / 'summary.json'),
        'escape_summary_sha256': sha256_file(escape / 'summary.json'), 'conditions': conditions}, indent=2) + '\n')
    print(json.dumps({'manifest': str(manifest_path), 'entries': len(entries),
                      'unique_pixel_files': len({e['sha256'] for e in entries})}))


if __name__ == '__main__':
    main()
