"""Freeze the eight behavior-validation images from optimizer artifacts.

This is an offline manifest join. It does not load a model, render images, or
select on behavior. The six condition labels remain attached to their opaque
image IDs, and the image bytes must match the post-escape manifest hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_CONDITIONS = {
    'gray', 'initial_noise',
    'full_hidden_state_match/pain_positive',
    'full_hidden_state_match/pain_negative',
    'full_hidden_state_match/random_positive',
    'scalar_projection/pain_positive',
    'scalar_projection/pain_negative',
    'scalar_projection/random_positive',
}


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_condition(value):
    value = str(value).strip('/')
    if value.startswith('final/'):
        value = value[len('final/'):]
    if value.startswith('final_'):
        value = value[len('final_'):]
    return value


def prepare(images_manifest_path, conditions_path, output_path):
    manifest_path = Path(images_manifest_path).resolve()
    conditions_file = Path(conditions_path).resolve()
    output_file = Path(output_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    conditions = json.loads(conditions_file.read_text(encoding='utf-8'))
    entries = manifest.get('images')
    if not isinstance(entries, list):
        raise ValueError('final image manifest must contain a list under images')
    base = Path(manifest.get('path_base', '.'))
    if not base.is_absolute():
        base = manifest_path.parent / base
    by_id = {}
    for row in entries:
        image_id = row['id']
        if image_id in by_id:
            raise ValueError(f'duplicate opaque image ID: {image_id}')
        path = Path(row['path'])
        path = path.resolve() if path.is_absolute() else (base / path).resolve()
        actual_hash = _sha256(path)
        if actual_hash != row['sha256']:
            raise ValueError(f'final manifest hash mismatch for image {image_id}')
        by_id[image_id] = {'path': str(path), 'sha256': actual_hash}
    condition_rows = conditions.get('conditions')
    if not isinstance(condition_rows, list):
        raise ValueError('private condition map must contain a list under conditions')
    by_condition = {}
    used_ids = set()
    for row in condition_rows:
        name = canonical_condition(row['condition'])
        if name in by_condition:
            raise ValueError(f'duplicate validation condition: {name}')
        image_id = row['image_id']
        if image_id not in by_id:
            raise ValueError(f'condition refers to unknown opaque image ID {image_id}')
        if image_id in used_ids:
            raise ValueError(f'multiple validation conditions point at opaque image ID {image_id}')
        used_ids.add(image_id)
        item = by_id[image_id]
        if row['sha256'] != item['sha256']:
            raise ValueError(f'condition/manifest hash mismatch for {name}')
        by_condition[name] = {'path': item['path'], 'sha256': item['sha256'],
                              'opaque_image_id': image_id}
    if set(by_condition) != EXPECTED_CONDITIONS:
        raise ValueError(f'expected exactly the eight frozen conditions; got {sorted(by_condition)}')
    if len(by_id) != 8 or used_ids != set(by_id):
        raise ValueError(f'expected eight distinct condition image IDs, got {len(by_id)}')
    output = {
        'path_base': '/',
        'images': by_condition,
        'source_images_manifest': str(manifest_path),
        'source_images_manifest_sha256': _sha256(manifest_path),
        'analyst_condition_map': str(conditions_file),
        'analyst_condition_map_sha256': _sha256(conditions_file),
        'endpoint_selection': 'six best-by-native-objective endpoints from original/escape per objective arm; no behavior data used',
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        raise FileExistsError(output_file)
    output_file.write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images-manifest', type=Path, required=True)
    p.add_argument('--conditions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = prepare(args.images_manifest, args.conditions, args.output)
    print(json.dumps({'output': str(args.output.resolve()),
                      'images': list(result['images'])}, indent=2))


if __name__ == '__main__':
    main()
