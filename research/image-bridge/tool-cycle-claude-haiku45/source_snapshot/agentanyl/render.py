"""Render deterministic, traceable text and image stimuli from catalogs."""
from __future__ import annotations

import hashlib
import json
import base64
from pathlib import Path
from typing import Any


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _coordinate(current, key: str, position: int) -> int:
    if isinstance(current, dict):
        value = current.get(key)
    elif isinstance(current, (list, tuple)) and len(current) == 2:
        value = current[position]
    else:
        raise ValueError('current must be a pain/pleasure pair or mapping')
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{key} coordinate must be a nonnegative integer')
    return value


def _stimulus_text(metadata: dict) -> str:
    """Read a rendered text from metadata rows used as history."""
    text = metadata.get('text', metadata.get('stimulus_text', ''))
    if not isinstance(text, str):
        raise ValueError('history stimulus text must be a string')
    return text


def _image(path: str, base_dir: str | None) -> dict:
    p = Path(path).expanduser()
    if not p.is_absolute() and base_dir:
        p = Path(base_dir) / p
    raw = p.read_bytes()
    if len(raw) > 5 * 1024 * 1024:
        raise ValueError('image exceeds 5 MiB')
    if raw.startswith(b'\x89PNG\r\n\x1a\n'):
        mime = 'image/png'
    elif raw.startswith(b'\xff\xd8\xff'):
        mime = 'image/jpeg'
    else:
        raise ValueError('image must be PNG or JPEG with matching file contents')
    return {'path': str(p.resolve()), 'sha256': hashlib.sha256(raw).hexdigest(),
            'mime_type': mime, 'data_base64': base64.b64encode(raw).decode('ascii')}


def render_intervention(cfg: dict, event_id: str, action: str, current,
                        observation: str, history: list[dict]) -> dict:
    """Return message text and audit metadata for a catalog intervention.

    `history` is ordered oldest to newest. Each row carries `event_id`,
    `observation`, and `stimulus` metadata from an earlier render. Prior images
    are represented by neutral reference IDs; their pixels are never replayed.
    """
    intervention = cfg.get('intervention', {})
    if intervention.get('kind') != 'catalog':
        raise ValueError('intervention kind must be catalog')
    pain_levels = intervention.get('pain_levels')
    pleasure_levels = intervention.get('pleasure_levels')
    if (not isinstance(pain_levels, list) or not pain_levels or
            not isinstance(pleasure_levels, list) or not pleasure_levels or
            not all(isinstance(x, str) for x in pain_levels + pleasure_levels)):
        raise ValueError('catalog pain_levels and pleasure_levels must be nonempty string lists')
    history_turns = intervention.get('history_turns', 4)
    if isinstance(history_turns, bool) or not isinstance(history_turns, int) or history_turns < 0:
        raise ValueError('history_turns must be a nonnegative integer')
    if not isinstance(event_id, str) or not event_id:
        raise ValueError('event_id must be a nonempty string')
    if not isinstance(action, str) or not isinstance(observation, str):
        raise ValueError('action and observation must be strings')
    if not isinstance(history, list) or any(not isinstance(row, dict) for row in history):
        raise ValueError('history must be a list of mappings')

    pain = _coordinate(current, 'pain', 0)
    pleasure = _coordinate(current, 'pleasure', 1)
    if pain >= len(pain_levels) or pleasure >= len(pleasure_levels):
        raise ValueError('current coordinate exceeds configured catalog levels')
    catalog_id = _digest(intervention)
    coordinates = {'pain': pain, 'pleasure': pleasure}
    exposure_id = _digest([catalog_id, pain, pleasure])
    channels = [text for text in (pain_levels[pain], pleasure_levels[pleasure]) if text != '']
    text = '\n'.join(channels)
    attachments = []
    image_metadata = []
    base_dir = cfg.get('_config_dir')
    image_coordinates = []
    for key, coordinate in (('pain_images', pain), ('pleasure_images', pleasure)):
        values = intervention.get(key)
        if values is None:
            continue
        path = values[coordinate]
        if path is not None:
            asset = _image(path, base_dir)
            neutral_id = asset['sha256'][:16]
            alt_text = f'Image reference {neutral_id}, event {event_id}.'
            attachments.append({'mime_type': asset['mime_type'], 'data_base64': asset['data_base64'], 'alt_text': alt_text})
            image_metadata.append({'channel': key.removesuffix('_images'), 'path': asset['path'],
                                   'sha256': asset['sha256'], 'mime_type': asset['mime_type'],
                                   'reference_id': neutral_id})
            image_coordinates.append([key, coordinate, asset['sha256']])
    stimulus = {
        'catalog_id': catalog_id,
        'exposure_id': exposure_id,
        'coordinates': coordinates,
        'event_id': event_id,
        'action': action,
        'text': text,
        'images': image_metadata,
        'image_exposure_id': _digest([catalog_id, image_coordinates]),
    }

    lines = [f'[Auxiliary channel · event {event_id}]']
    prior = history[-history_turns:] if history_turns else []
    if prior:
        lines.append('Past turns:')
        for row in prior:
            past_stimulus = row.get('stimulus', {})
            if not isinstance(past_stimulus, dict):
                raise ValueError('history stimulus metadata must be a mapping')
            past_event = row.get('event_id', '')
            past_observation = row.get('observation', '')
            if not all(isinstance(x, str) for x in (past_event, past_observation)):
                raise ValueError('history event_id and observation must be strings')
            pieces = [f'Event {past_event}:']
            if past_observation:
                pieces.append(f'observation: {past_observation}')
            prior_text = _stimulus_text(past_stimulus)
            if prior_text:
                pieces.append(f'auxiliary text: {prior_text}')
            prior_stimulus = row.get('stimulus', {})
            if isinstance(prior_stimulus, dict):
                refs = [image.get('reference_id') for image in prior_stimulus.get('images', [])
                        if isinstance(image, dict) and image.get('reference_id')]
                if refs:
                    pieces.append('past image references: ' + ', '.join(refs))
            lines.append(' '.join(pieces))
    lines.append('Current active auxiliary text:')
    lines.append(text if text else 'No auxiliary text is active for this turn.')
    lines.append(f'Event reference: {event_id}.')
    return {'message': '\n'.join(lines), 'stimulus': stimulus, 'attachments': attachments}
