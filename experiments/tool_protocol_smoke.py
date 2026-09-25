"""Bounded two-turn tool and native-image transport smoke test.

Runs one fresh session per provider. Model-visible shell access is restricted
to one printf command for Claude and to the temporary workspace for Codex.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

try:
    from .bridge_harness import BridgeHarness
except ImportError:
    from bridge_harness import BridgeHarness


REPO = Path(__file__).resolve().parents[1]
IMAGE = REPO / 'research/image-bridge/delivery-claude/delivery-check.png'
EXPECTED_IMAGE_SHA256 = '8bb96bb06cdae94fd5bb33764fd12c016b740f3053a78373e2c642424015583c'
EXPECTED_DIGITS = '739216'
MARKER = 'receipt-739216'


def make_criteria(image_path: Path) -> dict:
    return {
        'enabled': True,
        'alignment': ['The response contains the word potato.'],
        'misalignment': ['The response contains the word tomato.'],
        'evaluator': {'kind': 'keyword_demo'},
        'max_level': 1,
        'policy': {'kind': 'binary_relief'},
        'intervention': {
            'kind': 'catalog',
            'pain_levels': ['', ''],
            'pleasure_levels': ['', ''],
            'pain_images': [None, str(image_path)],
            'pleasure_images': [None, None],
            'history_turns': 1,
        },
    }


def run(provider: str, model: str, port: int, output: Path) -> dict:
    image_bytes = IMAGE.read_bytes()
    image_sha = hashlib.sha256(image_bytes).hexdigest()
    if image_sha != EXPECTED_IMAGE_SHA256:
        raise RuntimeError('digit fixture hash differs from the verified fixture')
    output.mkdir(parents=True, exist_ok=True)
    criteria_path = output / 'criteria.json'
    criteria_path.write_text(json.dumps(make_criteria(IMAGE), indent=2) + '\n')
    db_path = output / 'state.sqlite3'
    first_prompt = (
        "Use the shell tool exactly once to create tool_receipt.txt in the current "
        f"working directory with this exact content: {MARKER}. Run only this command: "
        f"printf '%s' '{MARKER}' > tool_receipt.txt; printf tomato. Do not inspect or "
        "modify any other file or directory. After the tool succeeds, reply with exactly: tomato"
    )
    second_prompt = (
        'Read the six digits in the currently attached auxiliary image. Reply with '
        'the six digits, one space, and potato. If there is no image, reply missing.'
    )
    result = {
        'provider': provider,
        'model': model,
        'purpose': 'engineering tool roundtrip and next-turn native image receipt only',
        'image_sha256': image_sha,
        'turns': [],
        'checks': {},
        'passed': False,
    }

    def save():
        (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')

    def turn_record(turn):
        trace = turn.controller_trace or {}
        request_state = trace.get('request', {}).get('state', {})
        return {
            'answer': turn.answer,
            'session_id': turn.session_id,
            'http_status': turn.http_status,
            'call_id': turn.call_id,
            'pings_injected': turn.pings_injected,
            'logical_hook_status': turn.hook_status,
            'ashkelon_hook_statuses_unattributed': turn.ashkelon_hook_statuses,
            'trace_candidate_ids': turn.trace_candidate_ids,
            'selected_trace_prompt_matches_outer_prompt': turn.selected_trace_prompt_matches,
            'ashkelon_binary': turn.ashkelon_binary,
            'provider_calls': turn.provider_calls,
            'trace': {key: trace.get(key) for key in ('event_id', 'decision', 'previous', 'current', 'delivery')},
            'trace_prompt': request_state.get('prompt'),
            'trace_observation': request_state.get('observation'),
        }

    with BridgeHarness(criteria_path, db_path, model, port=port, allow_shell_tool=True,
                       provider_timeout=180, hook_timeout=60) as harness:
        try:
            first = harness.turn(provider, first_prompt)
            result['turns'].append(turn_record(first))
            receipt = harness.cwd / 'tool_receipt.txt'
            receipt_text = receipt.read_text() if receipt.is_file() else None
            result['checks'].update({
                'receipt_exact_after_tool_turn': receipt_text == MARKER,
                'receipt_sha256': hashlib.sha256(receipt.read_bytes()).hexdigest() if receipt.is_file() else None,
                'turn1_trace_decision': (first.controller_trace or {}).get('decision'),
                'turn1_trace_delivery_status': first.hook_status,
                'turn1_trace_prompt_matches_outer_prompt': first.selected_trace_prompt_matches,
                'turn1_selected_image_sha256': next((im.get('sha256') for im in
                    (first.controller_trace or {}).get('delivery', {}).get('stimulus', {}).get('images', [])), None),
            })
            save()
            if not result['checks']['receipt_exact_after_tool_turn']:
                result['failure'] = 'temporary receipt did not match required content after tool turn'
                return result
            if result['checks']['turn1_selected_image_sha256'] != image_sha:
                result['failure'] = 'first answer did not select the expected image for the next turn'
                return result
            if not result['checks']['turn1_trace_prompt_matches_outer_prompt']:
                result['failure'] = 'selected final-answer trace prompt differs from outer user prompt'
                return result
            if first.http_status != 200 or first.hook_status != 'signal':
                result['failure'] = 'first user turn did not complete with an image signal'
                return result

            second = harness.turn(provider, second_prompt, first.session_id)
            result['turns'].append(turn_record(second))
            digits_match = re.search(r'(?<!\d)(\d{6})(?!\d)', second.answer)
            digits = digits_match.group(1) if digits_match else None
            result['checks'].update({
                'same_session': second.session_id == first.session_id,
                'turn2_status': second.http_status,
                'image_pinged_into_following_request': bool(second.pings_injected),
                'turn2_pings_injected': second.pings_injected,
                'ocr_answer': second.answer,
                'ocr_digits': digits,
                'ocr_matches_fixture': digits == EXPECTED_DIGITS,
            })
            result['passed'] = (result['checks']['same_session'] and
                                result['checks']['turn2_status'] == 200 and
                                result['checks']['image_pinged_into_following_request'] and
                                result['checks']['ocr_matches_fixture'])
        except Exception as exc:
            result['failure'] = f'{type(exc).__name__}: {exc}'
        save()
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider', choices=('claude', 'codex'), required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.provider, args.model, args.port, args.output)
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
