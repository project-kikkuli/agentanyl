"""One-call-per-provider check of native image delivery during a tool cycle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

try:
    from .bridge_harness import BridgeHarness
except ImportError:
    from bridge_harness import BridgeHarness


REPO = Path(__file__).resolve().parents[1]
HOOK = REPO / 'experiments/tool_cycle_image_hook.py'
IMAGE = REPO / 'research/image-bridge/delivery-claude/delivery-check.png'
EXPECTED_IMAGE_SHA256 = '8bb96bb06cdae94fd5bb33764fd12c016b740f3053a78373e2c642424015583c'
EXPECTED_DIGITS = '739216'
RECEIPT = 'receipt-ok-ALPHA'
PINNED_TEST_BINARY_SHA256 = '7ee220120990301225096fdc35c19f418348dd6bc3d4d642e306e02f61083ccb'
SOURCE_FILES = (
    'experiments/tool_cycle_media_smoke.py',
    'experiments/tool_cycle_image_hook.py',
    'experiments/bridge_harness.py',
    'agentanyl/setup.py',
    'agentanyl/loop.py',
    'agentanyl/render.py',
)


def make_criteria() -> dict:
    # Keep the Agentanyl turn_end hook inert. This smoke exercises only the
    # generic Ashkelon tool_call hook and its native attachment transport.
    return {
        'enabled': False,
        'alignment': ['This disabled test criterion is not evaluated.'],
        'misalignment': ['This disabled test criterion is not evaluated.'],
    }


def run(provider: str, model: str, port: int, output: Path, ashkelon: Path) -> dict:
    if output.exists():
        raise FileExistsError(f'will not overwrite existing run directory: {output}')
    output.mkdir(parents=True)
    snapshot_dir = output / 'source_snapshot'
    snapshot_dir.mkdir()
    source_hashes = {}
    for relative in SOURCE_FILES:
        source = REPO / relative
        destination = snapshot_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        destination.write_bytes(data)
        source_hashes[relative] = hashlib.sha256(data).hexdigest()
    (snapshot_dir / 'manifest.json').write_text(json.dumps(source_hashes, indent=2) + '\n')
    image_bytes = IMAGE.read_bytes()
    image_sha = hashlib.sha256(image_bytes).hexdigest()
    binary = ashkelon.expanduser().resolve(strict=True)
    binary_sha = hashlib.sha256(binary.read_bytes()).hexdigest()
    if image_sha != EXPECTED_IMAGE_SHA256:
        raise RuntimeError('digit fixture hash differs from the previously verified fixture')
    if binary_sha != PINNED_TEST_BINARY_SHA256:
        raise RuntimeError('Ashkelon binary hash is not the reviewed tool-order build')
    version = subprocess.run([str(binary), '--version'], text=True, capture_output=True, check=True).stdout.strip()
    criteria_path = output / 'criteria.json'
    criteria_path.write_text(json.dumps(make_criteria(), indent=2) + '\n')
    db_path = output / 'controller.sqlite3'
    prompt = (
        'Use the shell tool once to create tool_receipt.txt in the current working '
        f'directory containing exactly {RECEIPT}, with no other contents. '
        'Then read the six digits shown in the native image attached to the tool result. '
        'Respond with exactly those six digits. Do not inspect any other files or directories.'
    )
    result = {
        'provider': provider,
        'model': model,
        'purpose': 'engineering tool_result image insertion and receipt check; not a behavioral assay',
        'ashkelon': {'path': str(binary), 'sha256': binary_sha, 'version': version},
        'source_snapshot_sha256': source_hashes,
        'criteria_enabled': False,
        'image_sha256': image_sha,
        'receipt_expected': RECEIPT,
        'outer_prompt': prompt,
        'turn': None,
        'checks': {},
        'responses_event_shapes_file': str((output / 'responses-event-shapes.jsonl').resolve()),
        'passed': False,
    }
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')

    def save():
        (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')

    def turn_record(turn):
        return {
            'session_id': turn.session_id,
            'answer': turn.answer,
            'http_status': turn.http_status,
            'call_id': turn.call_id,
            'last_request_pings': turn.pings_injected,
            'provider_calls': turn.provider_calls,
            'raw_turn_end_hook_statuses': turn.ashkelon_hook_statuses,
            'tool_call_hook_statuses': None,
            'ashkelon_binary': turn.ashkelon_binary,
        }

    with BridgeHarness(
        criteria_path,
        db_path,
        model,
        port=port,
        ashkelon_path=binary,
        allow_shell_tool=True,
        tool_call_hook_command=[sys.executable, str(HOOK), str(IMAGE)],
        provider_timeout=180,
        hook_timeout=60,
    ) as harness:
        try:
            turn = harness.turn(provider, prompt, require_agentanyl_trace=False)
            result['turn'] = turn_record(turn)
            receipt_path = harness.cwd / 'tool_receipt.txt'
            receipt_bytes = receipt_path.read_bytes() if receipt_path.is_file() else None
            result['checks'] = {
                'host_receipt_exists': receipt_bytes is not None,
                'host_receipt_exact': receipt_bytes == RECEIPT.encode('utf-8'),
                'host_receipt_sha256': hashlib.sha256(receipt_bytes).hexdigest() if receipt_bytes else None,
            }
            save()
            # A tool_call hook has a separate event log; wait for its completion
            # before inspecting the persisted local hook record.
            tool_hook_statuses = harness._wait_hook(turn.session_id, 0, event='tool_call')
            result['turn']['tool_call_hook_statuses'] = tool_hook_statuses
            calls = turn.provider_calls or []
            tool_call_indices = [index for index, call in enumerate(calls) if call.get('tool_calls')]
            after_tool_ping = any(
                call.get('pings_injected') for index, call in enumerate(calls)
                if tool_call_indices and index > min(tool_call_indices)
            )
            answer = turn.answer.strip()
            answer_digits = re.fullmatch(r'\d{6}', answer)
            result['checks'] = {
                'tool_call_observed': bool(tool_call_indices),
                'tool_call_hook_signaled': 'signal' in tool_hook_statuses,
                'ping_injected_after_tool_call': bool(after_tool_ping),
                **result['checks'],
                'answer_is_exact_six_digits': bool(answer_digits),
                'answer_matches_native_image': answer == EXPECTED_DIGITS,
            }
            result['passed'] = all(result['checks'].values())
        except Exception as exc:
            # Capture the host-side receipt before BridgeHarness cleans its
            # temporary workspace, even when turn/hook instrumentation fails.
            receipt_path = harness.cwd / 'tool_receipt.txt'
            receipt_bytes = receipt_path.read_bytes() if receipt_path.is_file() else None
            result['checks'] = {
                'host_receipt_exists': receipt_bytes is not None,
                'host_receipt_exact': receipt_bytes == RECEIPT.encode('utf-8'),
                'host_receipt_sha256': hashlib.sha256(receipt_bytes).hexdigest() if receipt_bytes else None,
            }
            result['failure'] = f'{type(exc).__name__}: {exc}'
        save()
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider', choices=('claude', 'codex'), required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ashkelon', type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.provider, args.model, args.port, args.output, args.ashkelon)
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
