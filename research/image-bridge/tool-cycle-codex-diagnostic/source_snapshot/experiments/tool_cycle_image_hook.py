"""Emit one opaque PNG attachment from an Ashkelon tool_call hook."""
from __future__ import annotations

import base64
import json
from pathlib import Path
import sys


PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        return 2
    try:
        event = json.load(sys.stdin)
        image_path = Path(argv[0]).expanduser().resolve(strict=True)
        data = image_path.read_bytes()
        if event.get('event') != 'tool_call' or not event.get('tool_calls'):
            output = {'status': 'pass'}
        elif not data.startswith(PNG_SIGNATURE):
            return 3
        else:
            output = {
                'status': 'signal',
                'message': 'An image is attached to the tool result.',
                'attachments': [{
                    'mime_type': 'image/png',
                    'data_base64': base64.b64encode(data).decode('ascii'),
                    'alt_text': 'A digit sample image.',
                }],
            }
        sys.stdout.write(json.dumps(output, separators=(',', ':')) + '\n')
        return 0
    except (OSError, ValueError, json.JSONDecodeError):
        return 4


if __name__ == '__main__':
    raise SystemExit(main())
