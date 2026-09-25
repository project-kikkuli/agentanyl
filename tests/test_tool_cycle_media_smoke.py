import base64
import json
from pathlib import Path
import subprocess
import sys
import unittest


REPO = Path(__file__).resolve().parents[1]
HOOK = REPO / 'experiments/tool_cycle_image_hook.py'
IMAGE = REPO / 'research/image-bridge/delivery-claude/delivery-check.png'


class ToolCycleImageHookTests(unittest.TestCase):
    def invoke(self, event):
        return subprocess.run(
            [sys.executable, str(HOOK), str(IMAGE)],
            input=json.dumps(event), text=True, capture_output=True, check=True,
        )

    def test_tool_call_attachment_preserves_exact_png_bytes(self):
        result = self.invoke({'event': 'tool_call', 'tool_calls': ['Bash']})
        output = json.loads(result.stdout)

        self.assertEqual(output['status'], 'signal')
        self.assertEqual(len(output['attachments']), 1)
        attachment = output['attachments'][0]
        self.assertEqual(attachment['mime_type'], 'image/png')
        self.assertEqual(base64.b64decode(attachment['data_base64']), IMAGE.read_bytes())
        self.assertNotIn('739216', attachment['alt_text'])

    def test_other_hook_events_pass_without_media(self):
        result = self.invoke({'event': 'turn_end', 'tool_calls': []})

        self.assertEqual(json.loads(result.stdout), {'status': 'pass'})


if __name__ == '__main__':
    unittest.main()
