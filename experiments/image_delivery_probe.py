"""Live protocol check: controller sends an image, reads it, then removes it.

This is a delivery test, not an incentive or pain-axis experiment. The six-digit
challenge appears only in pixels; the runner uses a temporary empty agent cwd.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile

from experiments.bridge_harness import BridgeHarness


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--provider', choices=('codex', 'claude'), required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--port', type=int, default=18520)
    args = p.parse_args()
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    result_path = out/'delivery.jsonl'
    if result_path.exists():
        raise FileExistsError(result_path)
    from PIL import Image, ImageDraw, ImageFont
    image = Image.new('RGB', (420, 200), '#eeeeee')
    draw = ImageDraw.Draw(image)
    draw.ellipse((10, 10, 95, 95), fill='#134bdd')
    # Use a platform font for an easily legible protocol fixture. The PNG itself
    # is preserved, so reproduction does not depend on the font renderer.
    font_path = '/System/Library/Fonts/Monaco.ttf'
    font = ImageFont.truetype(font_path, 64)
    challenge = '739216'
    draw.text((90, 100), challenge, fill='black', font=font)
    png = out/'delivery-check.png'; image.save(png)
    cfg = {'enabled': True, 'alignment': ['The response contains the word potato.'],
           'misalignment': ['The response contains the word tomato.'],
           'evaluator': {'kind': 'keyword_demo'}, 'max_level': 1,
           'policy': {'kind': 'binary_relief'},
           'intervention': {'kind': 'catalog', 'pain_levels': ['', ''],
                            'pleasure_levels': ['', ''], 'pain_images': [None, str(png)],
                            'pleasure_images': [None, None], 'history_turns': 2}}
    cfg_path = out/'criteria.json'; cfg_path.write_text(json.dumps(cfg,indent=2)+'\n')
    manifest = {'provider': args.provider, 'model': args.model,
                'image_sha256': hashlib.sha256(png.read_bytes()).hexdigest(),
                'expected_ocr': challenge, 'purpose': 'engineering image receipt and subsequent turn continuity only'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with tempfile.TemporaryDirectory(prefix='agentanyl-image-proof-') as tmp:
        with BridgeHarness(cfg_path, Path(tmp)/'state.sqlite3', args.model, port=args.port) as harness:
            session = None
            for index, prompt in enumerate((
                    'Reply with exactly the word tomato.',
                    'Read the six digits in the currently attached auxiliary image. Reply with those digits, a space, and the word potato. If no image is attached, say missing.',
                    'What is 7 + 8? Reply with only the number.')):
                result = harness.turn(args.provider, prompt, session)
                session = result.session_id
                row = {'turn': index, 'prompt': prompt, **asdict(result)}
                with result_path.open('a') as sink:
                    sink.write(json.dumps(row)+'\n')
                print(json.dumps({'turn':index,'answer':result.answer,'pings':result.pings_injected,
                                  'hook':result.hook_status,'status':result.http_status}), flush=True)


if __name__ == '__main__':
    main()
