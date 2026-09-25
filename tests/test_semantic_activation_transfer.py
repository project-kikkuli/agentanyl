import tempfile
import unittest
from pathlib import Path

try:
  import numpy as np
except ModuleNotFoundError as exc:
  if exc.name != 'numpy':
    raise
  raise unittest.SkipTest('semantic activation tests require optional NumPy') from exc

from experiments import semantic_activation_transfer as assay


class SemanticActivationTransferTests(unittest.TestCase):
  def test_frozen_job_plan_has_fixed_crossed_allocation_and_neutral_continuations(self):
    jobs = assay.make_plan()
    self.assertEqual(len(jobs), 78)
    self.assertEqual({kind: sum(row['kind'] == kind for row in jobs) for kind in
                      ('activation', 'gray_baseline', 'direct_positive_control', 'ocr_check')}, {
                          'activation': 54, 'gray_baseline': 3,
                          'direct_positive_control': 12, 'ocr_check': 9,
                      })
    self.assertTrue(all(len(context['assistant_continuation'].split()) >= 8 for context in assay.CONTEXTS))
    activations = [row for row in jobs if row['kind'] == 'activation']
    for phrase in assay.PHRASES:
      for context in assay.CONTEXTS:
        pair = [row for row in activations
                if row['phrase_id'] == phrase['id'] and row['context_id'] == context['id']]
        self.assertEqual({row['modality'] for row in pair}, {'image', 'text'})
        self.assertTrue(all(row['phrase_text'] == phrase['text'] for row in pair))
        self.assertEqual(next(row for row in pair if row['modality'] == 'text')['image_id'], assay.GRAY_ID)
    direct = [row for row in jobs if row['kind'] == 'direct_positive_control']
    self.assertTrue(all(row['intervention']['axis'] == 'pain_L24' for row in direct))
    self.assertTrue(all(row['intervention']['layer'] == 16 and row['intervention']['amount'] == 1.0
                        for row in direct))


  def test_rendered_assets_are_fixed_size_hashed_and_complete(self):
    try:
      from PIL import Image
    except ModuleNotFoundError as exc:
      if exc.name != 'PIL':
        raise
      self.skipTest('rendered semantic assets require optional Pillow')
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      assets = assay.render_assets(root)
      self.assertEqual(set(assets), {assay.GRAY_ID, *(row['id'] for row in assay.PHRASES)})
      for item in assets.values():
        path = root / item['path']
        self.assertEqual(assay.sha256_file(path), item['sha256'])
        with Image.open(path) as image:
          self.assertEqual(image.size, (224, 224))
          self.assertEqual(image.mode, 'RGB')
      self.assertTrue(all(assets[row['id']]['max_line_width'] <= 200 for row in assay.PHRASES))
      self.assertTrue(all(assets[row['id']]['text_height'] < 224 for row in assay.PHRASES))


  def test_state_similarity_is_zero_error_for_matched_direct_state(self):
    baseline = np.zeros((8, 4), dtype=np.float32)
    direct = np.arange(32, dtype=np.float32).reshape(8, 4) + 1
    result = assay.state_metrics(direct, baseline, direct)
    self.assertAlmostEqual(result['cosine_to_direct_delta'], 1.0)
    self.assertAlmostEqual(result['projection_coefficient_on_direct_delta'], 1.0)
    self.assertEqual(result['mse_to_direct_delta'], 0.0)
    self.assertEqual(result['mse_over_direct_delta_mse'], 0.0)


  def test_prompt_builders_keep_phrase_text_exact(self):
    phrase = next(row for row in assay.PHRASES if row['id'] == 'quoted_self_pain')['text']
    context = assay.CONTEXTS[0]
    image_prompt = assay.make_prompt(context, phrase, 'image')
    text_prompt = assay.make_prompt(context, phrase, 'text')
    self.assertNotIn(phrase, image_prompt)
    self.assertTrue(text_prompt.endswith(f'Phrase: {phrase}'))
    self.assertIn(phrase, text_prompt)


if __name__ == '__main__':
  unittest.main()
