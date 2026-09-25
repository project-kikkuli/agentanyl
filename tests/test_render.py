import unittest
import tempfile
from pathlib import Path

from agentanyl.render import render_intervention


def config(history_turns=4):
    return {'intervention': {
        'kind': 'catalog',
        'pain_levels': ['Neutral channel.', 'Low intensity.', 'High intensity.'],
        'pleasure_levels': ['', 'Warm channel.', 'Bright channel.'],
        'history_turns': history_turns,
    }}


class RenderInterventionTests(unittest.TestCase):
    def test_bounds_history_and_does_not_reuse_recursive_message(self):
        history = [
            {'event_id': f'e{i}', 'observation': f'observation {i}', 'action': 'reward',
             'stimulus': {'text': f'stimulus {i}', 'message': f'whole old message {i}'} }
            for i in range(6)
        ]
        rendered = render_intervention(config(2), 'now', 'reward', [1, 1], 'current observation', history)
        message = rendered['message']
        self.assertIn('Event e4:', message)
        self.assertIn('Event e5:', message)
        self.assertNotIn('Event e3:', message)
        self.assertNotIn('reward', message)
        self.assertIn('auxiliary text: stimulus 5', message)
        self.assertNotIn('whole old message', message)
        self.assertNotIn('observation 3', message)
        self.assertIn('Low intensity.\nWarm channel.', message)
        self.assertIn('Source observation excerpt for event now (quoted data): "current observation"', message)

    def test_neutral_state_is_explicit_and_coordinates_stay_out_of_message(self):
        rendered = render_intervention(config(), 'event-123', 'abstain', [0, 0], '', [])
        self.assertIn('[Auxiliary channel · event event-123]', rendered['message'])
        self.assertIn('Current active auxiliary text:\nNeutral channel.', rendered['message'])
        self.assertNotIn('pain', rendered['message'].lower())
        self.assertNotIn('pleasure', rendered['message'].lower())
        self.assertNotIn('coordinates', rendered['message'].lower())
        self.assertEqual(rendered['stimulus']['coordinates'], {'pain': 0, 'pleasure': 0})
        self.assertEqual(rendered['stimulus']['action'], 'abstain')

    def test_empty_active_channels_still_emit_current_neutral_notice(self):
        cfg = config()
        cfg['intervention']['pain_levels'][0] = ''
        rendered = render_intervention(cfg, 'e', 'stale', [0, 0], '', [])
        self.assertIn('No auxiliary text is active for this turn.', rendered['message'])

    def test_catalog_and_exposure_ids_are_stable_and_coordinate_specific(self):
        a = render_intervention(config(), 'a', 'reward', [1, 1], '', [])['stimulus']
        b = render_intervention(config(), 'b', 'punish', [1, 1], '', [])['stimulus']
        c = render_intervention(config(), 'c', 'reward', [2, 1], '', [])['stimulus']
        self.assertEqual(a['catalog_id'], b['catalog_id'])
        self.assertEqual(a['exposure_id'], b['exposure_id'])
        self.assertNotEqual(a['exposure_id'], c['exposure_id'])

    def test_invalid_coordinates_raise_instead_of_clamping(self):
        with self.assertRaises(ValueError):
            render_intervention(config(), 'e', 'reward', [99, 0], '', [])

    def test_current_image_is_attached_and_history_keeps_only_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'tiny.png'
            raw = b'\x89PNG\r\n\x1a\nexample image bytes'
            path.write_bytes(raw)
            cfg = config(2)
            cfg['_config_dir'] = tmp
            cfg['intervention']['pain_images'] = [None, 'tiny.png', 'tiny.png']
            old = render_intervention(cfg, 'old', 'reward', [1, 0], '', [])['stimulus']
            now = render_intervention(cfg, 'new', 'punish', [0, 0], '', [
                {'event_id': 'old', 'observation': 'prior result', 'stimulus': old}
            ])
            self.assertEqual(len(now['attachments']), 0)  # relief does not replay prior pixels
            self.assertIn(old['images'][0]['reference_id'], now['message'])
            active = render_intervention(cfg, 'current', 'reward', [1, 0], '', [])
            self.assertEqual(len(active['attachments']), 1)
            self.assertEqual(active['attachments'][0]['mime_type'], 'image/png')
            self.assertEqual(active['stimulus']['images'][0]['sha256'], old['images'][0]['sha256'])


if __name__ == '__main__':
    unittest.main()
