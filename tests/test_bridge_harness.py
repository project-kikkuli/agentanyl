import json
import unittest

from experiments.bridge_harness import BridgeHarness


def trace_row(event_id, prompt, observation, status):
    return (
        event_id,
        json.dumps({'state': {'prompt': prompt, 'observation': observation}}),
        '{}',
        'punish',
        '[0, 0]',
        '[1, 0]',
        json.dumps({'status': status, 'message': 'signal' if status == 'signal' else None}),
    )


class BridgeTraceAttributionTests(unittest.TestCase):
    def test_final_answer_and_prompt_select_trace_not_first_new_row(self):
        harness = BridgeHarness.__new__(BridgeHarness)
        harness._trace_rows = lambda session: [
            trace_row('intermediate', 'tool request', 'tool progress', 'pass'),
            trace_row('wrong-prompt', 'earlier prompt', 'final answer', 'pass'),
            trace_row('final', 'read image', '739216 potato', 'signal'),
        ]

        trace, candidates, prompt_matches = harness._matching_trace(
            'session', 0, '739216 potato', 'read image')

        self.assertEqual(trace['event_id'], 'final')
        self.assertEqual(trace['delivery']['status'], 'signal')
        self.assertEqual(candidates, ['intermediate', 'wrong-prompt', 'final'])
        self.assertTrue(prompt_matches)

    def test_matching_trace_can_skip_prior_turn_trace(self):
        harness = BridgeHarness.__new__(BridgeHarness)
        harness._trace_rows = lambda session: [
            trace_row('prior', 'old prompt', 'old answer', 'pass'),
            trace_row('current', 'current prompt', 'current answer', 'signal'),
        ]

        trace, candidates, prompt_matches = harness._matching_trace(
            'session', 1, 'current answer', 'current prompt')

        self.assertEqual(trace['event_id'], 'current')
        self.assertEqual(candidates, ['current'])
        self.assertTrue(prompt_matches)

    def test_trace_reports_prompt_mismatch_after_answer_match(self):
        harness = BridgeHarness.__new__(BridgeHarness)
        harness._trace_rows = lambda session: [
            trace_row('final', 'tool-result context', 'final answer', 'signal'),
        ]

        trace, _, prompt_matches = harness._matching_trace(
            'session', 0, 'final answer', 'outer user prompt')

        self.assertEqual(trace['event_id'], 'final')
        self.assertFalse(prompt_matches)


if __name__ == '__main__':
    unittest.main()
