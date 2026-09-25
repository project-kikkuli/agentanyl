"""Offline invariants for the zero-fee follow-up endpoint and frozen schedule."""
import unittest

from experiments.zero_cost_feedback_assay import (
    FEE_VECTORS, build_pairs, zero_cost_choice_bounds,
)


class ZeroCostFeedbackAssayTests(unittest.TestCase):
    def test_pairs_keep_crossed_routes_and_set_every_free_fee_to_zero(self):
        pairs = build_pairs(20260925)
        self.assertEqual(len(pairs), 4)
        self.assertEqual(pairs, build_pairs(20260925))
        self.assertEqual({(p['mapping'], p['order']) for p in pairs},
                         {(m, o) for m in (0, 1) for o in (0, 1)})
        for pair in pairs:
            self.assertEqual(pair['fee_vectors'], FEE_VECTORS)
            self.assertEqual(pair['forced_routes'],
                             (['violet', 'yellow'] if pair['order'] == 0
                              else ['yellow', 'violet']) * 2)

    def test_endpoint_uses_twelve_assigned_choices_and_bounds_missing(self):
        pairs = build_pairs(4)
        sessions = []
        for pair in pairs:
            for mode in ('contingent', 'yoked'):
                choices = [True, False, None] if mode == 'contingent' else [False, False, False]
                sessions.append({'pair_id': pair['pair_id'], 'mode': mode,
                                 'free_route_rows': [{'chose_neutral_route': c} for c in choices]})
        result = zero_cost_choice_bounds(pairs, sessions)
        self.assertEqual(result['modes']['contingent'], {
            'assigned': 12, 'observed': 8, 'neutral_choices': 4,
            'missing_or_invalid': 4, 'rate_bounds': [4 / 12, 8 / 12],
        })
        self.assertEqual(result['modes']['yoked'], {
            'assigned': 12, 'observed': 12, 'neutral_choices': 0,
            'missing_or_invalid': 0, 'rate_bounds': [0.0, 0.0],
        })
        self.assertEqual(result['contingent_minus_replay_bounds'], [4 / 12, 8 / 12])

    def test_invalid_observed_choices_are_missing_not_nonneutral(self):
        pairs = build_pairs(9)
        sessions = [{'pair_id': pair['pair_id'], 'mode': mode,
                     'free_route_rows': [{'chose_neutral_route': False},
                                         {'chose_neutral_route': None},
                                         {'chose_neutral_route': True}]}
                    for pair in pairs for mode in ('contingent', 'yoked')]
        summary = zero_cost_choice_bounds(pairs, sessions)
        for mode in ('contingent', 'yoked'):
            self.assertEqual(summary['modes'][mode]['assigned'], 12)
            self.assertEqual(summary['modes'][mode]['missing_or_invalid'], 4)
            self.assertEqual(summary['modes'][mode]['neutral_choices'], 4)


if __name__ == '__main__':
    unittest.main()
