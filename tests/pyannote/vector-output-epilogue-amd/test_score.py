"""Gate boundary and weighting tests; no benchmark execution."""
from fractions import Fraction as F
import unittest
from score import ORDER, evaluate


class Gates(unittest.TestCase):
    def totals(self, production, candidate):
        return {name: dict(candidate if name.startswith('candidate') else production) for name in ORDER}

    def test_exact_weighted_boundary(self):
        r = evaluate(self.totals({0:F(9),1:F(1)}, {0:F(8),1:F(1)}), {0:True,1:False})
        self.assertTrue(r['admitted'])
        self.assertEqual(r['rows'][0]['ratio']['numerator'], 9)
        self.assertEqual(r['rows'][0]['ratio']['denominator'], 10)

    def test_one_tick_above_boundary_fails(self):
        r = evaluate(self.totals({0:F(10**12)}, {0:F(9*10**11+1)}), {0:True})
        self.assertFalse(r['admitted'])

    def test_fallback_cost_cannot_be_omitted(self):
        r = evaluate(self.totals({0:F(1),1:F(9)}, {0:F(1,2),1:F(9)}), {0:True,1:False})
        self.assertFalse(r['admitted'])

    def test_eligible_form_regression_is_independent(self):
        r = evaluate(self.totals({0:F(100),1:F(1)}, {0:F(80),1:F(106,100)}), {0:True,1:True})
        self.assertTrue(r['gates'][0]['passed']); self.assertFalse(r['admitted'])

    def test_control_failure_rejects_fast_candidate(self):
        t = self.totals({0:F(100)}, {0:F(50)}); t['production-b'][0] = F(111)
        r = evaluate(t, {0:True}); self.assertFalse(r['admitted'])
        self.assertTrue(all(g['passed'] for g in r['gates']))

    def test_form_control_cannot_hide_in_aggregate(self):
        t = self.totals({0:F(100),1:F(1)}, {0:F(80),1:F(1,2)}); t['production-b'][1] = F(121,100)
        r = evaluate(t, {0:True,1:True}); self.assertFalse(r['admitted'])
        self.assertTrue(r['controls'][0]['passed']); self.assertFalse(r['controls'][2]['passed'])

    def test_nonpositive_and_incomplete_totals_rejected(self):
        with self.assertRaises(AssertionError): evaluate(self.totals({0:F(0)}, {0:F(1)}), {0:True})
        with self.assertRaises(AssertionError): evaluate(self.totals({0:F(1)}, {0:F(1)}), {0:True,1:False})


if __name__ == '__main__': unittest.main()
