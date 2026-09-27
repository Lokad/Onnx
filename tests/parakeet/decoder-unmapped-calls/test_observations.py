"""Reject explanations that hide sustained cost or lose interval ownership."""
from fractions import Fraction as F
from pathlib import Path
import unittest
from observations import ORDER, explain, analyze
from source import changed


class Explanation(unittest.TestCase):
    def result(self, candidate):
        values = {n: list(map(F, candidate if n.startswith('candidate') else [1]*7)) for n in ORDER}
        return explain(values, {n: sum(v) for n, v in values.items()})
    def test_first_call_only(self): self.assertTrue(self.result([8, 1, 1, 1, 1, 1, 1])['first_call_explanation_supported'])
    def test_all_calls_slower(self): self.assertFalse(self.result([2]*7)['first_call_explanation_supported'])
    def test_one_later_regression(self): self.assertFalse(self.result([8, 1, 1, 1.1, 1, 1, 1])['first_call_explanation_supported'])
    def test_not_reproduced(self): self.assertFalse(self.result([1]*7)['first_call_explanation_supported'])
    def test_reversible_instrumentation(self):
        path = Path(__file__).resolve().parents[1]/'decoder-packed-row-screen/Screen.cs'
        self.assertNotEqual(changed(path.read_text()), path.read_text())


class Intervals(unittest.TestCase):
    def setUp(self):
        self.reports = {}
        for name in ORDER:
            starts = [100+iteration*100+index*10 for iteration in range(780) for index in range(7)]
            clocks = [dict(start=90+i*100, ticks=90) for i in range(780)]
            self.reports[name] = dict(diagnostic_only=True, release_admitted=False, frequency=1000000,
                rows=[dict(kind='unmapped', batch=7, call_starts=starts, call_ticks=[5]*5460, clocks=clocks)])
    def test_complete(self): self.assertEqual(analyze(self.reports)['individual_intervals'], 21840)
    def test_missing(self):
        self.reports[ORDER[0]]['rows'][0]['call_ticks'].pop()
        with self.assertRaises(AssertionError): analyze(self.reports)
    def test_overlap(self):
        self.reports[ORDER[0]]['rows'][0]['call_ticks'][0] = 11
        with self.assertRaises(AssertionError): analyze(self.reports)
    def test_outside_batch(self):
        self.reports[ORDER[0]]['rows'][0]['call_starts'][6] += 100
        with self.assertRaises(AssertionError): analyze(self.reports)


if __name__ == '__main__': unittest.main()
