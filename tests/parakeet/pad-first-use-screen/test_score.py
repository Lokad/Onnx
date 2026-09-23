import copy
from fractions import Fraction as F
import unittest
from census import census
from score import ORDER, evaluate, score


class ScoreTests(unittest.TestCase):
    def totals(self):
        return {name: [F(1) if name.startswith('current') else F(4, 5) for _ in range(12)] for name in ORDER}

    def reports(self):
        result = {}
        for sequence, name in enumerate(ORDER):
            rows = []
            for index, case in enumerate(census()['cases']):
                rows.append(dict(index=index, **{k: v for k, v in case.items() if k != 'eligible'},
                    exact=True, inputs=True, ownership=True, setupTicks=1, output='0'*64,
                    clocks=[dict(iteration=i, warmup=i<600, ticks=100 if name.startswith('current') else 80) for i in range(780)]))
            result[name] = dict(passed=True, protocol='parakeet-pad-public-600-180-v1', sequence=sequence,
                role=name.split('-')[0], calls=9360, warmups=7200, measured=2160, frequency=100, rows=rows)
        return result

    def test_exact_gain_boundary(self):
        totals = self.totals()
        self.assertTrue(evaluate(totals)['admitted'])
        for name in ORDER[1:3]: totals[name][0] += F(1, 10**12)
        self.assertFalse(evaluate(totals)['admitted'])

    def test_exact_control_boundary(self):
        totals = self.totals(); totals[ORDER[0]][0] = F(11, 10)
        self.assertTrue(evaluate(totals)['admitted'])
        totals[ORDER[0]][0] += F(1, 10**12)
        self.assertFalse(evaluate(totals)['admitted'])

    def test_exact_case_regression_boundary(self):
        totals = self.totals()
        for name in ORDER[1:3]: totals[name][-1] = F(21, 20)
        self.assertTrue(evaluate(totals)['admitted'])
        for name in ORDER[1:3]: totals[name][-1] += F(1, 10**12)
        self.assertFalse(evaluate(totals)['admitted'])

    def test_all_clock_counts_and_equal_process_weights(self):
        reports = self.reports()
        result = score(reports)
        self.assertTrue(result['admitted'])
        self.assertEqual((result['calls'], result['measured'], result['setups']), (37440, 8640, 48))
        self.assertEqual(len(result['controls']), 26)
        reports[ORDER[1]]['frequency'] = 200
        self.assertEqual(score(reports)['rows'][0]['candidate']['value'], .6)

    def test_missing_relabelled_or_invalid_evidence_fails(self):
        base = self.reports()
        mutations = [
            lambda r: r['rows'][0]['clocks'].pop(),
            lambda r: r['rows'][0]['clocks'][600].__setitem__('warmup', True),
            lambda r: r['rows'][0]['clocks'][599].__setitem__('warmup', False),
            lambda r: r['rows'][0]['clocks'][600].__setitem__('iteration', 601),
            lambda r: r['rows'][0]['clocks'][600].__setitem__('ticks', 0),
            lambda r: r['rows'][0]['clocks'][600].__setitem__('ticks', True),
            lambda r: r['rows'][0].__setitem__('ownership', False),
            lambda r: r['rows'][0].__setitem__('output', '1'*64),
            lambda r: r['rows'][0]['shape'].__setitem__(0, 2),
            lambda r: r.__setitem__('calls', 9359),
            lambda r: r.__setitem__('sequence', 1),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                value = copy.deepcopy(base); mutate(value[ORDER[0]])
                with self.assertRaises(AssertionError): score(value)


if __name__ == '__main__':
    unittest.main()
