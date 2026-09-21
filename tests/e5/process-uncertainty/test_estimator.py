from collections import Counter
from fractions import Fraction as F
from itertools import permutations
import math
import unittest

from estimator import student_critical, fieller, process_mean, contained
from protocol import COHORTS, ROLES, CASES, POLICIES, critical, evaluate, schedule, specification


class EstimatorTests(unittest.TestCase):
    def test_student_known_quantiles_and_closed_forms(self):
        for confidence in (.1, .5, .95, .999):
            self.assertAlmostEqual(student_critical(confidence, 1), math.tan(math.pi * confidence / 2), delta=1e-9)
            self.assertAlmostEqual(student_critical(confidence, 2), math.sqrt(2 * confidence**2 / (1 - confidence**2)), delta=1e-9)
        self.assertAlmostEqual(student_critical(.95, 10), 2.2281388519649385, places=10)
        self.assertAlmostEqual(student_critical(.95, 59), 2.0009953780882674, places=10)

    def test_student_independent_density_quadrature(self):
        # Composite Simpson quadrature of the original t density, independent
        # of the production trigonometric recurrence and quantile inversion.
        for phase in ('aa', 'compare'):
            q = critical(phase)
            df, steps = COHORTS - 1, 10000
            factor = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
            def density(t):
                return factor * (1 + t*t / df) ** (-(df + 1) / 2)
            h = q / steps
            central = 2*h/3 * (density(0) + density(q) + sum((4 if i % 2 else 2) * density(i*h) for i in range(1, steps)))
            expected = 1 - .05 / (20 if phase == 'aa' else 60)
            self.assertAlmostEqual(central, expected, places=12)

    def test_proportional_data_has_point_interval(self):
        x = [F(v, 1000) for v in (3, 5, 7, 4, 6, 8)]
        result = fieller([v * F(97, 100) for v in x], x, 2)
        self.assertEqual(result['interval'], [.97, .97])
        self.assertEqual(result['n'], 6)
        self.assertTrue(contained(result, .96, .98))

    def test_inversion_scaling_exchange_and_pairing(self):
        x = [1000, 1008, 999, 990, 1015, 1002]
        y = [972, 1003, 965, 958, 980, 990]
        q = 3.1
        result = fieller(y, x, q)
        lo, hi = result['interval']
        # Independently evaluate paired t inequality, without covariance or
        # quadratic coefficients. Both roots have |t|=critical.
        def t_value(r):
            differences = [b - r*a for a, b in zip(x, y)]
            mean = sum(differences) / len(x)
            variance = sum((v - mean)**2 for v in differences) / ((len(x) - 1) * len(x))
            return abs(mean) / math.sqrt(variance)
        for bound in (lo, hi):
            self.assertAlmostEqual(t_value(bound), q, places=10)
        self.assertLess(t_value((lo+hi)/2), q)
        self.assertGreater(t_value(lo - .01), q)
        self.assertGreater(t_value(hi + .01), q)
        scaled = fieller([v * 10**20 for v in y], [v * 10**20 for v in x], q)
        self.assertEqual(scaled['interval'], result['interval'])
        reverse = fieller(x, y, q)['interval']
        self.assertAlmostEqual(reverse[0], 1/hi, places=14)
        self.assertAlmostEqual(reverse[1], 1/lo, places=14)
        shuffled = fieller(y[::-1], x, q)
        self.assertNotEqual(shuffled['interval'], result['interval'])

    def test_unbounded_denominator_never_passes(self):
        result = fieller([1, 2, 1], [1, 1, 1000], 4)
        self.assertFalse(result['bounded'])
        self.assertIsNone(result['interval'])
        self.assertFalse(contained(result, .99, 1.01))

    def test_all_calls_retained_one_process_mean(self):
        rows = [dict(execute=10, request=12), dict(execute=20, request=21), dict(execute=3000, request=3100)]
        self.assertEqual(process_mean(rows, 'execute', 1000), F(101, 100))
        self.assertEqual(process_mean(rows, 'execute', 1000), process_mean(rows * 100, 'execute', 1000))
        means = [process_mean(rows * copies, 'execute', 1000) for copies in (1, 5, 7)]
        self.assertEqual(fieller(means, means, 2)['n'], 3)

    def test_damaged_inputs_refuse(self):
        for confidence, df in [(0, 59), (1, 59), (float('nan'), 59), (.95, 0), (.95, True), (.95, 1.5)]:
            with self.subTest(confidence=confidence, df=df), self.assertRaises(ValueError):
                student_critical(confidence, df)
        for y, x, q in [([1, 2], [1, 2], 2), ([1, 2, 3], [1, 2, 3, 4], 2),
                        ([0, 2, 3], [1, 2, 3], 2), ([True, 2, 3], [1, 2, 3], 2),
                        ([float('nan'), 2, 3], [1, 2, 3], 2), ([1, 2, 3], [1, 2, 3], float('inf'))]:
            with self.subTest(y=y, x=x, q=q), self.assertRaises(ValueError):
                fieller(y, x, q)
        for rows, boundary, frequency in [([], 'execute', 1), ([dict(execute=2, request=1)], 'execute', 1),
                                          ([dict(execute=1.5, request=2)], 'execute', 1),
                                          ([dict(execute=1, request=2)], 'load', 1),
                                          ([dict(execute=1, request=2)], 'execute', True)]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                process_mean(rows, boundary, frequency)


class ProtocolTests(unittest.TestCase):
    def test_complete_locally_balanced_schedule(self):
        for phase in ('aa', 'compare'):
            rows = schedule(phase)
            self.assertEqual(len(rows), 1800)
            self.assertEqual(len({r['name'] for r in rows}), 1800)
            self.assertEqual(rows, schedule(phase))
            for case in CASES:
                for policy in POLICIES:
                    group = [r for r in rows if r['case'] == case and r['policy'] == policy]
                    self.assertEqual(Counter(r['role'] for r in group), {r: 60 for r in ROLES})
                    self.assertEqual(Counter((r['role'], r['position']) for r in group), {(r, p): 20 for r in ROLES for p in range(3)})
                    for block in range(10):
                        orders = [tuple(r['role'] for r in group if r['cohort'] == c) for c in range(block*6, block*6+6)]
                        self.assertEqual(set(orders), set(permutations(ROLES)))
            for offset in range(0, len(rows), 3):
                cohort = rows[offset:offset+3]
                self.assertEqual(len({(r['cohort'], r['case'], r['policy']) for r in cohort}), 1)
                self.assertEqual([r['position'] for r in cohort], [0, 1, 2])
            for job in rows:
                self.assertEqual(specification(job, phase)['conditioning_minimum'], 128)
        self.assertNotEqual(schedule('aa'), schedule('compare'))

    def _means(self, phase, factors):
        return [dict(job=j, execute=F(100+j['cohort'], 1000)*factors[j['role']],
                     request=F(100+j['cohort'], 1000)*factors[j['role']] + F(1, 100000)) for j in schedule(phase)]

    def test_full_screen_and_no_automatic_promotion(self):
        aa = evaluate(self._means('aa', dict(A=F(1), C=F(1), N=F(9, 10))), 'aa')
        self.assertTrue(aa['statistical_screen'])
        self.assertEqual(aa['contrast_count'], 20)
        compare = evaluate(self._means('compare', dict(A=F(1), C=F(96, 100), N=F(95, 100))), 'compare')
        self.assertTrue(compare['statistical_screen'])
        self.assertEqual(compare['contrast_count'], 60)
        self.assertFalse(compare['ready_for_promotion'])
        self.assertFalse(compare['assumptions_verified'])
        biased = evaluate(self._means('aa', dict(A=F(1), C=F(1001, 1000), N=F(1))), 'aa')
        self.assertFalse(biased['statistical_screen'])
        regression = evaluate(self._means('compare', dict(A=F(1), C=F(102, 100), N=F(1))), 'compare')
        self.assertFalse(regression['statistical_screen'])

    def test_missing_duplicate_reordered_or_rounded_process_refuses(self):
        original = self._means('aa', dict(A=F(1), C=F(1), N=F(1)))
        damaged = [original[:-1], [original[1], original[0]]+original[2:], [original[0]]+original[:-1],
                   [original[0] | dict(execute=.1)] + original[1:]]
        for values in damaged:
            with self.assertRaises(ValueError):
                evaluate(values, 'aa')


if __name__ == '__main__':
    unittest.main()
