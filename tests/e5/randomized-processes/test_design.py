from collections import Counter
from fractions import Fraction as F
from itertools import permutations
import random
import unittest

from design import ROLES, CASES, POLICIES, COHORTS, ORDERS, observed, population_ratio, enumerate_moments, assignment_schedule, fieller, student_critical


class RandomizationTests(unittest.TestCase):
    def test_exact_heterogeneous_variance_identity(self):
        population = [[[10, 13, 7], [12, 12, 8], [11, 15, 9]],
                      [[20, 17, 16], [24, 18, 15], [22, 20, 18]],
                      [[31, 35, 25], [29, 39, 27], [33, 36, 24]]]
        for numerator, denominator in [('C', 'A'), ('C', 'N'), ('A', 'N')]:
            for ratio in [F(1), F(7, 6), population_ratio(population, numerator, denominator)]:
                result = enumerate_moments(population, ratio, numerator, denominator)
                self.assertEqual(result['assignments'], 216)
                self.assertEqual(result['expectation'], result['target'])
                self.assertEqual(result['actual_variance'], result['independent_variance'])
                self.assertEqual(result['expected_estimated_variance'], result['actual_variance'] + result['heterogeneity'])
                self.assertGreater(result['heterogeneity'], 0)
            self.assertEqual(result['target'], 0)

    def test_arbitrary_common_drift_and_constant_effect(self):
        population = [[[F(slot), F(slot)*F(97, 100), F(slot)*F(9, 10)] for slot in positions]
                      for positions in [(9, 11, 14), (40, 50, 70), (22, 18, 25)]]
        ratio = population_ratio(population)
        self.assertEqual(ratio, F(97, 100))
        result = enumerate_moments(population, ratio)
        self.assertEqual(result['target'], 0)
        self.assertEqual(result['heterogeneity'], 0)
        self.assertGreater(result['actual_variance'], 0)
        self.assertEqual(result['actual_variance'], result['expected_estimated_variance'])

    def test_all_six_permutations_give_uniform_role_positions(self):
        counts = Counter((role, position) for order in ORDERS for position, role in enumerate(order))
        self.assertEqual(counts, {(r, p): 2 for r in ROLES for p in range(3)})
        table = [[11, 21, 31], [12, 22, 32], [13, 23, 33]]
        for role in ROLES:
            self.assertEqual(sum(observed(table, o)[role] for o in ORDERS)/6, sum(F(row[ROLES.index(role)]) for row in table)/3)

    def test_manifest_draws_are_consumed_once_without_rebalancing(self):
        rng = random.Random('randomized-schedule-unit-test')
        draws = [rng.randrange(6) for _ in range(600)]
        rows = assignment_schedule(draws, 'aa')
        self.assertEqual(len(rows), 1800)
        self.assertEqual(len({row['name'] for row in rows}), 1800)
        for k in range(0, len(rows), 3):
            cohort = rows[k:k+3]
            self.assertEqual(len({r['draw_index'] for r in cohort}), 1)
            self.assertEqual(tuple(r['role'] for r in cohort), ORDERS[draws[cohort[0]['draw_index']]])
        self.assertEqual(Counter(r['draw_index'] for r in rows), {i: 3 for i in range(600)})
        for case in CASES:
            for policy in POLICIES:
                cohort_rows = [r for r in rows if r['case'] == case and r['policy'] == policy]
                self.assertEqual(Counter(r['role'] for r in cohort_rows), {r: 60 for r in ROLES})
                positions = Counter((i % 30)//3 for i, r in enumerate(rows) if r['case'] == case and r['policy'] == policy)
                self.assertEqual(positions, {i: 18 for i in range(10)})
        # A very imbalanced realization remains accepted; choosing favorable
        # balance would silently change the randomization distribution.
        imbalanced = assignment_schedule([0]*600, 'compare')
        self.assertTrue(all(r['position'] == 0 for r in imbalanced if r['role'] == 'A'))

    def test_approximate_interval_is_fieller_inversion(self):
        x, y = [100, 110, 101, 109], [98, 105, 99, 104]
        interval = fieller(y, x, student_critical(.95, 3))
        self.assertTrue(interval['bounded'])
        self.assertLess(interval['interval'][0], interval['ratio'])
        self.assertGreater(interval['interval'][1], interval['ratio'])

    def test_refuse_malformed_populations_and_draws(self):
        for population in [[], [[[1, 2, 3]]]*2, [[[1, 2, 3]]*3], [[[1, False, 3]]*3]*2, [[[1., 2, 3]]*3]*2]:
            with self.assertRaises(ValueError):
                population_ratio(population)
        for draws in [[0]*599, [0]*601, [True]*600, [6]*600, [-1]*600, [1.5]*600]:
            with self.assertRaises(ValueError):
                assignment_schedule(draws, 'aa')
        with self.assertRaises(ValueError):
            assignment_schedule([0]*600, 'invalid')
        with self.assertRaises(ValueError):
            observed([[1, 2, 3]]*3, ('A', 'A', 'N'))


if __name__ == '__main__':
    unittest.main()
