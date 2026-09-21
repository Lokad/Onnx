from fractions import Fraction as F
import copy
import unittest

from contract import timing, worker_specification
from design import assignment_schedule


def fixture(phase):
    jobs = assignment_schedule([i % 6 for i in range(600)], phase)
    values = []
    for job in jobs:
        specification = worker_specification(job, phase)
        scale = F(97, 100) if phase == 'compare' and job['role'] == 'C' else F(96, 100) if job['role'] == 'N' else F(1)
        ticks = int((10000000+10000*job['cohort'])*scale)
        values.append(dict(specification=specification, frequency=1000000000,
                           measured=[dict(execute=ticks, request=ticks+1000)]*(specification['blocks']*specification['calls'])))
    return jobs, values


class ContractTests(unittest.TestCase):
    def test_complete_aa_screen(self):
        jobs, values = fixture('aa')
        result = timing(values, jobs, 'aa')
        self.assertTrue(result['statistical_screen'])
        self.assertEqual(result['contrast_count'], 20)
        self.assertFalse(result['ready_for_promotion'])
        self.assertEqual(len(result['results']), 20)

    def test_candidate_ratios_and_exact_zero_contrast_variance(self):
        jobs, values = fixture('compare')
        result = timing(values, jobs, 'compare')
        self.assertTrue(result['statistical_screen'])
        self.assertEqual(result['contrast_count'], 60)
        for row in result['results']:
            if row['boundary'] == 'execute':
                ratio = row['contrasts']['C/A']
                self.assertEqual(ratio['ratio'], .97)
                self.assertEqual(ratio['interval'], [.97, .97])
                self.assertEqual(ratio['largest_observed_variance_share'], 0)

    def test_missing_duplicate_and_changed_specification_refuse(self):
        jobs, values = fixture('aa')
        with self.assertRaises(AssertionError):
            timing(values[:-1], jobs, 'aa')
        duplicated = jobs[:]; duplicated[1] = duplicated[0]
        with self.assertRaises(AssertionError):
            timing(values, duplicated, 'aa')
        changed = copy.deepcopy(values[0]); changed['specification']['cohort'] = 59
        with self.assertRaises(AssertionError):
            timing([changed]+values[1:], jobs, 'aa')


if __name__ == '__main__':
    unittest.main()
