import copy
import math
import unittest
from evaluate import evaluate, schedule, CASES, ROLES


def fixture():
    workers = []
    for row in schedule():
        ms = 98.0 if row['name'] == 'e5-30pad128' and row['role'] == 'candidate' else 100.0
        workers.append(row | {b: dict(samples_ms=[ms] * 33) for b in ('execute', 'request')})
    return workers, dict(maximum_foreign_cpu_fraction=0., maximum_steal_fraction=0.)


class EvaluationTests(unittest.TestCase):
    def test_balanced_schedule_and_known_result(self):
        workers, health = fixture()
        for name in CASES:
            groups = [[w for w in workers if w['name'] == name and w['visit'] == v] for v in range(6)]
            self.assertEqual(6, len({tuple(w['role'] for w in g) for g in groups}))
            for role in ROLES:
                self.assertEqual([0, 0, 1, 1, 2, 2], sorted(w['position'] for w in workers if w['name'] == name and w['role'] == role))
        value = evaluate(workers, health)
        self.assertTrue(value['passed'])
        self.assertAlmostEqual(.98, value['cases']['e5-30pad128']['execute']['candidate_ratio'])

    def test_malformed_data_refused(self):
        mutations = [lambda w, h: w.pop(), lambda w, h: w.reverse(),
                     lambda w, h: w[0].update(role='candidate'),
                     lambda w, h: w[0]['execute']['samples_ms'].pop(),
                     lambda w, h: h.update(maximum_foreign_cpu_fraction=-1),
                     lambda w, h: h.update(maximum_steal_fraction=math.nan)]
        for bad in (0, -1, math.nan, math.inf, True, '100'):
            mutations.append(lambda w, h, bad=bad: w[0]['request']['samples_ms'].__setitem__(0, bad))
        for mutation in mutations:
            workers, health = fixture()
            mutation(workers, health)
            with self.assertRaises(ValueError):
                evaluate(workers, health)

    def test_every_declared_criterion_can_fail(self):
        # Each injection has enough gain elsewhere to distinguish visit and aggregate limits.
        cases = [('controlB', 'e5-8tok', None, 102., 'control_passed'),
                 ('controlB', 'e5-8tok', 0, 104., 'control_passed'),
                 ('candidate', 'e5-30pad128', None, 99.5, 'candidate_passed'),
                 ('candidate', 'e5-30pad128', 0, 103., 'candidate_passed'),
                 ('candidate', 'e5-30tok', None, 103., 'candidate_passed'),
                 ('candidate', 'e5-128tok', 0, 106., 'candidate_passed'),
                 ('candidate', 'e5-512tok', None, 103., 'candidate_passed')]
        for boundary in ('execute', 'request'):
            for role, name, visit, ms, key in cases:
                workers, health = fixture()
                for worker in workers:
                    if worker['role'] == role and worker['name'] == name and (visit is None or worker['visit'] == visit):
                        worker[boundary]['samples_ms'] = [ms] * 33
                value = evaluate(workers, health)
                self.assertFalse(value[key], (boundary, role, name, visit))
                self.assertFalse(value['passed'])
        for key, value in [('maximum_foreign_cpu_fraction', .021), ('maximum_steal_fraction', .006)]:
            workers, health = fixture(); health[key] = value
            result = evaluate(workers, health)
            self.assertFalse(result['passed']); self.assertFalse(all(result['health'].values()))

    def test_all_samples_used_and_arithmetic_control_mean(self):
        workers, health = fixture()
        for worker in workers:
            ms = {'controlA': 90., 'controlB': 110., 'candidate': 98.}[worker['role']]
            worker['execute']['samples_ms'] = [ms] * 32 + [ms * 2]
        result = evaluate(workers, health)
        row = result['cases']['e5-30pad128']['execute']
        self.assertAlmostEqual(98. * 34 / 33, row['mean_ms']['candidate'])
        self.assertAlmostEqual(.98, row['candidate_ratio'])
        self.assertFalse(result['control_passed'])


if __name__ == '__main__':
    unittest.main()
