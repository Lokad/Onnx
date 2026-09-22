import copy
import unittest
from admission import evaluate


def fixture():
    return [dict(name=str(i), audio_seconds=seconds, **{role: dict(seconds=value,
        processes=[dict(mean=value), dict(mean=value)]) for role, value in
        [('production', 10), ('portable', 9), ('rows', 8), ('ort', 6)]}) for i, seconds in enumerate([30, 10, 10, 10])]


class AdmissionTests(unittest.TestCase):
    def test_requires_every_control_and_fixture(self):
        rows = fixture()
        self.assertTrue(evaluate(rows)['admitted'])
        for index in range(4):
            for role in ('production', 'portable', 'rows', 'ort'):
                damaged = copy.deepcopy(rows)
                damaged[index][role]['processes'][1]['mean'] *= 1.21
                self.assertFalse(evaluate(damaged)['controls_passed'])
                self.assertFalse(evaluate(damaged)['admitted'])

    def test_small_gain_and_crop_regression_refuse_admission(self):
        rows = fixture()
        rows[0]['rows']['seconds'] = 8.9
        self.assertTrue(evaluate(rows)['controls_passed'])
        self.assertFalse(evaluate(rows)['admitted'])
        rows = fixture()
        rows[3]['rows']['seconds'] = 9.5
        self.assertFalse(evaluate(rows)['admitted'])

    def test_missing_long_case_and_truncated_processes_refused(self):
        rows = fixture()
        with self.assertRaises(AssertionError):
            evaluate(rows[1:])
        rows[0]['ort']['processes'].pop()
        with self.assertRaises(AssertionError):
            evaluate(rows)
