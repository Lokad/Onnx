from fractions import Fraction
import copy
import unittest
from admission import evaluate


def pair(value, second=None):
    value = Fraction(value); second = value if second is None else Fraction(second)
    def item(v): return dict(mean=float(v), exact_mean=dict(numerator=v.numerator, denominator=v.denominator))
    return dict(seconds=float((value+second)/2), **{'exact_mean': item((value+second)/2)['exact_mean']}, processes=[item(value), item(second)])


def fixture():
    return [dict(name=str(i), audio_seconds=213.265 if i == 20 else 10, is_corpus=i == 20,
        production=pair(10), portable=pair(9), ort=pair(6)) for i in range(21)]


class AdmissionTests(unittest.TestCase):
    def test_requires_every_control_and_fixture(self):
        rows = fixture(); result = evaluate(rows)
        self.assertTrue(result['admitted']); self.assertEqual(len(result['controls']), 63); self.assertEqual(len(result['gains']), 21)
        for index in range(21):
            for role in ('production', 'portable', 'ort'):
                damaged = copy.deepcopy(rows); damaged[index][role] = pair(10, Fraction(121, 10))
                self.assertFalse(evaluate(damaged)['controls_passed'])
                self.assertFalse(evaluate(damaged)['admitted'])

    def test_every_speed_gate_exact_boundary(self):
        for index in range(21):
            rows = fixture(); boundary = Fraction(95 if index == 20 else 105, 10)
            rows[index]['portable'] = pair(boundary)
            self.assertTrue(evaluate(rows)['speed_threshold_passed'])
            rows[index]['portable'] = pair(boundary + Fraction(1, 10**18))
            self.assertFalse(evaluate(rows)['speed_threshold_passed'])

    def test_every_control_exact_boundary(self):
        for index in range(21):
            for role in ('production', 'portable', 'ort'):
                rows = fixture(); boundary = Fraction(110 if index == 20 else 120, 100)
                rows[index][role] = pair(1, boundary)
                self.assertTrue(evaluate(rows)['controls_passed'])
                rows[index][role] = pair(1, boundary + Fraction(1, 10**18))
                self.assertFalse(evaluate(rows)['controls_passed'])

    def test_missing_corpus_clip_duplicate_and_process_refused(self):
        rows = fixture()
        for damaged in [rows[1:], rows[:-1]]:
            with self.assertRaises(AssertionError): evaluate(damaged)
        rows[0]['ort']['processes'].pop()
        with self.assertRaises(AssertionError): evaluate(rows)
        rows = fixture(); rows[0]['name'] = rows[1]['name']
        with self.assertRaises(AssertionError): evaluate(rows)
