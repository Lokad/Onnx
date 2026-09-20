import unittest
from shared import *
from calculation import ideal_decimal, ideal_double, calculate, scalar_calculate

class Coefficients(unittest.TestCase):
    def test_formula_and_all_value_scalar_control(self):
        decimal = ideal_decimal(); double = ideal_double()
        self.assertLessEqual(float(np.max(np.abs(decimal - double))), 1e-12)
        self.assertEqual(decimal.shape, (80, 256)); self.assertTrue(np.all((decimal >= 0) & (decimal <= 1)))
        np.testing.assert_array_equal(decimal[:, 0], 0.)
        self.assertTrue(np.all(np.count_nonzero(decimal, axis=1) > 0))
        for scale in (0., 1e-20, 1., 1e8):
            power = np.random.default_rng(20260920).uniform(0, scale, (7, 257)) if scale else np.zeros((7, 257))
            actual = calculate(power, decimal); expected = scalar_calculate(power, decimal)
            for stage in actual:
                self.assertEqual(metric(actual[stage], expected[stage], CONTROL_LIMIT)['failed'], 0, (scale, stage))
            self.assertLess(float(np.abs(actual['features'].mean(axis=1)).max()), 1e-12)

    def test_invalid_inputs(self):
        weights = ideal_double(); power = np.ones((2, 257), dtype=np.float64)
        for a, b in [(power[:, :-1], weights), (power, weights[:, :-1]), (power.astype(np.float32), weights),
                     (-power, weights), (power, -weights), (power * np.nan, weights), (power, weights * np.nan)]:
            with self.assertRaises(AssertionError): calculate(a, b)

if __name__ == '__main__':
    psutil_module().Process().cpu_affinity([0]); unittest.main()
