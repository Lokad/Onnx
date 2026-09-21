"""Small numerical examples with independent exact/scalar expectations."""
import math
import unittest
from protocol import *
from onnx import helper


class PrecisionTests(unittest.TestCase):
    def test_cancellation_and_fp32_boundary(self):
        node = helper.make_node('MatMul', ['a', 'b'], ['y'])
        # Exercise the matrix kernel: the one-row three-term dot on this BLAS
        # already preserves the small term, despite ordinary float32 inputs.
        a = np.ones((2, 17), dtype=np.float32)
        a[:, 0] = 2**25; a[:, -1] = -(2**25)
        b = np.ones((17, 2), dtype=np.float32)
        expected = math.fsum(float(x) for x in a[0])
        wide = calculate(node, [a, b], 'wide-matmul')
        self.assertEqual(wide.dtype, np.float32)
        self.assertEqual(float(wide[0, 0]), expected)
        self.assertNotEqual(float(calculate(node, [a, b], 'float32')[0, 0]), expected)
        self.assertEqual(len(scalar_dots(a, b, wide)), 3)

    def test_broadcast_scalar_check(self):
        a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)/7
        b = np.arange(20, dtype=np.float32).reshape(1, 4, 5)/11
        node = helper.make_node('MatMul', ['a', 'b'], ['y'])
        self.assertEqual(len(scalar_dots(a, b, calculate(node, [a, b], 'wide-matmul'))), 3)

    def test_stable_softmax_and_explicit_mean(self):
        x = np.array([[10000., 10000.], [-10000., -10000.]], dtype=np.float32)
        softmax = helper.make_node('Softmax', ['x'], ['y'], axis=-1)
        mean = helper.make_node('ReduceMean', ['x'], ['y'], axes=[1], keepdims=1)
        np.testing.assert_array_equal(calculate(softmax, [x], 'float32'), np.full_like(x, .5))
        np.testing.assert_array_equal(calculate(mean, [x], 'wide-matmul'), x[:, :1])

    def test_bad_mode_nonfinite_and_independent_metrics(self):
        node = helper.make_node('Add', ['a', 'b'], ['y'])
        x = np.array([1.], dtype=np.float32)
        with self.assertRaises(ValueError):
            calculate(node, [x, x], 'unknown')
        with self.assertRaises(AssertionError):
            calculate(node, [x, np.array([float('inf')], dtype=np.float32)], 'float32')
        actual = np.arange(100001, dtype=np.float32)/300.
        reference = actual.astype(np.float64)+.0002
        a, b = metric(actual, reference), metric(actual, reference, True)
        self.assertEqual(a['max_scaled'], b['max_scaled'])
        self.assertEqual(a['failed_values'], b['failed_values'])
        self.assertAlmostEqual(a['squared_error'], b['squared_error'], places=13)


if __name__ == '__main__':
    unittest.main()
