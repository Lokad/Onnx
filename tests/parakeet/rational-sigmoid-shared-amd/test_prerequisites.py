"""Reject broken product bindings and numerical differences outside the contract."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from protocol import pin, read
from prepare import MODELS, PREVIOUS, OLD, APP
from prerequisites import validate
from cross_numeric import compare


class Prerequisites(unittest.TestCase):
    def fixture(self):
        return [read(MODELS/'collected/evidence/compatibility.json'), read(MODELS/'analysis.json'),
                read(PREVIOUS/'analysis.json'), read(APP/'analysis.json'), pin(OLD/'runtimes/baseline/Replay.dll')]

    def test_closed_application_and_actual_consumer(self): validate(*self.fixture())

    def test_different_consumer_is_rejected(self):
        args = self.fixture(); args[-1]['sha256'] = '0'*64
        with self.assertRaises(AssertionError): validate(*args)

    def test_failed_application_control_is_rejected(self):
        args = self.fixture(); args[3]['performance']['controls'][0]['passed'] = False
        with self.assertRaises(AssertionError): validate(*args)

    def test_wrong_current_product_is_rejected(self):
        args = self.fixture(); args[3]['identities']['current']['Lokad.Onnx.dll']['sha256'] = '0'*64
        with self.assertRaises(AssertionError): validate(*args)

    def test_removed_old_binding_is_rejected(self):
        args = self.fixture(); args[0]['original_public_bindings_preserved'] = False
        with self.assertRaises(AssertionError): validate(*args)

    def test_component_failure_cannot_disappear(self):
        args = self.fixture(); args[0]['failed_component_cases'].pop()
        with self.assertRaises(AssertionError): validate(*args)

    def test_unreviewed_method_change_is_rejected(self):
        args = self.fixture(); args[0]['changed_core_methods'].append('MatMul')
        with self.assertRaises(AssertionError): validate(*args)


class Numerical(unittest.TestCase):
    def compare(self, candidate, current):
        with tempfile.TemporaryDirectory() as folder:
            a, b = [Path(folder)/n for n in ['candidate.f32', 'current.f32']]
            np.asarray(candidate, dtype='<f4').tofile(a); np.asarray(current, dtype='<f4').tofile(b)
            return compare(a, b)

    def test_bounded_change_is_recorded(self):
        result = self.compare([1.00001, -2.], [1., -2.])
        self.assertFalse(result['bit_identical']); self.assertGreater(result['maximum_scaled_error'], 0)

    def test_native_scale_is_preserved(self):
        result = self.compare([100.005], [100.])
        self.assertLess(result['maximum_scaled_error'], 1e-4)

    def test_excess_difference_is_rejected(self):
        with self.assertRaises(AssertionError): self.compare([1.001], [1.])

    def test_nonfinite_is_rejected(self):
        for candidate, current in [([float('nan')], [0.]), ([0.], [float('inf')])]:
            with self.assertRaises(AssertionError): self.compare(candidate, current)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(AssertionError): self.compare([1., 2.], [1.])

    def test_exact_and_empty_are_recorded(self):
        for values in [[], [1., -2., 0.]]:
            result = self.compare(values, values)
            self.assertTrue(result['bit_identical']); self.assertEqual(result['maximum_scaled_error'], 0)


if __name__ == '__main__': unittest.main()
