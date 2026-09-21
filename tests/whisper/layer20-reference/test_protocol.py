import unittest
from protocol import decompose, metric, np


class DecompositionTests(unittest.TestCase):
    def test_inherited_and_local_signed_components(self):
        ideal = np.array([0., 2., -3.])
        inherited = np.array([.25, -.5, .75])
        local = np.array([-.25, 1., -.5])
        result = decompose(ideal+inherited+local, ideal+inherited, ideal)
        self.assertEqual(result['total']['max_scaled'], .25)
        self.assertEqual(result['inherited']['max_scaled'], .25)
        self.assertEqual(result['local']['max_scaled'], .5)
        self.assertEqual(result['signed_cross_term'], -1.875)
        self.assertEqual(result['closure_max'], 0)

    def test_exact_reference_and_own_input_are_distinct(self):
        ideal = np.array([0., 10.])
        own = ideal+np.array([.001, .002])
        result = decompose(own, own, ideal)
        self.assertEqual(result['local']['failed_values'], 0)
        self.assertEqual(result['local_own']['l2'], 0)
        self.assertEqual(result['total']['failed_values'], 2)

    def test_refuse_incompatible_and_nonfinite_arrays(self):
        with self.assertRaises(AssertionError):
            decompose(np.ones(3), np.ones(2), np.ones(3))
        with self.assertRaises(AssertionError):
            metric(np.array([np.nan]), np.ones(1))
        with self.assertRaises(AssertionError):
            metric(np.ones(1), np.zeros(1))


if __name__ == '__main__':
    unittest.main()
