"""Keep every semantic field exact and bound all public centroid values."""
import copy
import unittest
from checks import centroid_agreement


class Centroids(unittest.TestCase):
    def fixture(self):
        return dict(Intervals=[], ExclusiveIntervals=[], Speakers=[dict(Speaker=0, Centroid=[.1]*256, HasEmbedding=True)],
                    Status=0, AudioDuration=1., Windows=1)

    def test_small_difference_is_retained(self):
        a = self.fixture(); b = copy.deepcopy(a); a['Speakers'][0]['Centroid'][0] += 1e-6
        result = centroid_agreement(a, b)
        self.assertFalse(result[0]['bit_identical']); self.assertGreater(result[0]['maximum'], 0)

    def test_large_difference_is_rejected(self):
        a = self.fixture(); b = copy.deepcopy(a); a['Speakers'][0]['Centroid'][0] += 1e-3
        with self.assertRaises(AssertionError): centroid_agreement(a, b)

    def test_nonfinite_or_missing_value_is_rejected(self):
        for values in [[float('nan')]*256, [.1]*255]:
            a = self.fixture(); b = copy.deepcopy(a); a['Speakers'][0]['Centroid'] = values
            with self.assertRaises(AssertionError): centroid_agreement(a, b)

    def test_semantics_cannot_be_relaxed_by_numeric_bound(self):
        a = self.fixture(); b = copy.deepcopy(a); a['AudioDuration'] += 1e-12
        with self.assertRaises(AssertionError): centroid_agreement(a, b)


if __name__ == '__main__': unittest.main()
