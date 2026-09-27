"""The addressing-only candidate must retain exact tensors and public centroids."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch
import checks


class ExactResults(unittest.TestCase):
    def fixture(self):
        value = dict(Intervals=[], ExclusiveIntervals=[],
                     Speakers=[dict(Speaker=0, Centroid=[.1]*256, HasEmbedding=True)],
                     Status=0, AudioDuration=1., Windows=1)
        selected = dict(applications=[dict(result=copy.deepcopy(value)) for _ in range(16)])
        result = dict(passed=True, arrays=18, values=2917107, public_calls=16,
                      comparisons=[dict(reference='production', bit_identical=True) for _ in range(18)])
        return result, selected, copy.deepcopy(selected)

    def run_check(self, result, selected, candidate):
        with patch.object(checks, 'pyannote', return_value=result), patch.object(
                checks, 'read', side_effect=[candidate, selected]):
            return checks.model(Path('fixture'), 'candidate')

    def test_exact_outputs_pass(self):
        self.assertTrue(self.run_check(*self.fixture())['complete_public_results_exact'])

    def test_small_centroid_change_fails(self):
        result, selected, candidate = self.fixture()
        candidate['applications'][0]['result']['Speakers'][0]['Centroid'][0] += 1e-12
        with self.assertRaisesRegex(AssertionError, 'centroid'):
            self.run_check(result, selected, candidate)

    def test_nonidentical_tensor_fails(self):
        result, selected, candidate = self.fixture()
        result['comparisons'][0]['bit_identical'] = False
        with self.assertRaises(AssertionError): self.run_check(result, selected, candidate)

    def test_missing_comparison_fails(self):
        result, selected, candidate = self.fixture()
        result['comparisons'].pop()
        with self.assertRaises(AssertionError): self.run_check(result, selected, candidate)


if __name__ == '__main__': unittest.main()
