"""Reject lost failures, numerical weakening and mismatched experimental products."""
import copy
import unittest
from protocol import read
from prerequisites import eligibility, verify_comparisons
from prepare import PRIOR


class Prerequisites(unittest.TestCase):
    def fixture(self):
        reports = {name:read(folder/'analysis.json') for name,folder in PRIOR.items() if name != 'models'}
        current, candidate = reports['qualified']['built'], reports['contracts']['product']
        reports['models'] = dict(identities=dict(selected=current,candidate=candidate))
        spec = dict(identities=dict(current=current,candidate=candidate), release_admitted=False,
            failed_component_controls=[r for r in reports['screen']['controls'] if not r['passed']])
        return copy.deepcopy(reports), copy.deepcopy(spec)

    def test_actual_fixed_candidate_allows_application_testing_only(self):
        eligibility(*self.fixture())

    def test_removed_failed_repeatability_control_is_rejected(self):
        reports, spec = self.fixture(); spec['failed_component_controls'].pop()
        with self.assertRaises(AssertionError): eligibility(reports, spec)

    def test_changed_product_is_rejected(self):
        reports, spec = self.fixture()
        spec['identities']['candidate']['Lokad.Onnx.Data.dll']['sha256'] = '0'*64
        with self.assertRaises(AssertionError): eligibility(reports, spec)

    def test_reinterpreted_weighted_gate_is_rejected(self):
        reports, spec = self.fixture(); reports['screen']['gates'][0]['passed'] = True
        with self.assertRaises(AssertionError): eligibility(reports, spec)

    def test_reinterpreted_diagnostic_admission_is_rejected(self):
        reports, spec = self.fixture(); reports['memory']['admitted'] = True
        with self.assertRaises(AssertionError): eligibility(reports, spec)

    def test_failed_focused_contract_is_rejected(self):
        reports, spec = self.fixture(); reports['contracts']['suites'][0]['sweep']['passed'] = False
        with self.assertRaises(AssertionError): eligibility(reports, spec)

    def rows(self):
        return [dict(case=str(i),label='encoder',output='x',values=3090494 if i == 0 else 0,
                     shape=[3090494 if i == 0 else 0],dtype='Float',maximum_scaled_error=1e-5,
                     bit_identical=False) for i in range(784)]

    def test_float_bound_does_not_require_identical_bits(self):
        verify_comparisons(self.rows())

    def test_float_bound_cannot_be_weakened(self):
        rows = self.rows(); rows[0]['maximum_scaled_error'] = 1.01e-4
        with self.assertRaises(AssertionError): verify_comparisons(rows)

    def test_integers_must_be_exact(self):
        rows = self.rows(); rows[0]['dtype'] = 'Int64'
        with self.assertRaises(AssertionError): verify_comparisons(rows)

    def test_duplicate_comparison_is_rejected(self):
        rows = self.rows(); rows[-1] = rows[-2]
        with self.assertRaises(AssertionError): verify_comparisons(rows)


if __name__ == '__main__': unittest.main()
