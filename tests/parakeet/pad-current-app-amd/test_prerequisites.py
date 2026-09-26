"""Reject lost component failures, wrong products and broader compiled changes."""
import copy
import unittest
from protocol import read
from prerequisites import eligibility
from prepare import PRIOR


class Prerequisites(unittest.TestCase):
    def fixture(self):
        reports = {name:read(folder/'analysis.json') for name,folder in PRIOR.items() if name != 'models'}
        current, candidate = reports['qualified']['built'], reports['contracts']['built']
        reports['models'] = dict(identities=dict(selected=current,candidate=candidate))
        spec = dict(identities=dict(current=current,candidate=candidate), release_admitted=False,
            failed_component_controls=[r for r in reports['screen']['controls'] if not r['passed']])
        return copy.deepcopy(reports), copy.deepcopy(spec)

    def test_actual_build_and_diagnostic_allow_testing_only(self):
        reports, spec = self.fixture()
        eligibility(reports, spec)

    def test_removed_component_failure_is_rejected(self):
        reports, spec = self.fixture()
        spec['failed_component_controls'].pop()
        with self.assertRaises(AssertionError): eligibility(reports, spec)

    def test_changed_data_identity_is_rejected(self):
        reports, spec = self.fixture()
        spec['identities']['candidate']['Lokad.Onnx.Data.dll']['sha256'] = '0'*64
        with self.assertRaises(AssertionError): eligibility(reports, spec)

    def test_broader_compiled_change_is_rejected(self):
        reports, spec = self.fixture()
        reports['contracts']['inventory']['core_unchanged_methods'] -= 1
        with self.assertRaises(AssertionError): eligibility(reports, spec)

    def test_reinterpreted_diagnostic_as_admission_is_rejected(self):
        reports, spec = self.fixture()
        reports['memory']['admitted'] = True
        with self.assertRaises(AssertionError): eligibility(reports, spec)

    def test_failed_focused_contract_is_rejected(self):
        reports, spec = self.fixture()
        reports['contracts']['suites']['pad-tests-256']['passed'] = 5
        with self.assertRaises(AssertionError): eligibility(reports, spec)


if __name__ == '__main__': unittest.main()
