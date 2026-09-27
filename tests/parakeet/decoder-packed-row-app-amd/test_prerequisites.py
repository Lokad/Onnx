"""Reject substituted products or reinterpretation of the failed component results."""
import copy
import unittest
from protocol import read
from prerequisites import eligibility
from prepare import PRIOR


class Prerequisites(unittest.TestCase):
    def fixture(self):
        reports = {name: read(folder/'analysis.json') for name, folder in PRIOR.items() if name != 'models'}
        current = reports['qualified']['built']
        candidate = dict(current, **{'Lokad.Onnx.dll': reports['contracts']['products']['candidate']})
        reports['models'] = dict(identities=dict(selected=current, candidate=candidate))
        spec = dict(identities=dict(current=current, candidate=candidate), release_admitted=False,
            failed_component_controls=[r for r in reports['screen']['controls'] if not r['passed']])
        return copy.deepcopy(reports), copy.deepcopy(spec)

    def test_actual_fixed_pair_allows_application_test_only(self):
        eligibility(*self.fixture())

    def test_lost_failed_control(self):
        reports, spec = self.fixture(); spec['failed_component_controls'].pop()
        with self.assertRaises(AssertionError): eligibility(reports, spec)

    def test_changed_product(self):
        reports, spec = self.fixture(); spec['identities']['candidate']['Lokad.Onnx.Data.dll']['sha256'] = '0'*64
        with self.assertRaises(AssertionError): eligibility(reports, spec)

    def test_reinterpreted_screen(self):
        reports, spec = self.fixture(); reports['screen']['admitted'] = True
        with self.assertRaises(AssertionError): eligibility(reports, spec)

    def test_reinterpreted_diagnostic(self):
        reports, spec = self.fixture(); reports['diagnosis']['first_call_explanation_supported'] = True
        with self.assertRaises(AssertionError): eligibility(reports, spec)

    def test_unrelated_method_changed(self):
        reports, spec = self.fixture(); reports['contracts']['compiled']['assemblies'][1]['changed'].append('Unexpected')
        with self.assertRaises(AssertionError): eligibility(reports, spec)


if __name__ == '__main__': unittest.main()
