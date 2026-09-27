"""Reject mismatched products, missing prerequisites and changed consumer scope."""
import unittest
from prepare import PRIOR, PRODUCT, SCOPE, read
from prerequisites import validate


class Prerequisites(unittest.TestCase):
    def fixture(self):
        return [{name: read(folder/'analysis.json') for name, folder in PRIOR.items()},
                read(PRODUCT/'collected/evidence/compatibility.json'), read(SCOPE)]

    def test_complete_closed_chain(self): validate(*self.fixture())

    def test_wrong_candidate(self):
        args = self.fixture(); args[0]['models']['identities']['candidate']['Lokad.Onnx.dll']['sha256'] = '0'*64
        with self.assertRaises(AssertionError): validate(*args)

    def test_wrong_consumer(self):
        args = self.fixture(); args[0]['graph']['short_consumer']['consumer']['sha256'] = '0'*64
        with self.assertRaises(AssertionError): validate(*args)

    def test_missing_application_control(self):
        args = self.fixture(); args[0]['app']['performance']['controls'].pop()
        with self.assertRaises(AssertionError): validate(*args)

    def test_failed_prior_graph(self):
        args = self.fixture(); args[0]['graph']['performance'][0]['qualified'] = False
        with self.assertRaises(AssertionError): validate(*args)

    def test_wrong_previously_qualified_core(self):
        args = self.fixture(); args[1]['qualified_model_product']['Lokad.Onnx.dll']['sha256'] = '0'*64
        with self.assertRaises(AssertionError): validate(*args)

    def test_lost_method_flags(self):
        args = self.fixture(); args[1]['all_original_method_flags_preserved'] = False
        with self.assertRaises(AssertionError): validate(*args)

    def test_component_failure_cannot_disappear(self):
        args = self.fixture(); args[1]['failed_component_cases'].pop()
        with self.assertRaises(AssertionError): validate(*args)

    def test_changed_public_pyannote_results(self):
        args = self.fixture(); args[0]['pyannote']['results']['candidate']['complete_public_results_exact'] = False
        with self.assertRaises(AssertionError): validate(*args)

    def test_missing_case_invalidates_export_census(self):
        args = self.fixture(); args[2]['models'][0]['cases'].pop()
        with self.assertRaises(AssertionError): validate(*args)


if __name__ == '__main__': unittest.main()
