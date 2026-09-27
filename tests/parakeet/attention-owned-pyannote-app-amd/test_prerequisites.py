"""Reject mixed products, missing numeric coverage and wrong census provenance."""
import unittest
from prepare import PRIOR, COMPATIBLE, QUALIFIED, ROOT
from protocol import read
from prerequisites import validate
from graph_prerequisite import validate as validate_graph


class Prerequisites(unittest.TestCase):
    def fixture(self):
        reports = {key: read(folder/'analysis.json') for key, folder in PRIOR.items()}
        spec = dict(identities=reports['models']['identities'], consumers=reports['baseline']['consumers'])
        return reports, spec, read(COMPATIBLE), read(QUALIFIED/'analysis.json')

    def test_actual_closed_pair(self):
        self.assertTrue(validate(*self.fixture()))

    def test_wrong_data_identity(self):
        args = self.fixture(); args[1]['identities']['candidate']['Lokad.Onnx.Data.dll']['sha256'] = '0'*64
        with self.assertRaises(AssertionError): validate(*args)

    def test_missing_native_comparison(self):
        args = self.fixture(); args[0]['parakeet']['results']['candidate-native-512']['native']['exact_selected_comparisons'].pop()
        with self.assertRaises(AssertionError): validate(*args)

    def test_native_excess_error(self):
        args = self.fixture(); args[0]['parakeet']['results']['candidate-native-256']['native']['maximum'] = .001
        with self.assertRaises(AssertionError): validate(*args)

    def test_array_byte_change(self):
        args = self.fixture()
        rows = args[0]['parakeet']['results']['candidate-native-512']['native']['exact_selected_comparisons']
        rows[0]['bit_identical'] = False
        with self.assertRaises(AssertionError): validate(*args)

    def test_changed_public_results(self):
        for family, key in [('parakeet','candidate-public-256'), ('models','candidate')]:
            args = self.fixture()
            field = 'complete_selected_results_exact' if family == 'parakeet' else 'complete_public_results_exact'
            args[0][family]['results'][key][field] = False
            with self.assertRaises(AssertionError): validate(*args)

    def test_changed_consumer(self):
        args = self.fixture(); args[1]['consumers']['NaturalMeetings']['sha256'] = '0'*64
        with self.assertRaises(AssertionError): validate(*args)

    def test_failed_application_gate(self):
        args = self.fixture(); args[0]['parakeet-app']['performance']['gates'][0]['passed'] = False
        with self.assertRaises(AssertionError): validate(*args)

    def test_changed_bindings(self):
        for field in ['original_public_bindings_preserved', 'all_original_method_flags_preserved', 'all_data_methods_exact']:
            args = self.fixture(); args[2][field] = False
            with self.assertRaises(AssertionError): validate(*args)

    def test_wrong_census(self):
        args = self.fixture(); args[2]['actual_model_census']['sha256'] = '0'*64
        with self.assertRaises(AssertionError): validate(*args)

    def test_old_graph_cannot_qualify_new_products(self):
        folder = ROOT/'artifacts/parakeet-pad-current-graphs-v2-amd-20260926'
        pair = self.fixture()[1]['identities']
        products = {role: {'Lokad.Onnx.dll': pair[key]['Lokad.Onnx.dll']} for role, key in [('current','selected'),('candidate','candidate')]}
        with self.assertRaises(AssertionError):
            validate_graph(read(folder/'analysis.json'), read(folder/'closed.json'), products)


if __name__ == '__main__': unittest.main()
