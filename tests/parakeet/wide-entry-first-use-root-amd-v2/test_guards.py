"""Reject unrelated source edits and cross-campaign binary substitution before mutation."""
import copy
import unittest

from admission import product_identities
from source_scope import CHANGED, delta


class IntegrationGuards(unittest.TestCase):
    def source(self):
        before = {f'unchanged/{i}': {'sha256': str(i)} for i in range(419)}
        before[CHANGED[0]] = {'sha256': 'selected'}
        after = copy.deepcopy(before)
        for name in CHANGED:
            after[name] = {'sha256': 'candidate'}
        return dict(before=before, source=after)

    def identities(self):
        selected = {name: dict(bytes=100, sha256='selected-' + name)
                    for name in ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll']}
        candidate = {name: dict(bytes=100, sha256='candidate-' + name) for name in selected}
        return [dict(built=candidate), dict(measured=selected),
                dict(identities=dict(current=selected, candidate=candidate)),
                dict(identities=dict(selected=selected, candidate=candidate)),
                dict(products={role: {'Lokad.Onnx.dll': value['Lokad.Onnx.dll']}
                               for role, value in [('current', selected), ('candidate', candidate)]})]

    def test_exact_source_delta(self):
        self.assertEqual(delta(self.source()), CHANGED)

    def test_unrelated_edit_with_unchanged_counts_rejected(self):
        value = self.source()
        value['source']['unchanged/17'] = {'sha256': 'unrelated'}
        with self.assertRaises(AssertionError):
            delta(value)

    def test_replaced_input_with_unchanged_counts_rejected(self):
        value = self.source()
        value['source']['unplanned/new-file'] = value['source'].pop('unchanged/17')
        with self.assertRaises(AssertionError):
            delta(value)

    def test_missing_addition_rejected(self):
        value = self.source()
        value['source'].pop(CHANGED[2])
        with self.assertRaises(AssertionError):
            delta(value)

    def test_matching_binary_campaigns(self):
        self.assertTrue(product_identities(*self.identities())['passed'])

    def test_each_audio_role_and_assembly_substitution_rejected(self):
        for position, roles in [(2, ['current', 'candidate']), (3, ['selected', 'candidate'])]:
            for role in roles:
                for assembly in ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll']:
                    with self.subTest(campaign=position, role=role, assembly=assembly):
                        args = self.identities()
                        args[position] = copy.deepcopy(args[position])
                        args[position]['identities'][role][assembly]['sha256'] = 'other-build'
                        with self.assertRaises(AssertionError):
                            product_identities(*args)

    def test_each_graph_role_substitution_rejected(self):
        for role in ['current', 'candidate']:
            with self.subTest(role=role):
                args = self.identities()
                args[4] = copy.deepcopy(args[4])
                args[4]['products'][role]['Lokad.Onnx.dll']['sha256'] = 'other-build'
                with self.assertRaises(AssertionError):
                    product_identities(*args)

    def test_unexpected_product_assembly_rejected(self):
        args = self.identities()
        args[0] = copy.deepcopy(args[0])
        args[0]['built']['other.dll'] = dict(bytes=100, sha256='unexpected')
        with self.assertRaises(AssertionError):
            product_identities(*args)


if __name__ == '__main__':
    unittest.main()
