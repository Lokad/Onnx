"""The retained consumers must reject unrelated product, body or flag drift."""
import copy
import unittest
from compatibility import CURRENT, QUALIFIED, INVENTORY, CONTRACTS, reconcile
from protocol import read


class Compatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = read(QUALIFIED/'collected/inventory/instructions.json')
        cls.packed = read(INVENTORY/'collected/inventory/instructions.json')
        cls.model = read(CURRENT/'analysis.json')['identities']['candidate']
        cls.selected = read(QUALIFIED/'analysis.json')['built']
        cls.candidate = dict(cls.selected, **{'Lokad.Onnx.dll': read(CONTRACTS/'analysis.json')['products']['candidate']})
    def check(self, root=None, packed=None, candidate=None):
        return reconcile(root or self.root, packed or self.packed, self.model, self.selected, candidate or self.candidate)
    def test_original_chain(self): self.assertTrue(self.check()['passed'])
    def test_wrong_candidate(self):
        product = copy.deepcopy(self.candidate); product['Lokad.Onnx.dll']['sha256'] = '0'*64
        with self.assertRaises(AssertionError): self.check(candidate=product)
    def test_unrelated_changed_method(self):
        value = copy.deepcopy(self.packed); value['observations'][0]['differences'][0] = 'Lokad.Onnx.Tensor`1[T]::Sigmoid::Other'
        with self.assertRaises(AssertionError): self.check(packed=value)
    def test_existing_method_flags(self):
        value = copy.deepcopy(self.packed); flags = value['observations'][0]['method_flags_after']; key = next(iter(flags)); flags[key] ^= 8
        with self.assertRaises(AssertionError): self.check(packed=value)
    def test_root_body_drift(self):
        value = copy.deepcopy(self.root); methods = value['observations'][0]['normalized_methods']; methods[next(iter(methods))] = 'changed'
        with self.assertRaises(AssertionError): self.check(root=value)
    def test_public_binding_drift(self):
        value = copy.deepcopy(self.packed); value['observations'][1]['public_surface_after'] = []
        with self.assertRaises(AssertionError): self.check(packed=value)


if __name__ == '__main__': unittest.main()
