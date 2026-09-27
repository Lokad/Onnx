"""Reject scope drift and stale product bindings using the actual inventories."""
from copy import deepcopy
import unittest
from compatibility import CURRENT,QUALIFIED,CONTRACTS,read,reconcile


class Compatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = read(QUALIFIED/'collected/inventory/instructions.json')
        cls.tail = read(CONTRACTS/'build-collected/logs/instructions.json')
        cls.models = read(CURRENT/'analysis.json')['identities']['candidate']
        cls.products = read(CONTRACTS/'build-review.json')['products']

    def check(self,root=None,tail=None,models=None):
        return reconcile(self.root if root is None else root,self.tail if tail is None else tail,
            self.models if models is None else models,self.products['baseline'],self.products['candidate'])

    def test_actual_compiled_scope(self):
        self.assertTrue(self.check()['passed'])
        self.assertEqual(self.check()['underlying_methods_reconciled'],3983)

    def test_rejects_unrelated_changes_and_stale_identity(self):
        tail = deepcopy(self.tail); tail['observations'][0]['differences'].append('Unexpected::Body')
        with self.assertRaises(AssertionError): self.check(tail=tail)
        tail = deepcopy(self.tail); row = tail['observations'][0]
        row['method_flags_after'][next(iter(row['method_flags_before']))] = 'corrupt-flags'
        with self.assertRaises(AssertionError): self.check(tail=tail)
        tail = deepcopy(self.tail); tail['observations'][1]['public_surface_after'] = ['corrupt-binding']
        with self.assertRaises(AssertionError): self.check(tail=tail)
        models = deepcopy(self.models); models['Lokad.Onnx.dll']['sha256'] = '0'*64
        with self.assertRaises(AssertionError): self.check(models=models)
        root = deepcopy(self.root); row = root['observations'][0]
        row['normalized_methods'].pop(next(iter(row['normalized_methods'])))
        with self.assertRaises(AssertionError): self.check(root=root)


if __name__ == '__main__': unittest.main()
