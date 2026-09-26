import copy
import unittest
from compatibility import QUALIFIED,BUILD,CURRENT,read,reconcile,review


class Compatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=read(QUALIFIED/'collected/inventory/instructions.json')
        cls.pad=read(BUILD/'collected/inventory/instructions.json')
        cls.original=read(CURRENT/'analysis.json')['identities']['candidate']
        cls.selected=read(QUALIFIED/'analysis.json')['built']
        cls.candidate=read(BUILD/'analysis.json')['built']

    def check(self,root,pad):return reconcile(root,pad,self.original,self.selected,self.candidate)

    def test_retained_products_and_consumers(self):
        self.assertTrue(review()['passed'])

    def test_removed_public_binding(self):
        changed=copy.deepcopy(self.root)
        changed['observations'][0]['public_surface_after'].remove(changed['observations'][0]['public_surface'][0])
        with self.assertRaises(AssertionError):self.check(changed,self.pad)

    def test_wrong_product(self):
        changed=copy.deepcopy(self.pad);changed['observations'][1]['before_sha256']='0'*64
        with self.assertRaises(AssertionError):self.check(self.root,changed)

    def test_changed_data_body(self):
        changed=copy.deepcopy(self.pad);key=next(iter(changed['observations'][1]['normalized_methods']))
        changed['observations'][1]['normalized_methods'][key]='changed'
        with self.assertRaises(AssertionError):self.check(self.root,changed)

    def test_changed_method_flag(self):
        changed=copy.deepcopy(self.pad);key=next(iter(changed['observations'][0]['method_flags_before']))
        changed['observations'][0]['method_flags_before'][key]=512
        with self.assertRaises(AssertionError):self.check(self.root,changed)


if __name__=='__main__':unittest.main()
