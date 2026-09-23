"""Mutation tests against the real completed build inventory; no product execution."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/pyannote-convolution-four-build-amd-20260923'
spec=importlib.util.spec_from_file_location('four_scope_checks',ROOT/'tests/pyannote/convolution-four-build-amd/checks.py')
checks=importlib.util.module_from_spec(spec);spec.loader.exec_module(checks)


class Scope(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original=json.loads((BASE/'collected/inventory/instructions.json').read_text())
        cls.analysis=json.loads((BASE/'analysis.json').read_text())

    def setUp(self):self.value=copy.deepcopy(self.original)

    def check(self):return checks.inventory(self.value,self.analysis['measured'],self.analysis['built'])

    def test_real_inventory(self):self.assertEqual(self.check(),self.analysis['inventory'])

    def test_fallback_change_rejected(self):
        row=self.value['observations'][0]
        key=next(k for k in row['normalized_methods'] if k.startswith('Lokad.Onnx.ConvBlockedSpatial::Kernel512::'))
        row['differences'].append(key);row['candidate_methods'][key]='changed';row['unchanged_methods']-=1
        with self.assertRaises((AssertionError,ValueError)):self.check()

    def test_wrong_existing_method_rejected(self):
        row=self.value['observations'][0];old,=row['differences']
        key=next(k for k in row['normalized_methods'] if k.startswith('Lokad.Onnx.ConvBlockedSpatial::Kernel512::'))
        row['differences']=[key];row['candidate_methods'][key]=row['candidate_methods'].pop(old)
        with self.assertRaises(AssertionError):self.check()

    def test_extra_addition_rejected(self):
        row=self.value['observations'][0];row['added'].append('Unexpected::Method::Void Method()')
        with self.assertRaises((AssertionError,ValueError)):self.check()

    def test_data_change_rejected(self):
        row=self.value['observations'][1];key=next(iter(row['normalized_methods']))
        row['differences']=[key];row['candidate_methods'][key]='changed';row['unchanged_methods']-=1
        with self.assertRaises(AssertionError):self.check()

    def test_public_change_rejected(self):
        self.value['observations'][0]['public_surface_equal']=False
        with self.assertRaises(AssertionError):self.check()

    def test_wrong_product_rejected(self):
        self.value['observations'][0]['after_sha256']='0'*64
        with self.assertRaises(AssertionError):self.check()


if __name__=='__main__':unittest.main()
