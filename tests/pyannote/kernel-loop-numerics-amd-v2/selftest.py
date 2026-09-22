"""Mutation checks ensure the expanded census cannot silently lose its new coverage."""
import copy
from pathlib import Path
import unittest
from checks import raw_check, layers_check
from protocol import read

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'artifacts/pyannote-spatial-weight-reuse-20260922'
RAW = read(BASE/'output/raw-256.json')
WIDE = read(BASE/'output/wide-256.json')
SPAN = read(BASE/'output/span-256.json')
LAYERS = read(BASE/'output/layers-256.json')
FIXTURE = read(ROOT/'artifacts/pyannote-blocked-spatial-fixtures-20260922/output/result.json')
CORE = RAW['core']


class CoverageTests(unittest.TestCase):
    def test_original_family(self): self.assertTrue(raw_check(RAW, 8, CORE, False)['passed'])
    def test_wide_family(self): self.assertTrue(raw_check(WIDE, 8, CORE, True)['passed'])
    def test_layer_family(self): self.assertTrue(layers_check(LAYERS, 8, LAYERS, FIXTURE)['passed'])

    def test_spatial_family(self): self.assertTrue(raw_check(SPAN, 8, CORE, False, True)['passed'])

    def test_original_cannot_substitute_for_spatial(self):
        with self.assertRaises(AssertionError): raw_check(RAW, 8, CORE, False, True)

    def test_spatial_cannot_substitute_for_original(self):
        with self.assertRaises(AssertionError): raw_check(SPAN, 8, CORE, False)

    def test_missing_spatial_stride_two(self):
        value = copy.deepcopy(SPAN)
        for row in value['observations']: row['stride'] = 1
        with self.assertRaises(AssertionError): raw_check(value, 8, CORE, False, True)

    def test_original_cannot_substitute_for_wide(self):
        with self.assertRaises(AssertionError): raw_check(RAW, 8, CORE, True)

    def test_wide_supplemental_must_use_wide_channels(self):
        value = copy.deepcopy(WIDE); value['graph_cases'][-1]['m'] = 32
        with self.assertRaises(AssertionError): raw_check(value, 8, CORE, True)

    def test_missing_wide_case(self):
        value = copy.deepcopy(WIDE); value['observations'].pop()
        with self.assertRaises(AssertionError): raw_check(value, 8, CORE, True)

    def test_changed_finite_graph_output(self):
        value = copy.deepcopy(WIDE); value['graph_cases'][0]['executions'][0]['differences'] = 1
        with self.assertRaises(AssertionError): raw_check(value, 8, CORE, True)

    def test_wrong_prepared_weight_size(self):
        value = copy.deepcopy(WIDE); value['graph_cases'][0]['retained_weights'] -= 64
        with self.assertRaises(AssertionError): raw_check(value, 8, CORE, True)

    def test_missing_prepared_dispatch(self):
        value = copy.deepcopy(WIDE); value['graph_cases'][0]['executions'][0]['scratch_bytes'] = 0
        with self.assertRaises(AssertionError): raw_check(value, 8, CORE, True)

    def test_wrong_instruction_width(self):
        with self.assertRaises(AssertionError): raw_check(WIDE, 16, CORE, True)

    def test_missing_layer_dispatch(self):
        value = copy.deepcopy(LAYERS); value['graph_dispatch'].pop()
        with self.assertRaises(AssertionError): layers_check(value, 8, LAYERS, FIXTURE)

    def test_native_layer_failure(self):
        value = copy.deepcopy(LAYERS); value['native_failures'] = 1
        with self.assertRaises(AssertionError): layers_check(value, 8, LAYERS, FIXTURE)


if __name__ == '__main__': unittest.main(verbosity=2)
