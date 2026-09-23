"""Check the one-method boundary against the complete retained root inventory."""
import copy
from pathlib import Path
import unittest
from checks import inventory
from protocol import read

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'artifacts/pyannote-winograd-product-root-amd-20260923'
ORIGINAL = read(BASE/'collected/inventory/instructions.json')
ANALYSIS = read(BASE/'analysis.json')


def example():
    value = copy.deepcopy(ORIGINAL); row = value['observations'][0]
    key, = [k for k in row['normalized_methods'] if k.startswith('Lokad.Onnx.ConvBlockedSpatial::MultiplyWinograd512::')]
    row['differences'] = [key]; row['candidate_methods'] = {key:row['normalized_methods'][key]+' candidate'}
    row['unchanged_methods'] -= 1
    return value


def check(value): return inventory(value,ANALYSIS['measured'],ANALYSIS['built'])


class Scope(unittest.TestCase):
    def test_complete_retained_census(self): self.assertTrue(check(example())['passed'])
    def test_outside_scope_rejected(self):
        for change in ['other-method','extra','missing','public','identity','census','rename','same-body','candidate-map']:
            with self.subTest(change=change):
                value = example(); row = value['observations'][0]; key, = row['differences']
                if change == 'other-method':
                    other = next(k for k in row['normalized_methods'] if k != key)
                    row['differences'].append(other); row['candidate_methods'][other] = 'changed'; row['unchanged_methods'] -= 1
                elif change == 'extra': row['added'] = ['unexpected']
                elif change == 'missing': row['removed'] = ['unexpected']
                elif change == 'public': row['public_surface_equal'] = False
                elif change == 'identity': row['after_sha256'] = 'wrong'
                elif change == 'census': row['normalized_methods'].pop(key)
                elif change == 'rename': row['compiler_rename'] = 'unexpected'
                elif change == 'same-body': row['candidate_methods'][key] = row['normalized_methods'][key]
                else: row['candidate_methods']['unexpected'] = 'unexpected'
                with self.assertRaises((AssertionError,ValueError)): check(value)


if __name__ == '__main__': unittest.main(verbosity=2)
