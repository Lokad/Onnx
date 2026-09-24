"""Reject scope drift with independently retained current and helper inventories."""
import copy
import json
from pathlib import Path
import unittest
from checks import ADDED, PROVIDER, inventory

ROOT = Path(__file__).resolve().parents[3]


def read(path): return json.loads(path.read_text(encoding='utf8'))


def fixture():
    current = ROOT/'artifacts/parakeet-validated-composition-root-amd-20260924'
    retained = ROOT/'artifacts/parakeet-dense-scalar-where-build-amd-20260924'
    value = read(current/'collected/inventory/instructions.json')
    measured = read(current/'analysis.json')['measured']
    built = {name:dict(p, sha256='synthetic-'+name) for name,p in measured.items()}
    prior = dict(passed=True, current_methods={}, current_flags={})
    for row in value['observations']:
        name = row['assembly']
        prior['current_methods'][name] = copy.deepcopy(row['normalized_methods'])
        prior['current_flags'][name] = copy.deepcopy(row['method_flags_before'])
        row['before_sha256'] = measured[name]['sha256']; row['after_sha256'] = built[name]['sha256']
    old = read(retained/'collected/inventory/instructions.json')['observations'][0]
    prior['retained_methods'] = copy.deepcopy(old['candidate_methods'])
    prior['retained_flags'] = {k:old['method_flags_after'][k] for k in old['candidate_methods']}
    row = value['observations'][0]
    row.update(differences=[PROVIDER], added=list(ADDED), unchanged_methods=3250,
               candidate_methods=copy.deepcopy(old['candidate_methods']))
    row['method_flags_after'].update(ADDED)
    return value, measured, built, prior


class ScopeTests(unittest.TestCase):
    def test_exact_composition(self):
        self.assertEqual(inventory(*fixture())['candidate_core_methods'], 3253)

    def test_each_retained_body(self):
        for key in [PROVIDER, *ADDED]:
            args = fixture(); row = args[0]['observations'][0]
            body = json.loads(row['candidate_methods'][key]); body['MaxStackSize'] += 1
            row['candidate_methods'][key] = json.dumps(body)
            with self.assertRaises(AssertionError): inventory(*args)

    def test_method_flags(self):
        for key in [PROVIDER, *ADDED]:
            args = fixture(); args[0]['observations'][0]['method_flags_after'][key] ^= 8
            with self.assertRaises(AssertionError): inventory(*args)

    def test_other_core_or_data_changes(self):
        for index in [0,1]:
            for field in ['normalized_methods','method_flags_before']:
                args = fixture(); row = args[0]['observations'][index]
                key = next(iter(row[field])); row[field][key] = 'changed'
                with self.assertRaises(AssertionError): inventory(*args)

    def test_identity_and_surface(self):
        for index in [0,1]:
            for field,value in [('public_surface_equal',False),('compiler_rename',{}),
                                ('before_sha256','wrong'),('after_sha256','wrong'),('removed',['extra'])]:
                args = fixture(); args[0]['observations'][index][field] = value
                with self.assertRaises(AssertionError): inventory(*args)

    def test_added_and_changed_census(self):
        for field,value in [('added',[]),('differences',[]),('unchanged_methods',3251),('candidate_methods',{})]:
            args = fixture(); args[0]['observations'][0][field] = value
            with self.assertRaises(AssertionError): inventory(*args)


if __name__ == '__main__': unittest.main()
