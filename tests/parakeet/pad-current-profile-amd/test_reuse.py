"""Reject missing metadata and observer-scope drift; fixtures are not qualification."""
import copy
import unittest
from reuse import ROOT, BUILD, BASELINE_ROOT, metadata_and_data
from run import read


class ReuseScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = read(BASELINE_ROOT / 'collected/inventory/instructions.json')
        cls.candidate = read(BUILD / 'collected/inventory/instructions.json')
        cls.measured = read(BUILD / 'analysis.json')['built']
        # Build a checker fixture in memory from measured candidate bodies and
        # baseline metadata. It is never an actual-root receipt or input bundle.
        cls.fixture = copy.deepcopy(cls.baseline)
        cls.built = copy.deepcopy(cls.measured)
        for index, (name, count) in enumerate([('Lokad.Onnx.dll', 3282), ('Lokad.Onnx.Data.dll', 697)]):
            row = cls.fixture['observations'][index]
            source = cls.candidate['observations'][index]
            row.update(before_sha256=cls.measured[name]['sha256'], after_sha256=cls.built[name]['sha256'],
                       methods=count, unchanged_methods=count, differences=[], added=[], removed=[],
                       candidate_methods={}, public_surface_equal=True)
            row['normalized_methods'] = dict(source['normalized_methods'], **source['candidate_methods'])
            row['method_flags_before'] = row['method_flags_after'] = copy.deepcopy(source['method_flags_after'])
            row['public_surface'] = copy.deepcopy(row['public_surface_after'])
            row['assembly_attributes_before'] = copy.deepcopy(row['assembly_attributes_after'])

    def check(self, value):
        return metadata_and_data(value, self.baseline, self.candidate, self.measured, self.built)

    def test_complete_equal_metadata_and_bodies(self):
        self.assertTrue(self.check(self.fixture)['passed'])

    def test_missing_assembly_attributes_are_not_equality(self):
        value = copy.deepcopy(self.fixture)
        del value['observations'][1]['assembly_attributes_after']
        with self.assertRaises(KeyError):
            self.check(value)

    def test_changed_root_metadata_is_rejected(self):
        for index in range(2):
            with self.subTest(assembly=index):
                value = copy.deepcopy(self.fixture)
                value['observations'][index]['assembly_attributes_after'].append('unexpected attribute')
                with self.assertRaises(AssertionError):
                    self.check(value)

    def test_changed_public_surface_is_rejected(self):
        value = copy.deepcopy(self.fixture)
        value['observations'][0]['public_surface_after'].append('unexpected public member')
        with self.assertRaises(AssertionError):
            self.check(value)

    def test_changed_data_body_or_flags_is_rejected(self):
        for field in ['normalized_methods', 'method_flags_after']:
            with self.subTest(field=field):
                value = copy.deepcopy(self.fixture)
                data = value['observations'][1]
                key = next(iter(data[field]))
                data[field][key] = 'changed'
                with self.assertRaises(AssertionError):
                    self.check(value)

    def test_different_product_is_rejected(self):
        value = copy.deepcopy(self.fixture)
        value['observations'][0]['after_sha256'] = '0' * 64
        with self.assertRaises(AssertionError):
            self.check(value)


if __name__ == '__main__':
    unittest.main()
