"""Exercise attribution safeguards against the retained, fully reconciled capture."""
import copy
import unittest
from partition import native_descriptors_equal, partition, references


class CompletePartitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapping, cls.managed, cls.native = references()

    def calculate(self, mapping=None, old_managed=None, old_native=None):
        return partition(self.managed, self.native, mapping or self.mapping,
                         old_managed or self.managed, old_native or self.native)

    def test_reproduces_every_published_historical_group(self):
        rows = self.calculate()
        self.assertEqual(len(rows), len(self.mapping['partition']))
        for actual, expected in zip(rows, self.mapping['partition'], strict=True):
            self.assertEqual(actual['group'], expected['group'])
            for field in ['managed_seconds','ort_seconds','excess_seconds']:
                self.assertAlmostEqual(actual[field], expected[field], places=10)
            for field in ['managed_members','ort_members']:
                self.assertEqual(actual.get(field), expected.get(field))

    def test_old_clocks_cannot_enter_fresh_partition(self):
        mapping, managed, native = map(copy.deepcopy, [self.mapping, self.managed, self.native])
        for row in mapping['partition']:
            for key in ['managed_seconds','ort_seconds','excess_seconds']:
                row[key] = float('nan')
        for row in managed['phases']['wall']['node_rows']:
            row['ticks'] = row['corpus_seconds'] = float('nan')
        for graph in native['profiles'].values():
            for row in graph['node_clocks']:
                row['inclusive_us'] = row['exclusive_us'] = float('nan')
        self.assertEqual(self.calculate(), self.calculate(mapping, managed, native))

    def test_rejects_overlapping_or_missing_memberships(self):
        for mutation in ['duplicate','missing']:
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                mapping = copy.deepcopy(self.mapping)
                if mutation == 'duplicate':
                    mapping['partition'].append(copy.deepcopy(mapping['partition'][0]))
                else:
                    mapping['partition'].pop(0)
                self.calculate(mapping)

    def test_native_runtime_shape_and_coverage_changes_require_review(self):
        for mutation in ['shape','calls','node','duplicate']:
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                changed = copy.deepcopy(self.native['profiles'])
                encoder = changed['encoder']
                if mutation == 'shape':
                    encoder['shapes'][0]['inputs'] = ['__changed_shape__']
                elif mutation == 'calls':
                    encoder['node_clocks'][0]['calls'] -= 1
                elif mutation == 'node':
                    encoder['nodes'].pop(next(iter(encoder['nodes'])))
                else:
                    encoder['shapes'].append(copy.deepcopy(encoder['shapes'][0]))
                native_descriptors_equal(self.native['profiles'], changed)


if __name__ == '__main__':
    unittest.main()
