"""Check complete accounting on retained data, without creating a new profile."""
import copy
import unittest
from diagnose import MAPPING, ROOT, read, partition_reports, descriptors_equal


class CompletePartitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapping = read(MAPPING / 'analysis.json')
        cls.managed = read(ROOT / 'artifacts/parakeet-owned-batch-isolation-profile-resume-amd-20260925/analysis.json')
        cls.previous = read(ROOT / 'artifacts/parakeet-packed-final-row-profile-closure-amd-20260925/analysis.json')
        cls.native = read(ROOT / 'artifacts/parakeet-ort-diagnosis-amd-20260924/analysis.json')
        cls.expected = read(ROOT / 'tests/parakeet/owned-batch-isolation-profile-results/observations-20260925.json')

    def test_retained_complete_accounting_and_activation_split(self):
        partition, activations = partition_reports(self.managed, self.mapping, self.native, self.previous)
        self.assertEqual(partition, self.expected['partition'])
        self.assertEqual(len(activations), 72)
        groups = [r for r in partition if 'SiLU' in r['group']]
        self.assertEqual(len(groups), 2)
        self.assertAlmostEqual(sum(r['managed_seconds'] for r in activations),
                               sum(r['managed_seconds'] for r in groups), places=12)
        self.assertAlmostEqual(sum(r['ort_seconds'] for r in activations),
                               sum(r['ort_seconds'] for r in groups), places=12)

    def test_missing_partition_group_is_rejected(self):
        mapping = copy.deepcopy(self.mapping)
        mapping['partition'] = mapping['partition'][1:]
        with self.assertRaises(AssertionError):
            partition_reports(self.managed, mapping, self.native, self.previous)

    def test_duplicate_or_changed_node_is_rejected(self):
        rows = self.managed['phases']['wall']['node_rows']
        with self.assertRaises(AssertionError):
            descriptors_equal(rows, rows + [rows[0]])
        changed = copy.deepcopy(rows)
        changed[0]['name'] += '-changed'
        with self.assertRaises(AssertionError):
            descriptors_equal(rows, changed)


if __name__ == '__main__':
    unittest.main()
