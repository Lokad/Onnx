import copy
import json
import unittest
from review import BASE, PAD, compare, read


class CompositionTests(unittest.TestCase):
    def fixture(self):
        row = read(BASE / 'collected/inventory/instructions.json')['observations'][0]
        return json.loads(row['normalized_methods'][PAD]), json.loads(row['candidate_methods'][PAD])

    def test_actual_insertion_passes(self):
        before, after = self.fixture()
        self.assertTrue(compare(before, after)['branch_targets_preserved'])

    def test_rejects_collateral_arithmetic_branch_local_and_guard_changes(self):
        before, after = self.fixture()
        changes = [
            lambda v: v['instructions'][37].__setitem__('operand', '00'),
            lambda v: v['instructions'][41].__setitem__('operand', 'wrong helper'),
            lambda v: v['instructions'][55].__setitem__('opcode', 'sub'),
            lambda v: v['instructions'][2].__setitem__('operand', '09'),
            lambda v: v['locals'][3].__setitem__('type', 'System.Int32'),
            lambda v: v.__setitem__('MaxStackSize', 5),
        ]
        for index, mutate in enumerate(changes):
            with self.subTest(index=index):
                value = copy.deepcopy(after); mutate(value)
                with self.assertRaises(AssertionError): compare(before, value)


if __name__ == '__main__':
    unittest.main()
