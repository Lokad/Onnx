"""Ensure symbol reconciliation cannot conceal executable differences."""
import copy
import json
from pathlib import Path
import unittest
from generated_names import body_after_rename, reconcile, renamed

ROOT = Path(__file__).resolve().parents[3]


class GeneratedNamesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.row = json.loads((ROOT / 'artifacts/parakeet-owned-packed-weight-amd-20260925/build-collected/logs/instructions.json').read_text())['observations'][0]
        cls.key = next(k for k in cls.row['normalized_methods'] if '::NormalizeTransposePerm::' in k)

    def changed_body(self, mutate):
        row = copy.deepcopy(self.row)
        value = json.loads(row['candidate_methods'][self.key])
        mutate(value)
        row['candidate_methods'][self.key] = json.dumps(value)
        self.assertIn(self.key, reconcile(row)['differences'])

    def test_existing_bodies_only_have_the_seven_declared_edits(self):
        row = reconcile(self.row)
        self.assertEqual(7, len(row['differences']))
        self.assertEqual(41, len(row['compiler_renames']))
        self.assertEqual(23, len(row['added']))

    def test_opcode_change_survives(self):
        self.changed_body(lambda b: b['instructions'][0].update(opcode='nop'))

    def test_branch_change_survives(self):
        self.changed_body(lambda b: next(i for i in b['instructions'] if i['opcode'].startswith('br')).update(operand='FF'))

    def test_stack_change_survives(self):
        self.changed_body(lambda b: b.update(MaxStackSize=b['MaxStackSize'] + 1))

    def test_literal_symbol_text_is_not_renamed(self):
        body = {'locals': [], 'exceptions': [], 'instructions': [
            {'opcode': 'ldstr', 'operand': 'Lokad.Onnx.Tensor`1+<>c__DisplayClass517_0[T]'}]}
        self.assertEqual(body, body_after_rename(json.dumps(body)))

    def test_implementation_flag_change_rejected(self):
        row = copy.deepcopy(self.row)
        row['method_flags_after'][self.key] ^= 256
        with self.assertRaises(AssertionError):
            reconcile(row)

    def test_missing_method_rejected(self):
        row = copy.deepcopy(self.row)
        target = renamed(next(k for k in row['normalized_methods'] if '::<Resize>g__CubicWeight|492_2::' in k))
        del row['candidate_methods'][target]
        del row['method_flags_after'][target]
        with self.assertRaises(AssertionError):
            reconcile(row)


if __name__ == '__main__':
    unittest.main()
