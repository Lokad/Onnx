import copy
import json
from pathlib import Path
import unittest
from normalize import normalize, canonical


def body(operand, opcode='call'):
    return json.dumps(dict(InitLocals=True, MaxStackSize=1, locals=[], exceptions=[], instructions=[dict(offset=0, opcode=opcode, operand=operand)]))


class Names(unittest.TestCase):
    def example(self):
        owner = 'Lokad.Onnx.ComputationalGraph'
        parent = owner+'::Parent::Void Parent()'
        old = owner+'::<Parent>b__10_0::Void <Parent>b__10_0()'
        new = owner+'::<Parent>b__12_0::Void <Parent>b__12_0()'
        old_body, new_body = body(owner+'::Void <Parent>b__10_0()'), body(owner+'::Void <Parent>b__12_0()')
        return dict(normalized_methods={parent: old_body, old: body('0', 'ret')},
            candidate_methods={parent: new_body, new: body('0', 'ret')}, removed=[old], differences=[parent], added=[new])

    def test_bijective_rename_keeps_every_body(self):
        result, _ = normalize(self.example()); self.assertEqual(result['changed'], []); self.assertEqual(len(result['renames']), 1)

    def test_changed_callback_is_rejected(self):
        row = self.example(); key = row['added'][0]; row['candidate_methods'][key] = body('1', 'ret')
        with self.assertRaises(AssertionError): normalize(row)

    def test_changed_constant_is_not_normalized(self):
        row = self.example(); key = row['differences'][0]
        b = json.loads(row['candidate_methods'][key]); b['instructions'].append(dict(offset=1, opcode='ldc.i4', operand='02000000'))
        row['candidate_methods'][key] = json.dumps(b)
        result, _ = normalize(row); self.assertEqual(result['changed'], [key])

    def test_literal_looks_like_symbol_but_stays_literal(self):
        row = self.example(); key = 'Lokad.Onnx.ComputationalGraph::Literal::Void Literal()'
        row['normalized_methods'][key] = body('<Parent>b__10_0', 'ldstr')
        row['candidate_methods'][key] = body('<Parent>b__12_0', 'ldstr'); row['differences'].append(key)
        result, _ = normalize(row); self.assertEqual(result['changed'], [key])

    def test_actual_inventory_has_no_missing_methods(self):
        root = Path(__file__).resolve().parents[3]
        value = json.loads((root/'artifacts/pyannote-blocked-spatial-composition-20260922/instructions.json').read_text())
        result, _ = normalize(value['observations'][0])
        self.assertEqual(result['methods'], 3113); self.assertEqual(len(result['renames']), 197)
        self.assertEqual(result['unchanged'], 3103); self.assertEqual(len(result['changed']), 10)

    def test_one_old_symbol_cannot_map_to_two_new_symbols(self):
        row = self.example(); key = 'Lokad.Onnx.ComputationalGraph::Other::Void Other()'
        row['normalized_methods'][key] = body('Lokad.Onnx.ComputationalGraph::Void <Parent>b__10_0()')
        row['candidate_methods'][key] = body('Lokad.Onnx.ComputationalGraph::Void <Parent>b__13_0()'); row['differences'].append(key)
        with self.assertRaises(AssertionError): normalize(row)

    def test_two_old_symbols_cannot_map_to_one_new_symbol(self):
        row = self.example(); key = 'Lokad.Onnx.ComputationalGraph::Other::Void Other()'
        row['normalized_methods'][key] = body('Lokad.Onnx.ComputationalGraph::Void <Parent>b__11_0()')
        row['candidate_methods'][key] = body('Lokad.Onnx.ComputationalGraph::Void <Parent>b__12_0()'); row['differences'].append(key)
        with self.assertRaises(AssertionError): normalize(row)


if __name__ == '__main__': unittest.main()
