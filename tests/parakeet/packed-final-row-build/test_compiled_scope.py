"""Reject concealed literals, branches, flags, missing methods and extra changes."""
import copy
import json
import unittest
from compiled_scope import DELETED,RUN_OLD,RUN_NEW,HELPER,reconcile,renamed,body_after_rename

OLD='Lokad.Onnx.Tensor`1+<>c__DisplayClass497_0[T]::.ctor::Void .ctor()'
NEW='Lokad.Onnx.Tensor`1+<>c__DisplayClass496_0[T]::.ctor::Void .ctor()'
CALLER='Lokad.Onnx.Tensor`1[T]::Resize::Void Resize()'


def body(symbol=None):
    return dict(InitLocals=True,MaxStackSize=1,locals=[],exceptions=[],instructions=[
        dict(offset=0,opcode='ldstr',operand=OLD),
        dict(offset=5,opcode='call' if symbol else 'nop',operand=symbol or ''),
        dict(offset=10,opcode='br.s',operand='00'),dict(offset=12,opcode='ret',operand='')])


def fixture():
    old={DELETED:json.dumps(body()),RUN_OLD:json.dumps(body()),OLD:json.dumps(body()),CALLER:json.dumps(body(OLD))}
    new={RUN_NEW:json.dumps(body(HELPER)),NEW:json.dumps(body()),CALLER:json.dumps(body(NEW)),HELPER:json.dumps(body())}
    return dict(assembly='Lokad.Onnx.dll',normalized_methods=old,removed=[DELETED,RUN_OLD,OLD],
        candidate_methods=new,method_flags_before={key:0 for key in old},method_flags_after={key:0 for key in new})


class Tests(unittest.TestCase):
    def test_only_declared_signature_body_changes(self):
        value=reconcile(fixture())
        self.assertEqual(value['differences'],[RUN_OLD]);self.assertEqual(value['removed'],[DELETED])
        self.assertEqual(value['added'],[HELPER]);self.assertEqual(value['compiler_renames'],{OLD:NEW})

    def changed_body(self,mutation):
        row=fixture();value=json.loads(row['candidate_methods'][CALLER]);mutation(value)
        row['candidate_methods'][CALLER]=json.dumps(value)
        self.assertIn(CALLER,reconcile(row)['differences'])

    def test_changed_opcode_survives(self):self.changed_body(lambda v:v['instructions'][1].update(opcode='callvirt'))
    def test_changed_branch_survives(self):self.changed_body(lambda v:v['instructions'][2].update(operand='FF'))
    def test_changed_stack_survives(self):self.changed_body(lambda v:v.update(MaxStackSize=2))
    def test_literal_looks_like_symbol_but_must_not_change(self):
        self.changed_body(lambda v:v['instructions'][0].update(operand=NEW))
        self.assertEqual(body_after_rename(json.dumps(body()))['instructions'][0]['operand'],OLD)
    def test_flags_are_not_ignored(self):
        row=fixture();row['method_flags_after'][NEW]=512
        with self.assertRaises(AssertionError):reconcile(row)
    def test_missing_original_method_rejected(self):
        row=fixture();del row['candidate_methods'][NEW];del row['method_flags_after'][NEW]
        with self.assertRaises(AssertionError):reconcile(row)
    def test_retained_deleted_method_rejected(self):
        row=fixture();row['removed'].remove(DELETED);row['method_flags_after'][DELETED]=0
        with self.assertRaises(AssertionError):reconcile(row)
    def test_unlisted_ordinals_and_other_owners_stay_intact(self):
        self.assertEqual(renamed(OLD),NEW)
        for value in [OLD.replace('497','496'),OLD.replace('Tensor','Other')]:self.assertEqual(renamed(value),value)


if __name__=='__main__':unittest.main()
