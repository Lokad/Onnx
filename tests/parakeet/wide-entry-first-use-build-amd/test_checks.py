"""Reject unauthorized scope changes using the retained real M52 inventory."""
import copy
import json
from pathlib import Path
import unittest
from checks import inventory, one, operand, replace_body, ENTRY, CLONE, DISPATCH, HELPER

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'artifacts/parakeet-wide-projection-isolation-build-amd-20260923'


def fixture():
    value = json.loads((BASE/'collected/inventory/instructions.json').read_text())
    measured = json.loads((BASE/'collected/built.json').read_text())['product']
    built = {k:dict(v,sha256='synthetic-'+k) for k,v in measured.items()}
    for row in value['observations']:
        methods = row['normalized_methods'] | row['candidate_methods']
        row.update(normalized_methods=methods,methods=len(methods),unchanged_methods=len(methods),
            compiler_rename=None,added=[],removed=[],differences=[],candidate_methods={},
            method_flags_before=copy.deepcopy(row['method_flags_after']),
            before_sha256=measured[row['assembly']]['sha256'],after_sha256=built[row['assembly']]['sha256'])
    row=value['observations'][0]; old=row['normalized_methods']; new={}; flags=row['method_flags_after']
    entry=one(old,'Lokad.Onnx.Tensor`1[T]::'+ENTRY+'::');clone=entry.replace(ENTRY,CLONE);dispatch=entry.replace(ENTRY,DISPATCH)
    prior=dict(passed=True,entry_key=entry,entry_body=old[entry])
    for key,body in old.items():
        if body!='NO-BODY' and any(i['opcode']=='call' and i['operand']==operand(entry) for i in json.loads(body)['instructions']):
            new[key]=json.dumps(replace_body(body,{operand(entry):operand(dispatch)}));row['differences'].append(key)
    closure=json.loads(old[entry])['locals'][0]['type'];cloned=closure.replace('419_0','999_0')
    lam=one(old,closure+'::<'+ENTRY+'>b__0::');ctor=one(old,closure+'::.ctor::')
    for key,target in [(entry,clone),(lam,lam.replace(closure,cloned).replace(ENTRY,CLONE)),(ctor,ctor.replace(closure,cloned))]:
        new[target]=json.dumps(replace_body(old[key],{closure:cloned,'<'+ENTRY+'>b__0':'<'+CLONE+'>b__0'}))
        flags[target]=520 if key==entry else flags[key]
    new[dispatch]=json.dumps(dict(InitLocals=True,MaxStackSize=5,locals=[],exceptions=[],instructions=[
        dict(offset=i,opcode=op,operand=arg) for i,(op,arg) in enumerate([
            ('ldc.i4.s','30'),('ldc.i4','00040000'),('ldc.i4','00040000'),('ldc.i4','00000004'),
            ('call',operand(clone)),('call',operand(entry))])]))
    flags[dispatch]=256;flags[one(old,'Lokad.Onnx.Tensor`1[T]::'+HELPER+'::')]=520
    row.update(candidate_methods=new,added=list(set(new)-set(old)),unchanged_methods=3181)
    return value,measured,built,prior


class ScopeTests(unittest.TestCase):
    def test_exact_declared_clone_and_flag(self):
        self.assertEqual(inventory(*fixture())['candidate_core_methods'],3189)

    def test_original_entry_flag_and_shared_arithmetic_flag(self):
        for name in [ENTRY,'RunFloatMatMulKernel','RunIsolatedShortWideKernel']:
            args=fixture();row=args[0]['observations'][0]
            key=one(row['method_flags_after'],'Lokad.Onnx.Tensor`1[T]::'+name+'::')
            row['method_flags_after'][key]=512
            with self.assertRaises(AssertionError):inventory(*args)

    def test_clone_body_and_parallel_lambda(self):
        for needle in ['::'+CLONE+'::','::<'+CLONE+'>b__0::']:
            args=fixture();row=args[0]['observations'][0]
            key=next(k for k in row['added'] if needle in k)
            body=json.loads(row['candidate_methods'][key]);body['instructions'][0]['operand']='incorrect'
            row['candidate_methods'][key]=json.dumps(body)
            with self.assertRaises(AssertionError):inventory(*args)

    def test_wrong_caller_operand(self):
        args=fixture();row=args[0]['observations'][0];key=row['differences'][0]
        row['candidate_methods'][key]=row['normalized_methods'][key]
        with self.assertRaises(AssertionError):inventory(*args)

    def test_unrelated_change_or_added_method(self):
        for field in ['differences','added','removed']:
            args=fixture();args[0]['observations'][0][field].append('unexpected')
            with self.assertRaises(AssertionError):inventory(*args)

    def test_data_change_and_api_change(self):
        args=fixture();args[0]['observations'][1]['unchanged_methods']=696
        with self.assertRaises(AssertionError):inventory(*args)
        args=fixture();args[0]['observations'][0]['public_surface_equal']=False
        with self.assertRaises(AssertionError):inventory(*args)

    def test_helper_flag_required(self):
        args=fixture();row=args[0]['observations'][0]
        row['method_flags_after'][one(row['normalized_methods'],'Lokad.Onnx.Tensor`1[T]::'+HELPER+'::')]=8
        with self.assertRaises(AssertionError):inventory(*args)


if __name__=='__main__':unittest.main()
