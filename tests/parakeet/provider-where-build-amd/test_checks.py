"""Reject scope drift using the real selected inventory and explicit mutations."""
import copy
import json
from pathlib import Path
import unittest
from checks import ADDED, HELPER, TRY, provider_body, inventory
from il_body import normalized_body
ROOT = Path(__file__).resolve().parents[3]
PARENT = ROOT / 'artifacts/parakeet-wide-entry-first-use-build-amd-20260923'

def encoded(body):
    # Synthetic offsets deliberately differ from the parent. Targets and region
    # endpoints, not those byte distances, must survive the declared insertion.
    body = copy.deepcopy(body); instructions = body['instructions']
    def offset(index):
        return (len(instructions) - 1) * 16 + 1 if index == len(instructions) else index * 16
    for index, item in enumerate(instructions):
        item['offset'] = offset(index); op = item['opcode']
        if op in ['ldloc', 'ldloca', 'stloc']:
            item['opcode'] += '.s'; item['operand'] = item['operand'].to_bytes(1, 'little').hex().upper()
        if op == 'ldc.i4':
            item['operand'] = item['operand'].to_bytes(4, 'little', signed=True).hex().upper()
        elif op == 'switch':
            item['operand'] = b''.join((offset(t)-offset(index+1)).to_bytes(4,'little',signed=True) for t in item['operand']).hex().upper()
        elif op != 'break' and op.startswith(('br', 'beq', 'bne', 'bge', 'bgt', 'ble', 'blt', 'leave')):
            item['operand'] = (offset(item['operand']) - offset(index + 1)).to_bytes(4, 'little', signed=True).hex().upper()
    for region in body['exceptions']:
        for prefix in ['Try', 'Handler']:
            start = region[prefix + 'Offset']; end = start + region[prefix + 'Length']
            region[prefix + 'Offset'] = offset(start)
            region[prefix + 'Length'] = offset(end) - offset(start)
        assert region['filter'] == -1
    return json.dumps(body)



def fixture():
    value = json.loads((PARENT / 'collected/inventory/instructions.json').read_text())
    measured = json.loads((PARENT / 'analysis.json').read_text())['built']
    built = {name: dict(pin, sha256='synthetic-' + name) for name, pin in measured.items()}
    for row in value['observations']:
        methods = row['normalized_methods'] | row['candidate_methods']
        rename = row['compiler_rename']
        if rename:
            methods[rename['newKey']] = methods.pop(rename['oldKey'])
            methods['Lokad.Onnx.Tensor`1[T]::.cctor::Void .cctor()'] = rename['afterConstructor']
        row.update(normalized_methods=methods, methods=len(methods), unchanged_methods=len(methods),
                   compiler_rename=None, added=[], removed=[], differences=[], candidate_methods={},
                   method_flags_before=copy.deepcopy(row['method_flags_after']),
                   before_sha256=measured[row['assembly']]['sha256'], after_sha256=built[row['assembly']]['sha256'])
    row = value['observations'][0]
    assert len(row['normalized_methods']) == 3189
    assert set(row['normalized_methods']) == set(row['method_flags_before'])
    qualified = json.loads((ROOT / 'artifacts/parakeet-scalar-where-build-amd-v3-20260924/collected/inventory/instructions.json').read_text())['observations'][0]['candidate_methods'][TRY]
    prior = dict(passed=True, helper_key=HELPER, helper_body=row['normalized_methods'][HELPER], uniform_key=TRY, uniform_body=qualified)
    candidates = {HELPER: encoded(provider_body(prior['helper_body'])), TRY: qualified}
    row.update(candidate_methods=candidates, differences=[HELPER], added=list(ADDED), unchanged_methods=3188)
    row['method_flags_after'].update(ADDED)
    return value, measured, built, prior


class ScopeTests(unittest.TestCase):
    def test_declared_scope(self):
        self.assertEqual(inventory(*fixture())['candidate_core_methods'], 3190)

    def test_guard_types_arguments_and_edges(self):
        for index in [146,147,155,156,157,159,161,164,166,170,171,172,180,190]:
            with self.subTest(index=index):
                args=fixture(); value=provider_body(args[3]['helper_body'])
                item=value['instructions'][index]
                item['operand'] = item['operand']+1 if isinstance(item['operand'],int) else item['operand']+'changed'
                args[0]['observations'][0]['candidate_methods'][HELPER]=encoded(value)
                with self.assertRaises(AssertionError): inventory(*args)

    def test_original_body_cannot_change(self):
        for index in [0,1,16,21,24,40,54,65,91,200,228]:
            args=fixture(); value=provider_body(args[3]['helper_body'])
            value['instructions'][index]=dict(opcode='nop',operand='')
            args[0]['observations'][0]['candidate_methods'][HELPER]=encoded(value)
            with self.assertRaises(AssertionError): inventory(*args)

    def test_locals_stack_flags(self):
        for mode in ['locals','stack','flag']:
            args=fixture(); row=args[0]['observations'][0];value=provider_body(args[3]['helper_body'])
            if mode=='locals': value['locals'][1]['IsPinned']=True
            if mode=='stack': value['MaxStackSize']+=1
            if mode=='flag': row['method_flags_after'][HELPER]=520
            row['candidate_methods'][HELPER]=encoded(value)
            with self.assertRaises(AssertionError): inventory(*args)

    def test_other_core_or_data_changes(self):
        for index in [0,1]:
            args=fixture(); args[0]['observations'][index]['differences'].append('unrelated')
            with self.assertRaises(AssertionError): inventory(*args)

    def test_api_added_removed_or_rename(self):
        for field,value in [('public_surface_equal',False),('compiler_rename',{}),('added',[]),('removed',['anything'])]:
            args=fixture();args[0]['observations'][0][field]=value
            with self.assertRaises(AssertionError): inventory(*args)

    def test_helper_must_remain_exact(self):
        for fragment in ['IndexOf[Byte]','IndexOfAnyExcept[Byte]','Fill(T)','CopyTo(System.Span`1[T])']:
            args=fixture();row=args[0]['observations'][0]
            body=normalized_body(row['candidate_methods'][TRY]);changes=0
            for item in body['instructions']:
                if isinstance(item['operand'],str) and fragment in item['operand']:
                    item['operand']=item['operand'].replace(fragment,'wrong');changes+=1
            self.assertEqual(changes,1)
            row['candidate_methods'][TRY]=encoded(body)
            with self.assertRaises(AssertionError): inventory(*args)


if __name__=='__main__': unittest.main()
