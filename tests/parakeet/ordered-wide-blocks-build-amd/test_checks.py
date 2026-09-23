"""Adversarial scope checks using the real parent plus synthetic declared additions."""
import copy
import json
from pathlib import Path
import unittest
from checks import ADDED, HELPER, TRY, TWO, THREE, guarded_body, inventory, operand

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
        if op == 'ldc.i4':
            item['operand'] = item['operand'].to_bytes(4, 'little', signed=True).hex().upper()
        elif op != 'break' and op.startswith(('br', 'beq', 'bne', 'bge', 'bgt', 'ble', 'blt', 'leave')):
            item['operand'] = (offset(item['operand']) - offset(index + 1)).to_bytes(4, 'little', signed=True).hex().upper()
    for region in body['exceptions']:
        for prefix in ['Try', 'Handler']:
            start = region[prefix + 'Offset']; end = start + region[prefix + 'Length']
            region[prefix + 'Offset'] = offset(start)
            region[prefix + 'Length'] = offset(end) - offset(start)
        assert region['filter'] == -1
    return json.dumps(body)


def body(pairs):
    return dict(InitLocals=False, MaxStackSize=8, locals=[], exceptions=[],
                instructions=[dict(opcode=op, operand=arg) for op, arg in [*pairs, ('ret', '')]])


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
    prior = dict(passed=True, helper_key=HELPER, helper_body=row['normalized_methods'][HELPER])
    candidates = {HELPER: encoded(guarded_body(prior['helper_body']))}
    candidates[TRY] = encoded(body([
        ('call', 'System.Runtime.Intrinsics.X86.Fma::Boolean get_IsSupported()'),
        *[('ldc.i4', n) for n in [64, 1024, 1024, 32, 67108864]],
        ('call', operand(THREE)), ('call', operand(TWO))]))
    for key, count in [(TWO, 8), (THREE, 12)]:
        candidates[key] = encoded(body([
            ('ldc.i4', 256), ('call', 'System.Math::Int32 Min(Int32, Int32)'),
            *[('call', 'System.Runtime.Intrinsics.X86.Fma::Vector256 MultiplyAdd(...)') for _ in range(count)],
            ('ldc.i4', 256)]))
    row.update(candidate_methods=candidates, differences=[HELPER], added=list(ADDED), unchanged_methods=3188)
    row['method_flags_after'].update(ADDED)
    return value, measured, built, prior


class ScopeTests(unittest.TestCase):
    def test_declared_insertion_and_method_set(self):
        self.assertEqual(inventory(*fixture())['candidate_core_methods'], 3192)

    def test_unchanged_callers_arithmetic_and_flags_rejected_if_modified(self):
        for name in ['RunFloatMatMulKernel', 'RunWideProjectionMatMul2DCore', 'ShortWideMultiply2Rows']:
            with self.subTest(method=name):
                args = fixture(); row = args[0]['observations'][0]
                key = next(k for k in row['method_flags_after'] if '::' + name + '::' in k)
                row['method_flags_after'][key] ^= 8
                with self.assertRaises(AssertionError): inventory(*args)

    def test_branch_after_guard_cannot_bypass_pool_return(self):
        args = fixture(); row = args[0]['observations'][0]
        value = guarded_body(args[3]['helper_body'])
        value['instructions'][61]['operand'] = 88  # Bypasses the leave/finally boundary.
        row['candidate_methods'][HELPER] = encoded(value)
        with self.assertRaises(AssertionError): inventory(*args)

    def test_fallback_branch_cannot_skip_new_guard(self):
        args = fixture(); row = args[0]['observations'][0]
        value = guarded_body(args[3]['helper_body'])
        self.assertEqual(value['instructions'][45]['operand'], 54)
        value['instructions'][45]['operand'] = 62
        row['candidate_methods'][HELPER] = encoded(value)
        with self.assertRaises(AssertionError): inventory(*args)

    def test_exception_boundary_or_local_mutation_rejected(self):
        for kind in ['region', 'local', 'stack']:
            with self.subTest(kind=kind):
                args = fixture(); row = args[0]['observations'][0]
                value = guarded_body(args[3]['helper_body'])
                if kind == 'region': value['exceptions'][0]['TryLength'] -= 1
                elif kind == 'local': value['locals'][0]['IsPinned'] = True
                else: value['MaxStackSize'] += 1
                row['candidate_methods'][HELPER] = encoded(value)
                with self.assertRaises(AssertionError): inventory(*args)

    def test_guard_argument_substitution_rejected(self):
        args = fixture(); row = args[0]['observations'][0]
        value = guarded_body(args[3]['helper_body'])
        value['instructions'][54]['opcode'] = 'ldarg.0'
        row['candidate_methods'][HELPER] = encoded(value)
        with self.assertRaises(AssertionError): inventory(*args)

    def test_undeclared_method_or_compiler_rename_rejected(self):
        for field in ['added', 'removed', 'differences']:
            args = fixture(); args[0]['observations'][0][field].append('undeclared')
            with self.assertRaises(AssertionError): inventory(*args)
        args = fixture(); args[0]['observations'][0]['compiler_rename'] = {'undeclared': True}
        with self.assertRaises(AssertionError): inventory(*args)

    def test_guard_block_and_kernel_flags_rejected_if_changed(self):
        for key in [TRY, TWO, THREE]:
            args = fixture(); args[0]['observations'][0]['method_flags_after'][key] ^= 8
            with self.assertRaises(AssertionError): inventory(*args)
        args = fixture(); row = args[0]['observations'][0]
        raw = json.loads(row['candidate_methods'][TWO])
        raw['instructions'][0]['operand'] = '00020000'  # 512 instead of256.
        row['candidate_methods'][TWO] = json.dumps(raw)
        with self.assertRaises(AssertionError): inventory(*args)

    def test_data_or_public_surface_change_rejected(self):
        args = fixture(); args[0]['observations'][1]['unchanged_methods'] -= 1
        with self.assertRaises(AssertionError): inventory(*args)
        args = fixture(); args[0]['observations'][0]['public_surface_equal'] = False
        with self.assertRaises(AssertionError): inventory(*args)


if __name__ == '__main__':
    unittest.main()
