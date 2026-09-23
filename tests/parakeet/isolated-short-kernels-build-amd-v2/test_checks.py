"""Reject incorrect isolation, arithmetic copies and caller rewrites."""
import copy
import json
import unittest
from checks import PAIRS, GENERAL, DISPATCH, SHORT, calls_only, inventory, operand


def body(calls=()):
    return json.dumps(dict(InitLocals=True, MaxStackSize=8, locals=[], exceptions=[],
        instructions=[dict(offset=i * 5, opcode='call', operand=call) for i, call in enumerate(calls)]))


def fixture():
    general = 'Lokad.Onnx.Tensor`1[T]::' + GENERAL + '::Void ' + GENERAL + '(Int32, Int32, Int32, Single*, Single*, Single*, Lokad.Onnx.TensorExecutionOptions)'
    dispatch = general.replace(GENERAL, DISPATCH); short = general.replace(GENERAL, SHORT)
    old_short = general.replace(GENERAL, 'RunShortWidePackedRows')
    old_general = general.replace(GENERAL, 'RunGeneralFloatMatMulKernel')
    originals = {str(i): body() for i in range(3170)}
    originals[general] = body()
    callers = ['caller' + str(i) for i in range(4)]
    for key in callers: originals[key] = body([operand(general)])
    copies = {}
    for old, new in PAIRS:
        key = 'Lokad.Onnx.MathOps::' + old + '::Void ' + old + '(Int32, Int32, Single*, Single*)'
        originals[key] = body(); copies[key] = key.replace(old, new)
    prior = dict(passed=True, original_general_body=originals[general],
        wrapper_body=body([operand(old_short), operand(old_general)]),
        short_body=body([operand(key) for key in copies]),
        old_short_operand=operand(old_short), old_general_operand=operand(old_general))
    candidates = {k: body([operand(dispatch)]) for k in callers}
    candidates.update({new: originals[old] for old, new in copies.items()})
    candidates[dispatch] = body([operand(short), operand(general)])
    candidates[short] = body([operand(key) for key in copies.values()])
    flags = {k: 0 for k in originals}
    after = dict(flags, **{k: 512 for k in copies.values()}, **{dispatch: 256, short: 8})
    products = {n: dict(sha256=n, bytes=1) for n in ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll']}
    row = dict(assembly='Lokad.Onnx.dll', methods=3179, unchanged_methods=3175,
        normalized_methods=originals, candidate_methods=candidates, differences=callers,
        added=[dispatch, short, *copies.values()], removed=[], before_sha256='Lokad.Onnx.dll', after_sha256='Lokad.Onnx.dll',
        public_surface_equal=True, compiler_rename=None, method_flags_before=flags, method_flags_after=after)
    data = dict(assembly='Lokad.Onnx.Data.dll', methods=697, unchanged_methods=697,
        normalized_methods={str(i): body() for i in range(697)}, candidate_methods={}, differences=[], added=[], removed=[],
        before_sha256='Lokad.Onnx.Data.dll', after_sha256='Lokad.Onnx.Data.dll', public_surface_equal=True,
        compiler_rename=None, method_flags_before={str(i): 0 for i in range(697)}, method_flags_after={str(i): 0 for i in range(697)})
    return dict(inventory_complete=True, observations=[row, data]), products, prior


class ChecksTests(unittest.TestCase):
    def test_abstract_and_external_methods_keep_scope_checks(self):
        value, products, prior = fixture()
        for row in value['observations']:
            row['normalized_methods']['0'] = 'NO-BODY'
        self.assertTrue(inventory(value, products, products, prior)['passed'])
        value['observations'][0]['method_flags_after']['0'] = 512
        with self.assertRaises(AssertionError): inventory(value, products, products, prior)
        value['observations'][0]['method_flags_after']['0'] = 0
        value['observations'][0]['differences'].append('0')
        with self.assertRaises(AssertionError): inventory(value, products, products, prior)

    def test_exact_declared_scope_passes(self):
        value, products, prior = fixture(); report = inventory(value, products, products, prior)
        self.assertEqual((report['call_operands_changed'], report['exact_copy_bodies']), (4, 4))
        self.assertTrue(report['existing_implementation_flags_equal'])

    def test_collateral_and_missing_flags_rejected(self):
        value, products, prior = fixture()
        for mutation in [
            lambda r: r['method_flags_after'].__setitem__('0', 512),
            lambda r: r['method_flags_after'].__setitem__(r['added'][0], 0),
            lambda r: r['method_flags_after'].__setitem__(r['added'][2], 0),
            lambda r: r['method_flags_after'].__setitem__('extra', 0),
            lambda r: r.__setitem__('public_surface_equal', False),
            lambda r: r['added'].append('extra'),
            lambda r: r['removed'].append('0'),
            lambda r: r['differences'].append('0'),
        ]:
            changed = copy.deepcopy(value); mutation(changed['observations'][0])
            with self.assertRaises(AssertionError): inventory(changed, products, products, prior)

    def test_wrong_caller_copy_and_helper_rejected(self):
        value, products, prior = fixture()
        for key in ['caller0', *value['observations'][0]['added']]:
            changed = copy.deepcopy(value); changed['observations'][0]['candidate_methods'][key] = body(['wrong'])
            with self.subTest(key=key), self.assertRaises(AssertionError): inventory(changed, products, products, prior)

    def test_changed_caller_metadata_or_instruction_rejected(self):
        original = body(['old']); candidate = json.loads(body(['new']))
        for mutate in [lambda r: r['locals'].append('Single'), lambda r: r['instructions'][0].__setitem__('offset', 7),
                       lambda r: r['instructions'][0].__setitem__('opcode', 'callvirt')]:
            changed = copy.deepcopy(candidate); mutate(changed)
            with self.assertRaises(AssertionError): calls_only(original, json.dumps(changed), {'old': 'new'})

    def test_data_change_rejected(self):
        value, products, prior = fixture(); value['observations'][1]['differences'].append('0')
        with self.assertRaises(AssertionError): inventory(value, products, products, prior)

    def test_initializer_rename_requires_single_exact_pointer(self):
        value, products, prior = fixture(); row = value['observations'][0]
        old = 'Lokad.Onnx.Tensor`1+<>c[T]::<.cctor>b__536_0::Void <.cctor>b__536_0()'
        new = old.replace('536', '538'); ctor = 'Lokad.Onnx.Tensor`1[T]::.cctor::Void .cctor()'
        constructor = json.loads(body([old])); constructor['instructions'][0]['opcode'] = 'ldftn'
        constructor = json.dumps(constructor)
        for key, replacement, code in [('0', old, body()), ('1', ctor, constructor)]:
            del row['normalized_methods'][key]; del row['method_flags_before'][key]; del row['method_flags_after'][key]
            row['normalized_methods'][replacement] = code; row['method_flags_before'][replacement] = 0; row['method_flags_after'][replacement] = 0
        row['method_flags_after'][new] = row['method_flags_after'].pop(old)
        row['compiler_rename'] = dict(oldKey=old, newKey=new, identical_lambda_body=body(),
            beforeConstructor=constructor, afterConstructor=constructor.replace('536', '538'))
        self.assertTrue(inventory(value, products, products, prior)['initializer_rename'])
        row['compiler_rename']['afterConstructor'] = constructor.replace('536', '539')
        with self.assertRaises(AssertionError): inventory(value, products, products, prior)


if __name__ == '__main__': unittest.main()
