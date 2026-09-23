"""Exercise the one-guard gate against the retained real M50 inventory."""
import copy
import json
from pathlib import Path
import unittest
from checks import inventory, extended_body

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'artifacts/parakeet-isolated-short-kernels-build-amd-v2-20260923'


def fixture():
    old = json.loads((BASE / 'collected/inventory/instructions.json').read_text())
    measured = json.loads((BASE / 'collected/built.json').read_text())['product']
    built = {k: dict(v, sha256='new-' + k) for k, v in measured.items()}
    value = copy.deepcopy(old)
    for row in value['observations']:
        name = row['assembly']; methods = dict(row['normalized_methods']); methods.update(row['candidate_methods'])
        rename = row['compiler_rename']
        if rename:
            methods[rename['newKey']] = methods.pop(rename['oldKey'])
            methods['Lokad.Onnx.Tensor`1[T]::.cctor::Void .cctor()'] = rename['afterConstructor']
        row.update(methods=len(methods), normalized_methods=methods, compiler_rename=None, added=[], removed=[],
            method_flags_before=copy.deepcopy(row['method_flags_after']), differences=[], candidate_methods={},
            unchanged_methods=len(methods), before_sha256=measured[name]['sha256'], after_sha256=built[name]['sha256'])
    core = value['observations'][0]
    key = next(k for k in core['normalized_methods'] if '::RunIsolatedShortWideKernel::' in k)
    prior = dict(passed=True, dispatcher_key=key, dispatcher_body=core['normalized_methods'][key])
    core.update(unchanged_methods=3184, differences=[key], candidate_methods={key: json.dumps(extended_body(prior['dispatcher_body']))})
    return value, measured, built, prior


class CompiledScopeTests(unittest.TestCase):
    def test_declared_guard_extension(self):
        result = inventory(*fixture())
        self.assertTrue(result['removed_upper_row_guard'])
        self.assertEqual((result['before_instructions'], result['after_instructions']), (46, 43))

    def test_axis_budget_fallback_and_branch(self):
        for opcode, operand, changed in [('ldc.i4', '00040000', '01040000'),
                                         ('ldc.i4', '00000004', '00000008'),
                                         ('call', None, 'wrong fallback'),
                                         ('blt.s', '46', '45')]:
            with self.subTest(opcode=opcode, operand=operand):
                args = fixture(); row = args[0]['observations'][0]; key = args[3]['dispatcher_key']
                body = json.loads(row['candidate_methods'][key])
                found = [i for i in body['instructions'] if i['opcode'] == opcode and (i['operand'] == operand if operand else 'RunFloatMatMulKernel' in i['operand'])]
                self.assertTrue(found); found[0]['operand'] = changed
                row['candidate_methods'][key] = json.dumps(body)
                with self.assertRaises(AssertionError): inventory(*args)

    def test_shared_flag(self):
        args = fixture(); row = args[0]['observations'][0]
        key = next(k for k in row['method_flags_after'] if '::RunFloatMatMulKernel::' in k)
        row['method_flags_after'][key] = 512
        with self.assertRaises(AssertionError): inventory(*args)

    def test_unrelated_method(self):
        args = fixture(); row = args[0]['observations'][0]
        row['differences'].append(next(k for k in row['normalized_methods'] if '::RunFloatMatMulKernel::' in k))
        row['unchanged_methods'] -= 1
        with self.assertRaises(AssertionError): inventory(*args)

    def test_new_method_or_data_change(self):
        args = fixture(); args[0]['observations'][0]['added'] = ['unexpected']
        with self.assertRaises(AssertionError): inventory(*args)
        args = fixture(); args[0]['observations'][1]['unchanged_methods'] -= 1
        with self.assertRaises(AssertionError): inventory(*args)

    def test_original_guard_identity(self):
        args = fixture(); prior = args[3]
        body = json.loads(prior['dispatcher_body']); body['instructions'][5]['operand'] = '45'
        prior['dispatcher_body'] = json.dumps(body)
        with self.assertRaises(AssertionError): inventory(*args)


if __name__ == '__main__': unittest.main()
