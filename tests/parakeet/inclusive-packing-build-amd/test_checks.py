"""Reject undeclared instruction, resource-policy and method-scope changes."""
import copy
import json
from pathlib import Path
import unittest
from checks import HELPER, expected_body, inventory

ROOT = Path(__file__).resolve().parents[3]
PARENT = ROOT / 'artifacts/parakeet-wide-entry-first-use-build-amd-20260923'


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
    prior = dict(passed=True, helper_key=HELPER, helper_body=row['normalized_methods'][HELPER])
    row.update(candidate_methods={HELPER: json.dumps(expected_body(prior['helper_body']))},
               differences=[HELPER], unchanged_methods=3188)
    return value, measured, built, prior


class ScopeTests(unittest.TestCase):
    def test_declared_comparison(self):
        self.assertEqual(inventory(*fixture())['opcode_after'], 'bgt.s')

    def test_every_other_instruction_and_operand(self):
        for index in range(22):
            args = fixture()
            body = expected_body(args[3]['helper_body'])
            body['instructions'][index]['operand'] += '01'
            args[0]['observations'][0]['candidate_methods'][HELPER] = json.dumps(body)
            with self.subTest(index=index), self.assertRaises(AssertionError): inventory(*args)

    def test_wrong_comparison_or_branch(self):
        for opcode in ['bge.s', 'bgt.un.s', 'blt.s']:
            args = fixture()
            body = expected_body(args[3]['helper_body'])
            body['instructions'][8]['opcode'] = opcode
            args[0]['observations'][0]['candidate_methods'][HELPER] = json.dumps(body)
            with self.subTest(opcode=opcode), self.assertRaises(AssertionError): inventory(*args)

    def test_headers_and_flags(self):
        for field, changed in [('InitLocals', True), ('MaxStackSize', 9), ('locals', ['new']), ('exceptions', ['new'])]:
            args = fixture(); body = expected_body(args[3]['helper_body']); body[field] = changed
            args[0]['observations'][0]['candidate_methods'][HELPER] = json.dumps(body)
            with self.subTest(field=field), self.assertRaises(AssertionError): inventory(*args)
        args = fixture(); args[0]['observations'][0]['method_flags_after'][HELPER] = 520
        with self.assertRaises(AssertionError): inventory(*args)

    def test_unrelated_core_data_and_interface_changes(self):
        for assembly in range(2):
            for field, changed in [('differences', ['unrelated']), ('public_surface_equal', False),
                                   ('added', ['new']), ('removed', ['old']), ('compiler_rename', {})]:
                args = fixture(); args[0]['observations'][assembly][field] = changed
                with self.subTest(assembly=assembly, field=field), self.assertRaises(AssertionError): inventory(*args)


if __name__ == '__main__': unittest.main()
