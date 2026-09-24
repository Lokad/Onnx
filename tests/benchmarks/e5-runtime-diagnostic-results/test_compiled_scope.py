"""Exercise compiled admission against the actual AMD consumer and deliberate damage."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/benchmarks/e5-runtime-diagnostic-amd'))
from checks import compiled_scope
from protocol import read
BASE=ROOT/'artifacts/e5-runtime-diagnostic-amd-20260924'


class Scope(unittest.TestCase):
    def setUp(self):
        self.value=read(BASE/'collected/consumer-inventory/instructions.json')
        self.spec=read(BASE/'payload.json');self.built=read(BASE/'collected/built.json')

    def check(self):return compiled_scope(self.value,self.spec,self.built)

    def change_main(self,edit):
        row=self.value['observations'][0];key,=row['differences'];body=json.loads(row['candidate_methods'][key])
        edit(body);row['candidate_methods'][key]=json.dumps(body)

    def test_actual_compiled_scope(self):self.assertEqual(self.check()['unchanged_methods'],65)

    def test_assertion_replacement_rejected(self):
        def edit(body):
            row=next(r for r in body['instructions'] if r['opcode']=='call' and 'g__Require' in str(r['operand']))
            row['operand']='Program::Void DiscardAssertion(Boolean, System.String)'
        self.change_main(edit)
        with self.assertRaises(AssertionError):self.check()

    def test_changed_branch_rejected(self):
        def edit(body):
            row=next(r for r in body['instructions'] if r['opcode']=='brtrue.s')
            row['operand']='00'
        self.change_main(edit)
        with self.assertRaises(AssertionError):self.check()

    def test_method_flag_change_rejected(self):
        row=self.value['observations'][0];key=next(iter(row['method_flags_before']))
        row['method_flags_after'][key]^=8
        with self.assertRaises(AssertionError):self.check()


if __name__=='__main__':unittest.main()
