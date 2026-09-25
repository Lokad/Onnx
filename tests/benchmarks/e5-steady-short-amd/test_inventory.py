"""Verify scope using actual retained AMD instructions and deliberate damage."""
import copy,json
from pathlib import Path
import unittest
from checks import consumer_inventory
from scope import verify,expected

ROOT=Path(__file__).resolve().parents[3]


def fixture():
    path=ROOT/'artifacts/warmed-release-amd-v2-20260923/collected/consumer-inventory/instructions.json'
    value=json.loads(path.read_text());row=value['observations'][0];key,=row['differences']
    body=json.loads(row['candidate_methods'][key])
    row['normalized_methods'][key]=json.dumps(body)
    edits={'0C030000':'24180000','58020000':'70170000'};changed=[]
    for instruction in body['instructions']:
        if instruction['opcode']=='ldc.i4' and instruction['operand'] in edits:
            changed.append(instruction['operand']);instruction['operand']=edits[instruction['operand']]
    assert changed==list(edits)
    row['candidate_methods'][key]=json.dumps(body)
    row['before_sha256']='before';row['after_sha256']='after'
    return value,dict(previous_consumer=dict(sha256='before')),dict(consumer=dict(sha256='after'))


class Inventory(unittest.TestCase):
    def test_every_source_change_is_declared(self):self.assertEqual(len(verify()),10)

    def test_actual_instruction_shape_two_constants_only(self):
        result=consumer_inventory(*fixture())
        self.assertEqual([(r['before'],r['after']) for r in result['changes']],[(780,6180),(600,6000)])

    def test_numerical_assertion_replacement_rejected(self):
        value,spec,built=fixture();row=value['observations'][0];key,=row['differences'];body=json.loads(row['candidate_methods'][key])
        instruction=next(r for r in body['instructions'] if r['opcode']=='call' and 'g__Require' in str(r['operand']))
        instruction['operand']='Program::Void DropAssertion(Boolean, System.String)'
        row['candidate_methods'][key]=json.dumps(body)
        with self.assertRaises(AssertionError):consumer_inventory(value,spec,built)

    def test_changed_branch_rejected(self):
        value,spec,built=fixture();row=value['observations'][0];key,=row['differences'];body=json.loads(row['candidate_methods'][key])
        next(r for r in body['instructions'] if r['opcode']=='brtrue.s')['operand']='00'
        row['candidate_methods'][key]=json.dumps(body)
        with self.assertRaises(AssertionError):consumer_inventory(value,spec,built)

    def test_changed_flags_rejected(self):
        value,spec,built=fixture();row=value['observations'][0];key=next(iter(row['method_flags_after']))
        row['method_flags_after'][key]^=8
        with self.assertRaises(AssertionError):consumer_inventory(value,spec,built)

    def test_wrong_product_binding_rejected(self):
        value,spec,built=fixture();built['consumer']['sha256']='different'
        with self.assertRaises(AssertionError):consumer_inventory(value,spec,built)


if __name__=='__main__':unittest.main()
