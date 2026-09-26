"""Reject altered identity checks, numerical work and unrelated consumer methods."""
import json
from pathlib import Path
import unittest
from scope import source,expected_main,inventory,OLD_DATA,NEW_USAGE

ROOT=Path(__file__).resolve().parents[3]
CURRENT=ROOT/'artifacts/parakeet-observed-dense-where-pyannote-amd-20260924'
OLD=ROOT/'artifacts/parakeet-packed-final-row-pyannote-amd-20260925/collected/consumer-inventory/instructions.json'


class Scope(unittest.TestCase):
    def fixture(self):
        value=json.loads(OLD.read_text())
        row=value['observations'][0];key,=row['differences']
        row['candidate_methods'][key]=json.dumps(expected_main(json.loads(row['normalized_methods'][key])))
        return value,dict(sha256=row['before_sha256']),dict(sha256=row['after_sha256']),key

    def test_actual_source_changes_only_two_lines(self):
        before=(CURRENT/'bundle/consumer/Program.cs').read_bytes();after=source(before)
        self.assertEqual(sum(a!=b for a,b in zip(before.splitlines(),after.splitlines(),strict=True)),2)
        self.assertIn(b'== args[4], "Qualified data"',after)
        self.assertIn(NEW_USAGE.encode(),after)
        self.assertNotIn(OLD_DATA.encode(),after)

    def test_complete_expected_instruction_change(self):
        value,before,after,_=self.fixture()
        self.assertTrue(inventory(value,before,after)['assembly_hash_comparison_preserved'])

    def test_prior_literal_only_consumer_is_rejected(self):
        value=json.loads(OLD.read_text());row=value['observations'][0]
        with self.assertRaises(AssertionError):inventory(value,dict(sha256=row['before_sha256']),dict(sha256=row['after_sha256']))

    def test_removed_identity_comparison_is_rejected(self):
        value,before,after,key=self.fixture();row=value['observations'][0]
        main=json.loads(row['candidate_methods'][key])
        next(r for r in main['instructions'] if r['opcode']=='ldstr' and r['operand']=='Qualified data')['operand']='Ignored data'
        row['candidate_methods'][key]=json.dumps(main)
        with self.assertRaises(AssertionError):inventory(value,before,after)

    def test_changed_non_main_method_is_rejected(self):
        value,before,after,_=self.fixture();value['observations'][0]['unchanged_methods']=94
        with self.assertRaises(AssertionError):inventory(value,before,after)

    def test_changed_numeric_work_is_rejected(self):
        value,before,after,key=self.fixture();row=value['observations'][0]
        main=json.loads(row['candidate_methods'][key]);main['instructions'][-1]['opcode']='nop'
        row['candidate_methods'][key]=json.dumps(main)
        with self.assertRaises(AssertionError):inventory(value,before,after)

    def test_unexpected_identity_literal_is_rejected_before_build(self):
        before=(CURRENT/'bundle/consumer/Program.cs').read_bytes().replace(OLD_DATA.encode(),b'0'*64)
        with self.assertRaises(AssertionError):source(before)


if __name__=='__main__':unittest.main()
