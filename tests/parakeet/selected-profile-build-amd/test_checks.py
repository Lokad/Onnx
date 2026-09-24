"""Refuse hidden changes in retained Main IL and a complete synthetic flag census."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from checks import OLD, CURRENT, inventory


def fixture():
    root=Path(__file__).resolve().parents[3]
    old=json.loads((root/'artifacts/parakeet-current-profile-build-amd-20260923/collected/inventory/instructions.json').read_text())['observations'][0]
    methods={**old['normalized_methods'],**old['candidate_methods']}
    key,=old['differences'];body=json.loads(methods[key])
    for instruction in body['instructions']:
        if instruction['opcode']=='ldstr':
            for name,value in OLD.items():
                if instruction['operand']==value:instruction['operand']=CURRENT[name]
    flags={k:8 if k.startswith(('SampledRequests::WarmupParakeet::','SampledRequests::FullParakeet::')) else 0 for k in methods}
    row=dict(assembly='SampledAudio.dll',methods=162,unchanged_methods=161,public_surface_equal=True,
        removed=[],added=[],compiler_rename=None,before_sha256='old',after_sha256='new',
        differences=[key],normalized_methods=methods,candidate_methods={key:json.dumps(body)},
        method_flags_before=flags,method_flags_after=dict(flags))
    return dict(inventory_complete=True,observations=[row])


def check(value):return inventory(value,{'sha256':'old'},{'sha256':'new'},{n:{'sha256':v} for n,v in CURRENT.items()})


class Tests(unittest.TestCase):
    def test_exact_two_literals(self):
        result=check(fixture());self.assertEqual(result['changed_hash_literals'],2)
        self.assertGreater(result['main_instructions'],800)

    def test_unexpected_instruction_and_method_header(self):
        original=fixture()
        for field in ['opcode','operand','offset','MaxStackSize','InitLocals','locals','exceptions']:
            with self.subTest(field=field):
                value=deepcopy(original);r=value['observations'][0];key=r['differences'][0]
                body=json.loads(r['candidate_methods'][key])
                if field in ['opcode','operand','offset']:body['instructions'][0][field]='corrupted'
                else:body[field]='corrupted'
                r['candidate_methods'][key]=json.dumps(body)
                with self.assertRaises(AssertionError):check(value)

    def test_method_census_and_flags(self):
        original=fixture()
        for field in ['methods','unchanged_methods','removed','added','public_surface_equal','compiler_rename']:
            with self.subTest(field=field):
                value=deepcopy(original);r=value['observations'][0]
                r[field]=False if field=='public_surface_equal' else 'corrupted'
                with self.assertRaises(AssertionError):check(value)
        value=deepcopy(original);r=value['observations'][0]
        r['method_flags_after'][r['differences'][0]]=520
        with self.assertRaises(AssertionError):check(value)

    def test_missing_hash_change(self):
        value=fixture();r=value['observations'][0];key=r['differences'][0]
        r['candidate_methods'][key]=r['normalized_methods'][key]
        with self.assertRaises(AssertionError):check(value)

    def test_missing_method_flag(self):
        value=fixture();r=value['observations'][0];key=r['differences'][0]
        del r['method_flags_before'][key];del r['method_flags_after'][key]
        with self.assertRaises(AssertionError):check(value)

    def test_changed_marker_flags(self):
        value=fixture();r=value['observations'][0]
        key=next(k for k in r['method_flags_before'] if k.startswith('SampledRequests::FullParakeet::'))
        r['method_flags_before'][key]=r['method_flags_after'][key]=520
        with self.assertRaises(AssertionError):check(value)


if __name__=='__main__':unittest.main()
