"""Mutation checks against the actual compiled inventory, including identifier collisions."""
import copy
import json
from pathlib import Path
import sys
import unittest
from renames import normalize

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/parakeet-prepared-recurrence-build-amd-20260924'
sys.path.insert(0,str(ROOT/'tests/parakeet/prepared-recurrence-build-amd'))
from checks import inventory


class RenameChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=json.loads((BASE/'collected/inventory/instructions.json').read_text())
        cls.measured=json.loads((BASE/'payload.json').read_text())['measured']
        cls.built=json.loads((BASE/'collected/built.json').read_text())['product']
        cls.prior=json.loads((BASE/'collected/evidence/prior-composition.json').read_text())

    def audit(self,value):
        normalized,report=normalize(value)
        return inventory(normalized,self.measured,self.built,self.prior),report

    def test_actual_inventory_and_colliding_overload_identifiers(self):
        result,report=self.audit(self.raw)
        self.assertEqual((result['candidate_core_methods'],result['unchanged_core_methods']),(3250,3179))
        self.assertEqual((len(report['renamed_methods']),len(report['name_only_methods'])),(28,35))
        self.assertEqual(len(result['existing_method_changes']),10)

    def test_generated_lambda_iterator_and_caller_instructions_cannot_change(self):
        for pattern in ['::<ResolveInputs>b__185_0::','<EnumerateAttributeTensors>d__234::MoveNext::','::RunNodeCoreInner::']:
            value=copy.deepcopy(self.raw);row=value['observations'][0]
            key=next(k for k in row['candidate_methods'] if pattern in k)
            body=json.loads(row['candidate_methods'][key]);body['instructions'][0]['opcode']='break'
            row['candidate_methods'][key]=json.dumps(body)
            with self.assertRaises(AssertionError):self.audit(value)

    def test_generated_flags_and_unrelated_kernel_flags_cannot_change(self):
        for pattern in ['::<ResolveInputs>b__185_0::','::LstmProjectOrdered::']:
            value=copy.deepcopy(self.raw);row=value['observations'][0]
            key=next(k for k in row['method_flags_after'] if pattern in k)
            row['method_flags_after'][key]^=8
            with self.assertRaises(AssertionError):self.audit(value)

    def test_public_surface_or_new_helpers_outside_declared_types_are_refused(self):
        value=copy.deepcopy(self.raw);value['observations'][0]['public_surface_equal']=False
        with self.assertRaises(AssertionError):self.audit(value)
        value=copy.deepcopy(self.raw);row=value['observations'][0]
        key=next(k for k in row['added'] if k.startswith('Lokad.Onnx.GraphLstmPacking::Resolve::'))
        other=key.replace('GraphLstmPacking','UnrelatedHelper');row['added'][row['added'].index(key)]=other
        row['candidate_methods'][other]=row['candidate_methods'].pop(key)
        row['method_flags_after'][other]=row['method_flags_after'].pop(key)
        with self.assertRaises(AssertionError):self.audit(value)


if __name__=='__main__':unittest.main()
