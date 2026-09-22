"""Exercise the verifier with the real failed inventory and deliberate scope changes."""
import copy
from pathlib import Path
import unittest
from protocol import read
from consumer_checks import consumer_inventory, OLD

ROOT=Path(__file__).resolve().parents[3]
FAILED=ROOT/'artifacts/pyannote-kernel-loop-numerics-amd-20260922'
VALUE=read(FAILED/'collected/raw-inventory/instructions.json')
SPEC=read(FAILED/'payload.json')
BUILT=read(FAILED/'collected/consumer-built.json')


def check(value):
    return consumer_inventory(value,'raw',SPEC['prior_consumers']['raw'],BUILT['consumers']['raw'],SPEC['core']['sha256'])


class ConsumerTests(unittest.TestCase):
    def test_actual_raw_consumer(self):self.assertTrue(check(VALUE)['passed'])

    def test_source_literal_propagates_twice(self):
        row=VALUE['observations'][0];key,=row['differences']
        self.assertEqual(row['normalized_methods'][key].count(OLD),2)

    def test_scope_mutations(self):
        for kind in ['extra_change','missing_method','public_change','renamed','arithmetic_change','unreplaced_identity']:
            with self.subTest(kind=kind):
                value=copy.deepcopy(VALUE);row=value['observations'][0];key,=row['differences']
                if kind=='extra_change':row['differences'].append('Probe::Supplemental::x')
                elif kind=='missing_method':row['normalized_methods'].pop(key)
                elif kind=='public_change':row['public_surface_equal']=False
                elif kind=='renamed':row['compiler_rename']='x'
                elif kind=='arithmetic_change':row['candidate_methods'][key]+=' '
                else:row['candidate_methods'][key]=row['normalized_methods'][key]
                with self.assertRaises((AssertionError,ValueError)):check(value)


if __name__=='__main__':unittest.main(verbosity=2)
