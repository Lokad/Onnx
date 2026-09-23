"""Reject consumer drift and missing channel coverage using retained real inventories."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from consumer_checks import OLD,CHANNELS,consumer_inventory,expected_channels
from checks import channel_census

ROOT=Path(__file__).resolve().parents[3]
NUM=ROOT/'artifacts/pyannote-kernel-loop-numerics-amd-v2-20260922/collected'
KEY='Probe::Main::Int32 Main(System.String[])'

class Checks(unittest.TestCase):
    def setUp(self):
        self.value=json.loads((NUM/'raw-inventory/instructions.json').read_text())
        row=self.value['observations'][0]
        self.before=dict(sha256=row['before_sha256']);self.after=dict(sha256=row['after_sha256']);self.core='new-core'

    def value_for(self,mode):
        value=deepcopy(self.value);row=value['observations'][0]
        text=row['normalized_methods'][KEY].replace(OLD,self.core)
        row['candidate_methods'][KEY]=json.dumps(expected_channels(text,CHANNELS[mode])) if mode in CHANNELS else text
        return value

    def test_original_and_both_channel_lists(self):
        for mode in ['raw',*CHANNELS]:
            value=self.value_for(mode);consumer_inventory(value,mode,self.before,self.after,self.core)
        small=expected_channels(self.value['observations'][0]['normalized_methods'][KEY],CHANNELS['channels64'])
        large=expected_channels(self.value['observations'][0]['normalized_methods'][KEY],CHANNELS['channels128'])
        old=json.loads(self.value['observations'][0]['normalized_methods'][KEY])
        self.assertEqual(small['exceptions'],old['exceptions'])
        self.assertEqual(large['exceptions'][0]['TryOffset'],old['exceptions'][0]['TryOffset']+6)
        self.assertEqual(large['exceptions'][0]['HandlerOffset'],old['exceptions'][0]['HandlerOffset']+6)

    def test_unrelated_instruction_and_exception_changes_fail(self):
        value=self.value_for('channels128')
        mutations=[lambda b:b.update(MaxStackSize=b['MaxStackSize']+1),
            lambda b:b['locals'].pop(),lambda b:b['exceptions'][0].update(TryLength=1),
            lambda b:b['instructions'][0].update(opcode='nop'),
            lambda b:next(r for r in b['instructions'] if r['opcode']=='br').update(operand='00000000'),
            lambda b:next(r for r in b['instructions'] if r['offset']==422).update(operand='81000000'),
            lambda b:b['instructions'][-1].update(offset=999999)]
        for mutate in mutations:
            damaged=deepcopy(value);row=damaged['observations'][0];body=json.loads(row['candidate_methods'][KEY]);mutate(body);row['candidate_methods'][KEY]=json.dumps(body)
            with self.assertRaises(AssertionError):consumer_inventory(damaged,'channels128',self.before,self.after,self.core)
        for key,new in [('added',['extra']),('removed',['missing']),('public_surface_equal',False),('unchanged_methods',143)]:
            damaged=deepcopy(value);damaged['observations'][0][key]=new
            with self.assertRaises(AssertionError):consumer_inventory(damaged,'channels128',self.before,self.after,self.core)

    def test_complete_channel_census(self):
        result=json.loads((NUM/'raw-512/result.json').read_text())
        for row in result['observations']:row['c']={16:64,32:80}[row['c']]
        channel_census(result,(64,80))
        mutations=[lambda v:v['observations'].pop(),lambda v:v['observations'].reverse(),
            lambda v:v['observations'][0].update(c=16),lambda v:v['observations'][0].update(stride=3),
            lambda v:v['observations'][0].update(scalar_differences=1),lambda v:v.update(graph_differences=1),
            lambda v:v['supplemental'][0].update(differences=1)]
        for mutate in mutations:
            damaged=deepcopy(result);mutate(damaged)
            with self.assertRaises(AssertionError):channel_census(damaged,(64,80))

if __name__=='__main__':unittest.main()
