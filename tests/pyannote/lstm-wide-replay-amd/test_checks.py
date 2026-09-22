"""Mutation checks against real complete AMD outputs and strict consumer scope."""
from copy import deepcopy
from pathlib import Path
import unittest
from protocol import read
from checks import check_result,consumer_inventory

ROOT=Path(__file__).resolve().parents[3]
OLD=ROOT/'artifacts/pyannote-lstm-input-blocks-amd-v2-20260922'


class Checks(unittest.TestCase):
    def test_complete_output_mutations(self):
        value=read(OLD/'collected/candidate-512/result.json')
        capture=read(OLD/'collected/references/capture.json');native=read(OLD/'collected/references/native.json')
        spec=dict(cores={r:dict(sha256=value['core']) for r in ['selected','candidate']},consumer=dict(sha256=value['executable']))
        for role in ['selected','candidate']:
            for width in ['512','simd']:
                actual=dict(value,role=role,width=width);check_result(actual,role,width,spec,None,capture,native)
        mutations=[lambda v:v.update(outputs=71),lambda v:v.update(values=0),lambda v:v.update(core='bad'),
            lambda v:v.update(held_outputs_unchanged=False),lambda v:v.update(readonly_operands=False),
            lambda v:v.update(flags=['DOTNET_TieredPGO']),lambda v:v.update(avx512=False),
            lambda v:v['observations'][0].update(sha256='bad'),lambda v:v['observations'][0].update(scratch_bytes=0),
            lambda v:v['observations'][0].update(native_maximum=.01),lambda v:v['observations'][0].update(exact=False),
            lambda v:v['observations'].reverse()]
        for mutate in mutations:
            damaged=deepcopy(value);mutate(damaged)
            with self.assertRaises(AssertionError):check_result(damaged,'candidate','512',spec,None,capture,native)

    def test_consumer_scope(self):
        main='ModelReplay::Main::signature';other='ModelReplay::Hash::signature'
        row=dict(assembly='LstmModelReplay.dll',before_sha256='old',after_sha256='new',public_surface_equal=True,
            added=[],removed=[],differences=[main],candidate_methods={main:'after'},normalized_methods={main:'before',other:'same'},methods=2,unchanged_methods=1)
        value=dict(inventory_complete=True,observations=[row]);old=dict(sha256='old');new=dict(sha256='new');consumer_inventory(value,old,new)
        mutations=[lambda r:r.update(public_surface_equal=False),lambda r:r.update(added=['extra']),lambda r:r.update(removed=[other]),
            lambda r:r['differences'].append(other),lambda r:r['candidate_methods'].update({other:'different'}),
            lambda r:r.update(unchanged_methods=2),lambda r:r.update(before_sha256='wrong')]
        for mutate in mutations:
            damaged=deepcopy(value);mutate(damaged['observations'][0])
            with self.assertRaises((AssertionError,ValueError)):consumer_inventory(damaged,old,new)


if __name__=='__main__':unittest.main()
