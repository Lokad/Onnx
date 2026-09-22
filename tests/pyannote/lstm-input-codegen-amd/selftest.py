"""Reject missing recurrent outputs, altered values/ownership, and wrong routes."""
import copy
from pathlib import Path
import shutil
import tempfile
import unittest
from checks import check_result, check_suite
from protocol import pin, read

ROOT=Path(__file__).resolve().parents[3]
LOCAL=ROOT/'artifacts/pyannote-lstm-input-blocks-v6-20260922'
FIXTURES=ROOT/'artifacts/pyannote-lstm-input-fixtures-v3-20260922'


class Checks(unittest.TestCase):
    def test_complete_and_mutations(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)
            for name in ['output','native']:
                (base/'fixtures'/name).mkdir(parents=True)
                shutil.copy2(FIXTURES/name/'result.json',base/'fixtures'/name/'result.json')
            original=read(LOCAL/'output/candidate-256.json');original['flags']=['DOTNET_EnableAVX512','DOTNET_JitDisasm']
            spec=dict(cores={'candidate':pin(LOCAL/'runtime/Lokad.Onnx.dll')},consumer=pin(LOCAL/'runtime/LstmModelReplay.dll'))
            self.assertTrue(check_result(original,'candidate','256',spec,base)['passed'])
            def mutate(action):
                value=copy.deepcopy(original);action(value)
                with self.assertRaises(AssertionError):check_result(value,'candidate','256',spec,base)
            mutate(lambda v:v['observations'].pop())
            mutate(lambda v:v['observations'][0].update(sha256='0'*64))
            mutate(lambda v:v['observations'][0].update(slot=1))
            mutate(lambda v:v['observations'][0].update(scratch_bytes=0))
            mutate(lambda v:v.update(held_outputs_unchanged=False))
            mutate(lambda v:v.update(readonly_operands=False))
            mutate(lambda v:v.update(avx512=True))
            mutate(lambda v:v.update(maximum=2e-4))
            mutate(lambda v:v.update(vector_count=16))
    def test_full_suite_census(self):
        path=LOCAL/'test-results/lstm-ordinary.trx'
        self.assertEqual(150,check_suite(path,path)['tests'])
        with self.assertRaises(AssertionError):check_suite(LOCAL/'test-results/lstm-scalar.trx',path)


if __name__=='__main__':unittest.main()
