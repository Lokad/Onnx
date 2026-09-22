from copy import deepcopy
from pathlib import Path
import unittest
from protocol import read
from checks import check_result,consumer_inventory

ROOT=Path(__file__).resolve().parents[3]
OLD=ROOT/'artifacts/pyannote-lstm-wide-replay-amd-20260923'

class Checks(unittest.TestCase):
    def test_diagnostic_flags(self):
        payload=read(OLD/'payload.json');payload['consumer']=read(OLD/'collected/built.json')['consumer']
        capture=read(OLD/'collected/references/capture.json');native=read(OLD/'collected/references/native.json')
        for role in ['selected','candidate']:
            for width in ['256','512','scalar','simd']:
                value=read(OLD/'collected'/(role+'-'+width)/'result.json');value['flags']+=['DOTNET_JitDisasm']
                check_result(value,role,width,payload,None,capture,native)
                value['flags']+=['DOTNET_TieredPGO']
                with self.assertRaises(AssertionError):check_result(value,role,width,payload,None,capture,native)

    def test_exact_lambdas(self):
        keys=['ModelReplay+<>c::<Main>b__4_2::Boolean <Main>b__4_2(System.String)',
            'ModelReplay+<>c__DisplayClass4_0::<Main>b__1::Boolean <Main>b__1(System.String)']
        methods={k:'old' for k in keys};methods.update({str(i):'same' for i in range(53)})
        row=dict(assembly='LstmModelReplay.dll',before_sha256='old',after_sha256='new',public_surface_equal=True,
            added=[],removed=[],differences=keys,candidate_methods={k:'new' for k in keys},normalized_methods=methods,methods=55,unchanged_methods=53)
        value=dict(inventory_complete=True,observations=[row]);old=dict(sha256='old');new=dict(sha256='new');consumer_inventory(value,old,new)
        for change in [lambda r:r['differences'].append('Main'),lambda r:r.update(unchanged_methods=54),lambda r:r['added'].append('extra')]:
            damaged=deepcopy(value);change(damaged['observations'][0])
            with self.assertRaises(AssertionError):consumer_inventory(damaged,old,new)

if __name__=='__main__':unittest.main()
