"""Reject changes to workload, assertions and exact call-boundary records."""
import copy
from pathlib import Path
import unittest
from source_scope import instrument,verify
from checks import merge_observation
from protocol import read
from codegen import SIGNATURES,reconcile_listings
ROOT=Path(__file__).resolve().parents[3]
PARENT=ROOT/'artifacts/e5-direct-tier-diagnostic-v2-amd-20260925/collected'


class Checks(unittest.TestCase):
    def source(self):return (PARENT/'source/consumer/Program.cs').read_text()
    def test_exact_reversal(self):
        source=self.source();self.assertTrue(verify(source,instrument(source))['original_recovered_exactly'])
    def test_numerical_bound(self):
        source=self.source()
        with self.assertRaises(AssertionError):verify(source,instrument(source).replace('error <= 1e-4','error <= 1e-3'))
    def test_warmup(self):
        source=self.source()
        with self.assertRaises(AssertionError):verify(source,instrument(source).replace('index < 600','index < 1200'))
    def test_extent(self):
        source=self.source()
        with self.assertRaises(AssertionError):verify(source,instrument(source).replace('3 : 6000','3 : 6600'))
    def test_owned_output(self):
        source=self.source()
        with self.assertRaises(AssertionError):verify(source,instrument(source).replace('Hash(first![j]) == hashes[j]','true'))
    def test_clock_mismatch(self):
        value=read(PARENT/'a-capture/output/result.json');diagnostic=read(PARENT/'a-capture/output/diagnostic.json')
        self.assertEqual(len(merge_observation(value,diagnostic)['clocks']),6000)
        diagnostic['clocks'][620]['end']+=1
        with self.assertRaises(AssertionError):merge_observation(value,diagnostic)
    def test_reordered_clock(self):
        value=read(PARENT/'a-capture/output/result.json');diagnostic=read(PARENT/'a-capture/output/diagnostic.json')
        diagnostic['clocks'][620],diagnostic['clocks'][621]=diagnostic['clocks'][621],diagnostic['clocks'][620]
        with self.assertRaises(AssertionError):merge_observation(value,diagnostic)

    def code_fixture(self):
        listing='';events=[]
        for index,(name,signature) in enumerate(SIGNATURES.items()):
            listing+=f'; Assembly listing for method {signature} (Tier1)\nG_M000_IG01: ;; offset=0x0000\n       C3                   ret\n; Total bytes of code 1\n'
            events.append(dict(provider='Microsoft-Windows-DotNETRuntime',name='Method/LoadVerbose',pid=1,
                payload=dict(MethodNamespace='Lokad.Onnx.Tensor`1[System.Single]',MethodName=name,MethodID=str(index),
                    MethodSignature=signature,OptimizationTier='OptimizedTier1',MethodSize='1')))
        return listing,events
    def test_missing_listing_rejected(self):
        listing,events=self.code_fixture();self.assertTrue(reconcile_listings(listing,events,1)['passed'])
        with self.assertRaises(AssertionError):reconcile_listings(listing.rsplit('; Total bytes of code ',1)[0],events,1)
    def test_wrong_native_size_rejected(self):
        listing,events=self.code_fixture();events[0]['payload']['MethodSize']='2'
        with self.assertRaises(AssertionError):reconcile_listings(listing,events,1)
    def test_wrong_tier_rejected(self):
        listing,events=self.code_fixture();events[0]['payload']['OptimizationTier']='QuickJitted'
        with self.assertRaises(AssertionError):reconcile_listings(listing,events,1)


if __name__=='__main__':unittest.main()
