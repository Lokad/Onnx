"""Reject altered original assertions and mismatched boundary records."""
import copy
import json
from pathlib import Path
import unittest
from source_scope import instrument, verify
from checks import merge_observation
from protocol import CALLS

ROOT=Path(__file__).resolve().parents[3]


class Checks(unittest.TestCase):
    def test_original_source_recovers(self):
        source=(ROOT/'tests/benchmarks/warmed-release-amd-v2/Program.cs').read_text()
        self.assertTrue(verify(source,instrument(source))['original_recovered_exactly'])

    def test_numerical_check_cannot_change(self):
        source=(ROOT/'tests/benchmarks/warmed-release-amd-v2/Program.cs').read_text()
        with self.assertRaises(AssertionError): verify(source,instrument(source).replace('error <= 1e-4','error <= 1e-3'))

    def test_warmup_cannot_change(self):
        source=(ROOT/'tests/benchmarks/warmed-release-amd-v2/Program.cs').read_text()
        with self.assertRaises(AssertionError): verify(source,instrument(source).replace('index < 600','index < 660'))

    def test_unplanned_extent_rejected(self):
        source=(ROOT/'tests/benchmarks/warmed-release-amd-v2/Program.cs').read_text()
        with self.assertRaises(AssertionError): verify(source,instrument(source).replace('3 : 6000','3 : 6600'))

    def test_owned_output_copy_cannot_disappear(self):
        source=(ROOT/'tests/benchmarks/warmed-release-amd-v2/Program.cs').read_text()
        with self.assertRaises(AssertionError): verify(source,instrument(source).replace('Hash(first![j]) == hashes[j]','true'))

    def test_clock_mismatch_rejected(self):
        # Actual retained duration records; fabricated boundaries are only a checker fixture.
        value=json.loads((ROOT/'artifacts/parakeet-validated-composition-graphs-amd-20260924/collected/timing-e5-8tok-current-a/output/result.json').read_text())
        original=copy.deepcopy(value['clocks']);value['clocks']=[dict(original[i%len(original)],index=i,warmup=i<600) for i in range(CALLS)];value['calls']=CALLS
        diagnostic=dict(diagnosticOnly=True,pid=value['pid'],nativeThread=1,clocks=[])
        for c in value['clocks']:
            diagnostic['clocks'].append(dict(index=c['index'],marker=1,start=2,end=2+c['ticks']))
        self.assertEqual(len(merge_observation(value,diagnostic)['clocks']),CALLS)
        bad=copy.deepcopy(diagnostic);bad['clocks'][620]['end']+=1
        with self.assertRaises(AssertionError):merge_observation(value,bad)

    def test_reordered_boundary_rejected(self):
        value=json.loads((ROOT/'artifacts/parakeet-validated-composition-graphs-amd-20260924/collected/timing-e5-8tok-current-a/output/result.json').read_text())
        original=copy.deepcopy(value['clocks']);value['clocks']=[dict(original[i%len(original)],index=i,warmup=i<600) for i in range(CALLS)];value['calls']=CALLS
        diagnostic=dict(diagnosticOnly=True,pid=value['pid'],nativeThread=1,
            clocks=[dict(index=c['index'],marker=1,start=2,end=2+c['ticks']) for c in value['clocks']])
        diagnostic['clocks'][620],diagnostic['clocks'][621]=diagnostic['clocks'][621],diagnostic['clocks'][620]
        with self.assertRaises(AssertionError):merge_observation(value,diagnostic)


if __name__=='__main__':unittest.main()
