import copy
import json
from pathlib import Path
import unittest
from expected import expected, audit_counts

ROOT = Path(__file__).resolve().parents[3]


class DirectCounts(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.diagnosis = json.loads((ROOT/'artifacts/parakeet-stem-diagnosis-20260925/analysis.json').read_text())
    def report(self):
        rows=[]
        for key, row in expected(self.diagnosis).items():
            value = copy.deepcopy(row)
            layout = 'Lokad.Onnx.DenseTensor`1[System.Single]:False:9,1:9'
            value.update(key=key, simd=True, intrinsics=True, segmented=False, degree=1,
                         layouts={layout+' -> '+layout+' | '+layout+' -> '+layout:value['calls']})
            rows.append(value)
        return dict(passed=True,protocol='parakeet-direct-depthwise-route-v1',runtime='10.0.8',processor_count=1,
                    flags={},fma=True,avx2=True,avx512=True,vector_width=8,rows=rows)
    def test_complete_direct_census(self):
        result=audit_counts(self.report(),self.diagnosis)
        self.assertEqual(520,result['per_corpus']['direct_batches']);self.assertEqual(0,result['per_corpus']['products'])
    def test_missing_geometry_rejected(self):
        r=self.report();r['rows'].pop()
        with self.assertRaises(AssertionError):audit_counts(r,self.diagnosis)
    def test_any_generic_work_rejected(self):
        for field in ['panels','products','views','patch_values','tiled_batches']:
            r=self.report();r['rows'][0][field]=1
            with self.assertRaises(AssertionError):audit_counts(r,self.diagnosis)
    def test_incomplete_direct_result_rejected(self):
        r=self.report();r['rows'][0]['direct_batches']-=1
        with self.assertRaises(AssertionError):audit_counts(r,self.diagnosis)
    def test_unexpected_scratch_or_matrix_leaf_rejected(self):
        for field in ['scratch_elements','leaves','matrix_shapes']:
            r=self.report();r['rows'][0][field]={'unexpected':1}
            with self.assertRaises(AssertionError):audit_counts(r,self.diagnosis)


if __name__=='__main__':unittest.main()
