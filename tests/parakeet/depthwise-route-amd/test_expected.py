import copy
import json
from pathlib import Path
import unittest
from expected import expected,audit_counts

ROOT=Path(__file__).resolve().parents[3]


class Counts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.diagnosis=json.loads((ROOT/'artifacts/parakeet-stem-diagnosis-20260925/analysis.json').read_text())
    def report(self):
        rows=[]
        for key,row in expected(self.diagnosis).items():
            value=copy.deepcopy(row);requested=value.pop('requested_scratch')
            value.update(key=key,simd=True,intrinsics=True,segmented=False,degree=1,
                layouts={'example-layout':value['calls']},scratch_elements={str(requested//4):value['tiled_batches']})
            rows.append(value)
        return dict(passed=True,protocol='parakeet-depthwise-route-v1',runtime='10.0.8',processor_count=1,flags={},
            fma=True,avx2=True,avx512=True,vector_width=8,rows=rows)
    def test_complete_shape_census(self):self.assertEqual(3948544,audit_counts(self.report(),self.diagnosis)['per_corpus']['products'])
    def test_missing_shape_rejected(self):
        r=self.report();r['rows'].pop()
        with self.assertRaises(AssertionError):audit_counts(r,self.diagnosis)
    def test_wrong_leaf_rejected(self):
        r=self.report();r['rows'][0]['leaves']={'scalar':r['rows'][0]['products']}
        with self.assertRaises(AssertionError):audit_counts(r,self.diagnosis)
    def test_partial_panel_change_rejected(self):
        r=self.report();r['rows'][0]['panel_widths']['31']=1
        with self.assertRaises(AssertionError):audit_counts(r,self.diagnosis)
    def test_physical_bucket_cannot_be_smaller_than_request(self):
        r=self.report();r['rows'][0]['scratch_elements']={'1':r['rows'][0]['tiled_batches']}
        with self.assertRaises(AssertionError):audit_counts(r,self.diagnosis)


if __name__=='__main__':unittest.main()
