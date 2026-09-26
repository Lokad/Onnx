"""Protect the intentional float exception without admitting changed diarization."""
import copy
import importlib.util
from pathlib import Path
import unittest
from semantics import compare_records

ROOT = Path(__file__).resolve().parents[3]
path = ROOT/'artifacts/pyannote-blocked-spatial-app-amd-payload-20260922/payload/runtime/protocol.py'
spec = importlib.util.spec_from_file_location('original_audio_contract', path)
original = importlib.util.module_from_spec(spec); spec.loader.exec_module(original)

class PublicContracts(unittest.TestCase):
    def setUp(self):
        self.result = dict(status='Completed',windows=1,audio_seconds=10.,intervals=[[0.,1.,0]],
            exclusive_intervals=[[0.,1.,0]],speakers=[dict(speaker=0,centroid=[.1]*256,has_embedding=True)])
        self.rows = [dict(name='crop',phase='warmup',**{'pass':0},result=copy.deepcopy(self.result))]
    def test_changed_float_is_diagnostic_and_native_bounded(self):
        other=copy.deepcopy(self.rows);other[0]['result']['speakers'][0]['centroid'][0]+=1e-6
        result=compare_records(other,self.rows,exact=False)
        self.assertTrue(result['semantic_exact']);self.assertFalse(result['full_results_exact'])
        self.assertLess(original.check_result(other[0]['result'],self.result,family='pyannote'),1e-4)
    def test_native_rejects_excess_error(self):
        other=copy.deepcopy(self.result);other['speakers'][0]['centroid'][0]+=.01
        with self.assertRaises(AssertionError):original.check_result(other,self.result,family='pyannote')
    def test_own_product_float_repeat_remains_exact(self):
        other=copy.deepcopy(self.rows);other[0]['result']['speakers'][0]['centroid'][0]+=1e-12
        with self.assertRaises(AssertionError):compare_records(other,self.rows,exact=True)
    def test_every_semantic_field(self):
        for field,value in [('status','NoSpeech'),('windows',2),('audio_seconds',11.),('intervals',[]),('exclusive_intervals',[])]:
            with self.subTest(field=field):
                other=copy.deepcopy(self.rows);other[0]['result'][field]=value
                with self.assertRaises(AssertionError):compare_records(other,self.rows,exact=False)
        for field,value in [('speaker',1),('has_embedding',False)]:
            other=copy.deepcopy(self.rows);other[0]['result']['speakers'][0][field]=value
            with self.assertRaises(AssertionError):compare_records(other,self.rows,exact=False)
    def test_last_bit_timeline_change(self):
        other=copy.deepcopy(self.rows);other[0]['result']['intervals'][0][1]=1.0000000000000002
        with self.assertRaises(AssertionError):compare_records(other,self.rows,exact=False)
    def test_unknown_fields(self):
        for speaker in [False,True]:
            other=copy.deepcopy(self.rows);target=other[0]['result']['speakers'][0] if speaker else other[0]['result'];target['unknown']=0
            with self.assertRaises(AssertionError):compare_records(other,self.rows,exact=False)
    def test_nonfinite_or_missing_coordinates(self):
        for centroid in [[.1]*255,[float('nan')]*256,[float('inf')]*256]:
            other=copy.deepcopy(self.rows);other[0]['result']['speakers'][0]['centroid']=centroid
            with self.assertRaises(AssertionError):compare_records(other,self.rows,exact=False)
    def test_case_order_and_coverage(self):
        for key,value in [('name','different'),('phase','measured'),('pass',1)]:
            other=copy.deepcopy(self.rows);other[0][key]=value
            with self.assertRaises(AssertionError):compare_records(other,self.rows,exact=False)
        with self.assertRaises(AssertionError):compare_records([],self.rows,exact=False)

if __name__=='__main__': unittest.main()
