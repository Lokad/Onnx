import copy
import unittest
from audit import error,result,same_previous,same_current,identity,raw_census
import itertools


def metric(count):
    return dict(Count=count,Absolute=0.,Scaled=0.,WorstIndex=0,Actual=0.,Reference=0.,DifferentBits=0,Passed=True)


def sample():
    rows=[]
    for index,((c,m),(h,w),pattern,epilogue) in enumerate(raw_census()):
        rows.append(dict(index=index,c=c,m=m,h=h,w=w,pattern=pattern,epilogue=epilogue,
            candidate=metric(m*h*w),current=metric(m*h*w),selectedDifference=metric(m*h*w),
            repeated=True,heldOutput=True,readonlyInputs=True,guards=True,output='a'*64))
    return dict(mode='raw',width=512,runtime='10.0.8',completed=True,noPerformanceMeasurement=True,rows=rows,refusals=21,contracts=24,rangeChecks=10566,singleBlockRefusals=2)


class AuditTests(unittest.TestCase):
    def test_previous_identity_may_change(self):
        x=sample();y=copy.deepcopy(x);y['rows']=y['rows'][:1920];y.update(pid=12,assembly='b'*64)
        self.assertTrue(same_previous(x,y))
    def test_previous_output_cannot_change(self):
        x=sample();y=copy.deepcopy(x);y['rows']=y['rows'][:1920];y['rows'][0]['output']='b'*64
        with self.assertRaises(AssertionError):same_previous(x,y)
    def test_previous_statistics_cannot_change(self):
        x=sample();y=copy.deepcopy(x);y['rows']=y['rows'][:1920];y['rows'][0]['candidate']['DifferentBits']=1
        with self.assertRaises(AssertionError):same_previous(x,y)
    def test_wide_census_deletion(self):
        x=sample();x['rows'].pop(1152)
        with self.assertRaises(AssertionError):result(x,'raw',512,{})
    def test_wide_geometry_duplicate(self):
        x=sample();x['rows'][1152]=copy.deepcopy(x['rows'][0]);x['rows'][1152]['index']=1152
        with self.assertRaises(AssertionError):result(x,'raw',512,{})
    def test_old_case_preservation_is_not_relaxed(self):
        x=sample();y=copy.deepcopy(x);y['rows']=y['rows'][:1920]
        x['rows'][1151]['output']='b'*64
        with self.assertRaises(AssertionError):same_previous(x,y)
    def test_missing_range_case(self):
        x=sample();x['rangeChecks']-=1
        with self.assertRaises(AssertionError):result(x,'raw',512,{})
    def test_complete_census(self):self.assertEqual(result(sample(),'raw',512,{})['cases'],3456)
    def test_expanded_geometry_duplicate(self):
        x=sample();x['rows'][1944]=copy.deepcopy(x['rows'][1920]);x['rows'][1944]['index']=1944
        with self.assertRaises(AssertionError):result(x,'raw',512,{})
    def test_missing_single_block_refusal(self):
        x=sample();x['singleBlockRefusals']=1
        with self.assertRaises(AssertionError):result(x,'raw',512,{})
    def test_fresh_role_identity_may_change(self):
        x=sample();y=copy.deepcopy(x);y.update(pid=19,core_sha256='b'*64)
        self.assertTrue(same_current(x,y))
    def test_fresh_role_expanded_output_cannot_change(self):
        x=sample();y=copy.deepcopy(x);y['rows'][-1]['output']='b'*64
        with self.assertRaises(AssertionError):same_current(x,y)
    def test_product_identity(self):
        x=dict(assembly='a'*64,pid=13,core_sha256='b'*64)
        payload=dict(products={'candidate':{'Lokad.Onnx.dll':dict(sha256='b'*64)}})
        built=dict(consumer=dict(sha256='a'*64));row=dict(child=dict(pid=13))
        identity(x,'candidate',payload,built,row)
        for key,value in [('assembly','c'*64),('pid',14),('core_sha256','d'*64)]:
            y=dict(x);y[key]=value
            with self.assertRaises(AssertionError):identity(y,'candidate',payload,built,row)
    def test_missing_case(self):
        x=sample();x['rows'].pop()
        with self.assertRaises(AssertionError):result(x,'raw',512,{})
    def test_duplicate_case(self):
        x=sample();x['rows'][1]=copy.deepcopy(x['rows'][0])
        with self.assertRaises(AssertionError):result(x,'raw',512,{})
    def test_wrong_width(self):
        with self.assertRaises(AssertionError):result(sample(),'raw',256,{})
    def test_readonly_failure(self):
        x=sample();x['rows'][2]['readonlyInputs']=False
        with self.assertRaises(AssertionError):result(x,'raw',512,{})
    def test_false_error_claim(self):
        x=metric(1);x.update(Actual=.01,Scaled=.01,Absolute=.01)
        with self.assertRaises(AssertionError):error(x,1)
    def test_wrong_worst_error(self):
        x=metric(1);x['Actual']=1
        with self.assertRaises(AssertionError):error(x,1)
    def test_nonfinite_error(self):
        x=metric(1);x['Scaled']=float('nan')
        with self.assertRaises(AssertionError):error(x,1)
    def test_failure_is_retained(self):
        x=sample();x['rows'][0]['candidate'].update(Actual=.01,Scaled=.01,Absolute=.01,Passed=False)
        failures=result(x,'raw',512,{})['failures'];self.assertEqual(len(failures),1)
        self.assertEqual(failures[0]['role'],'candidate')


if __name__=='__main__':unittest.main()
