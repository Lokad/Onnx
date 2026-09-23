import copy
import unittest
from audit import error,result,PAIRS,SHAPES
import itertools


def metric(count):
    return dict(Count=count,Absolute=0.,Scaled=0.,WorstIndex=0,Actual=0.,Reference=0.,DifferentBits=0,Passed=True)


def sample():
    rows=[]
    for index,((c,m),(h,w),pattern,epilogue) in enumerate(itertools.product(PAIRS,SHAPES,['random','impulse','cancellation'],range(8))):
        rows.append(dict(index=index,c=c,m=m,h=h,w=w,pattern=pattern,epilogue=epilogue,
            candidate=metric(m*h*w),current=metric(m*h*w),selectedDifference=metric(m*h*w),
            repeated=True,heldOutput=True,readonlyInputs=True,guards=True,output='a'*64))
    return dict(mode='raw',width=512,runtime='10.0.8',completed=True,noPerformanceMeasurement=True,rows=rows,refusals=21,contracts=24)


class AuditTests(unittest.TestCase):
    def test_complete_census(self):self.assertEqual(result(sample(),'raw',512,{})['cases'],1152)
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
