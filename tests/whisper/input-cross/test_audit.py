"""Exercise numerical decomposition and refusal of damaged observations."""
import copy,unittest
import numpy as np
from audit import decompose,metrics,raw,validate_record,resources

class AuditTests(unittest.TestCase):
    def test_known_decomposition_and_cancellation(self):
        nn=np.array([1.,-4.,0.]);mm=np.array([3.,-3.,.5]);mn=np.array([2.,-4.,.25]);nm=np.array([2.,-3.,.1])
        result=decompose(mm,nn,mn,nm)
        self.assertEqual(result['closure_max'],[0.,0.])
        self.assertEqual(result['terms']['original_MM-NN']['max_abs'],2.)
        self.assertEqual(result['terms']['original_MM-NN']['max_scaled'],2.)
        self.assertEqual(result['terms']['engine_MN-NN']['failed_values'],2)
        # Large opposing components can cancel; they are not causal percentages.
        result=decompose(np.array([1.]),np.array([1.]),np.array([101.]),np.array([101.]))
        self.assertEqual(result['terms']['original_MM-NN']['l2'],0.)
        self.assertEqual(result['terms']['engine_MM-NM']['l2'],100.)
        self.assertEqual(result['terms']['input_NM-NN']['l2'],100.)

    def test_threshold_denominator_and_invalid_values(self):
        result=metrics(np.array([1e-4,2e-4,2.00001e-4]),np.array([1.,2.,2.]))
        self.assertEqual(result['failed_values'],1)
        for value,denominator in [(np.array([np.nan]),np.ones(1)),(np.ones(1),np.array([.5])),(np.ones(2),np.ones(1))]:
            with self.assertRaises(AssertionError):metrics(value,denominator)

    def test_record_damage(self):
        values=np.zeros((1,1500,1280),dtype=np.float32);digest=raw(values)
        item=dict(name='fixture',managed_hidden=dict(raw_sha256=digest),native_hidden=dict(raw_sha256=digest))
        row=dict(request=0,name='fixture',kind='MM',shape=[1,1500,1280],input_sha256='input',inputs_unchanged=True,
                 held_outputs_unchanged=True,baseline_matches=True,values=values.size,failed_values=0,max_scaled=0.,numerical_passed=True)
        validate_record(row,0,item,'MM',values,values,'input')
        for key,value in [('request',1),('name','wrong'),('kind','MN'),('shape',[1,1500,1279]),('input_sha256','wrong'),
                          ('inputs_unchanged',False),('held_outputs_unchanged',False),('baseline_matches',False),('values',1),
                          ('failed_values',1),('max_scaled',1.),('numerical_passed',False)]:
            damaged=copy.deepcopy(row);damaged[key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):validate_record(damaged,0,item,'MM',values,values,'input')
        altered=values.copy();altered.flat[0]=1
        with self.assertRaises(AssertionError):validate_record(row,0,item,'MM',altered,values,'input')

    def test_resource_damage(self):
        limits=dict(seconds=1800,rss=800,available=100,preflight_available=1000);spec=dict(limits=limits)
        state=dict(complete=True,code=0,terminal_members=True,limits=limits,preflight_available=1200,seconds=1.,started=10.,ended=12.,samples=2,
                   peak_rss=400,members={'123':1.},child=dict(pid=123,birth=1.),supervisor=dict(pid=122,birth=.5))
        samples=[dict(seconds=t,available=1200,members=[dict(pid=123,birth=1.,rss=400,affinity=[2])]) for t in [.1,.6]]
        self.assertEqual(resources(state,samples,spec)['peak_rss'],400)
        for key,value in [('complete',False),('code',2),('terminal_members',False),('preflight_available',999),('seconds',1801),
                          ('samples',1),('peak_rss',399),('members',{'123':2.})]:
            damaged=copy.deepcopy(state);damaged[key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):resources(damaged,samples,spec)
        for key,value in [('available',99),('seconds',2.)]:
            damaged=copy.deepcopy(samples);damaged[0][key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):resources(state,damaged,spec)
        for key,value in [('affinity',[0]),('birth',2.),('rss',900)]:
            damaged=copy.deepcopy(samples);damaged[0]['members'][0][key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):resources(state,damaged,spec)

if __name__=='__main__':unittest.main()
