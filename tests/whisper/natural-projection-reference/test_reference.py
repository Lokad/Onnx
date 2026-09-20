"""Independent arithmetic and adversarial record checks, before natural data."""
import copy,math,unittest
import numpy as np
from common import reference,coordinate_checks,coordinates,decompose,LIMITS
from audit import resource_audit

class ReferenceTests(unittest.TestCase):
    def test_orientation_bias_and_cancellation(self):
        x=np.array([[2,3,-4],[1,0,2]],dtype=np.float32).astype(np.float64)
        w=np.array([[1,2],[4,-1],[3,5]],dtype=np.float32).astype(np.float64)
        b=np.array([.5,-.25]);y,bounds=reference(x,w,b)
        np.testing.assert_array_equal(y,[[2.5,-19.25],[7.5,11.75]])
        self.assertTrue(np.all(bounds>0));coordinate_checks(x,w,b,y,coordinates(y.shape))
        x=np.array([[2.**60,1,-2.**60]],dtype=np.float64);w=np.ones((3,1))
        y,bounds=reference(x,w);checks=coordinate_checks(x,w,None,y,[(0,0)])
        self.assertEqual(checks[0]['fsum'],1);self.assertLessEqual(abs(float(y[0,0])-1),bounds[0,0])

    def test_scalar_checks_refuse_corrupted_projection(self):
        x=np.arange(12,dtype=np.float64).reshape(3,4);w=np.arange(20,dtype=np.float64).reshape(4,5)
        y,bounds=reference(x,w);y[1,2]+=1e-5
        with self.assertRaises(AssertionError):coordinate_checks(x,w,None,y,[(1,2)])

    def test_decomposition_separates_input_and_local_effects(self):
        refs={k:np.array([[v,-v]],dtype=np.float64) for k,v in [('MM',4),('MN',1),('NM',4),('NN',1)]}
        actual={k:v.copy() for k,v in refs.items()};actual['MM']+=.125;actual['MN']+=.125
        value=decompose(actual,refs)['MM-NN']
        self.assertEqual(value['propagated']['max_absolute'],3)
        self.assertEqual(value['local_residual_difference']['max_absolute'],.125)
        self.assertEqual(value['closure_max'],0)
        self.assertEqual(value['actual']['failed_values'],2)

    def test_resource_record_refuses_identity_limit_order_and_affinity_changes(self):
        state=dict(complete=True,code=0,seconds=2,started=5,ended=7,preflight=dict(available=LIMITS['preflight_available'],disk=LIMITS['disk']),
            supervisor=dict(pid=1,birth=1),worker=dict(pid=2,birth=2))
        row=dict(seconds=1,pid=2,birth=2,rss=100,available=LIMITS['available'],affinity=[2]);spec=dict(limits=LIMITS)
        self.assertEqual(resource_audit(state,[row],spec)['samples'],1)
        for key,value in [('pid',3),('birth',3),('rss',LIMITS['rss']),('available',0),('affinity',[0]),('seconds',3)]:
            changed=row|{key:value}
            with self.assertRaises(AssertionError):resource_audit(state,[changed],spec)
        for changed in [state|dict(code=1),state|dict(complete=False),state|dict(error='failed'),state|dict(seconds=LIMITS['seconds'])]:
            with self.assertRaises(AssertionError):resource_audit(changed,[row],spec)
        with self.assertRaises(AssertionError):resource_audit(state,[row,row|dict(seconds=.5)],spec)

if __name__=='__main__':unittest.main()
