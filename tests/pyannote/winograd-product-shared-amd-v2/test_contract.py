import unittest
import numpy as np
from checks import compare_arrays
from scope import eligible

class ArithmeticContract(unittest.TestCase):
    def test_changed_arithmetic_must_still_pass_native(self):
        native=np.array([0.,1.],dtype=np.float32);actual=native+np.float32(1e-6)
        maximum,exact=compare_arrays(actual,native,native,True)
        self.assertGreater(maximum,0);self.assertFalse(exact)
        with self.assertRaises(AssertionError):compare_arrays(actual,native,native,False)
    def test_native_limit_never_depends_on_scope(self):
        for affected in [False,True]:
            native=np.array([0.],dtype=np.float32);actual=np.array([.01],dtype=np.float32)
            with self.assertRaises(AssertionError):compare_arrays(actual,native,actual,affected)
    def test_shape_dtype_and_finite(self):
        native=np.zeros(2,dtype=np.float32)
        for actual in [np.zeros(1,dtype=np.float32),np.zeros(2,dtype=np.float64),np.array([np.nan,0],dtype=np.float32),np.array([np.inf,0],dtype=np.float32)]:
            with self.assertRaises(AssertionError):compare_arrays(actual,native,None,True)
    def test_signed_zero_still_exact_for_unaffected(self):
        a=np.array([0.],dtype=np.float32);b=np.array([-0.],dtype=np.float32)
        with self.assertRaises(AssertionError):compare_arrays(a,b,b,False)
    def test_eligible_convolution_parameters(self):
        attrs=dict(pads=[1,1,1,1]);self.assertTrue(eligible([64,64,3,3],attrs,1))
        for change in [dict(group=2),dict(strides=[2,2]),dict(dilations=[2,2]),dict(pads=[0,0,0,0])]:
            self.assertFalse(eligible([64,64,3,3],dict(attrs,**change),1))
        for shape in [[384,3,16,16],[32,16,1,1],[16,16,3,3],[32,3,3,3]]:
            self.assertFalse(eligible(shape,attrs,1))
        self.assertFalse(eligible([64,64,3,3],attrs,11))

if __name__=='__main__':unittest.main()
