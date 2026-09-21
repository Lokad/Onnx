import unittest
from common import np
from routes import conv_numpy,scalar_conv


class ConvolutionTests(unittest.TestCase):
    def check_geometry(self,channels,out,groups,kernel,stride,pad):
        import torch
        torch.set_num_threads(1)
        rng=np.random.default_rng(731)
        x=rng.standard_normal((2,channels,7,9));w=rng.standard_normal((out,channels//groups,kernel,kernel));b=rng.standard_normal(out)
        a=conv_numpy(x,w,b,stride,pad,groups)
        t=torch.nn.functional.conv2d(torch.tensor(x),torch.tensor(w),torch.tensor(b),stride=stride,padding=pad,groups=groups).numpy()
        reference=np.asarray([scalar_conv(x,w,b,c,stride,pad,groups) for c in np.ndindex(a.shape)]).reshape(a.shape)
        np.testing.assert_allclose(a,reference,rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(t,reference,rtol=1e-13,atol=1e-13)

    def test_stem_entry_borders(self):self.check_geometry(1,4,1,3,2,1)
    def test_depthwise_borders(self):self.check_geometry(3,3,3,3,2,1)
    def test_pointwise(self):self.check_geometry(5,4,1,1,1,0)
    def test_group_channels(self):self.check_geometry(4,6,2,3,1,1)

    def test_projection_layout(self):
        x=np.arange(24).reshape(1,2,3,4)
        expected=np.array([[[0,1,2,3,12,13,14,15],[4,5,6,7,16,17,18,19],[8,9,10,11,20,21,22,23]]])
        np.testing.assert_array_equal(x.transpose(0,2,1,3).reshape(1,3,8),expected)


if __name__=='__main__':unittest.main()
