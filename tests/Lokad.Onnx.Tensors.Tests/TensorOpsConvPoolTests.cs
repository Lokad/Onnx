using System;
using System.Linq;

namespace Lokad.Onnx.Tensors.Tests;

public class TensorOpsConvPoolTests
{
    // Based on https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv
    [Fact]
    public void Conv2D_Float_MatchesReference()
    {
        var x = Tensor<float>.Arange(0.0f, 25.0f).Reshape(1, 1, 5, 5);
        var w = Tensor<float>.Ones(1, 1, 3, 3);
        var y = Tensor<float>.Conv2D(x, w, 1, MathOps.PadType.Value, padvalue: 1, bias: null, kernelshape: null, strides: null, dilations: null);
        var ye = DenseTensor<float>.OfValues(new float[1, 1, 5, 5] { { {
             {12.0f, 21.0f, 27.0f, 33.0f, 24.0f}, {33.0f, 54.0f, 63.0f, 72.0f, 51.0f}, {63.0f, 99.0f, 108.0f, 117.0f, 81.0f},  {93.0f, 144.0f, 153.0f, 162.0f, 111.0f}, {72.0f, 111.0f, 117.0f, 123.0f, 84.0f}
            } } });
        Assert.Equal(ye, y);

        y = Tensor<float>.Conv2D(x, w, 1, MathOps.PadType.Valid, null, null, null, null, null);
        ye = DenseTensor<float>.OfValues(new float[1, 1, 3, 3] { { {
             {54.0f, 63.0f, 72.0f}, {99.0f, 108.0f, 117.0f},  {144.0f, 153.0f, 162.0f}
            } } });
        Assert.Equal(y.Dimensions.ToArray(), w.Dimensions.ToArray());
        Assert.Equal(ye, y);

        x = Tensor<float>.Arange(0.0f, 35.0f).Reshape(1, 1, 7, 5);
        y = Tensor<float>.Conv2D(x, w, 1, MathOps.PadType.Value, strides: new int[] { 2, 2 }, padvalue: 1, bias: null, kernelshape: null, dilations: null);
        ye = DenseTensor<float>.OfValues(new float[1, 1, 4, 3] { { {
            { 12.0f, 27.0f, 24.0f }, { 63.0f, 108.0f, 81.0f },{ 123.0f, 198.0f, 141.0f }, { 112.0f, 177.0f, 124.0f },
        } } });
        Assert.Equal(ye, y);
    }


    [Fact]
    public void Conv2D_Double_SmokeTest()
    {
        var x = DenseTensor<double>.OfValues(new double[1, 1, 3, 3]
        { {
            { { 0.0, 1.0, 2.0 }, { 3.0, 4.0, 5.0 }, { 6.0, 7.0, 8.0 } }
        } });
        var w = Tensor<double>.Ones(1, 1, 2, 2);
        var bias = Tensor<double>.Zeros(1);
        var y = Tensor<double>.Conv2D(x, w, 1, MathOps.PadType.Valid, bias: bias, padvalue: null, kernelshape: null, strides: null, dilations: null);

        Assert.Equal(new[] { 1, 1, 2, 2 }, y.Dimensions.ToArray());
        Assert.Equal(8.0, y[0, 0, 0, 0], 10);
        Assert.Equal(12.0, y[0, 0, 0, 1], 10);
    }

    [Fact]
    public void MaxPool2D_Float_Int_Double()
    {
        var x = DenseTensor<float>.OfValues(new float[1, 1, 4, 4] { { {
            {12.0f, 20.0f, 30.0f, 0.0f  }, { 8.0f, 12.0f, 2.0f, 0.0f }, { 34.0f, 70.0f, 37.0f, 4.0f }, { 112.0f, 100.0f, 25.0f, 12.0f } } } });
        var y = Tensor<float>.MaxPool2D(x, new int[] { 2, 2 }, MathOps.PadType.Value, padvalue: 0, strides: new int[2] { 2, 2 }, dilations: null, ceilMode: false);
        Assert.Equal(DenseTensor<float>.OfValues(new float[2, 2] { { 20.0f, 30.0f }, { 112.0f, 37.0f } }), y);
        y = Tensor<float>.MaxPool2D(x, new int[] { 2, 2 }, MathOps.PadType.Value, padvalue: 0, strides: new int[2] { 1, 1 }, dilations: null, ceilMode: false);
        Assert.Equal(DenseTensor<float>.OfValues(new float[3, 3] { { 20.0f, 30.0f, 30.0f }, { 70.0f, 70.0f, 37.0f }, { 112.0f, 100.0f, 37.0f } }), y);

        var n = DenseTensor<int>.OfValues(new int[1, 1, 4, 4]
        { {
            { { 1, 1, 2, 4 }, { 5, 6, 7, 8 }, { 3, 2, 1, 0 }, { 1, 2, 3, 4 } }
        } });
        var y2 = Tensor<int>.MaxPool2D(n, new int[] { 2, 2 }, MathOps.PadType.Valid, null, null, null);
        Assert.Equal(DenseTensor<int>.OfValues(new int[3, 3] { { 6, 7, 8 }, { 6, 7, 8 }, { 3, 3, 4 } }), y2);

        var d = DenseTensor<double>.OfValues(new double[1, 1, 3, 3] { { {
            { 1.0, 5.0, 2.0 }, { 4.0, 3.0, 6.0 }, { 7.0, 0.0, 8.0 }
        } } });
        var yd = Tensor<double>.MaxPool2D(d, new int[] { 2, 2 }, MathOps.PadType.Valid, null, null, null, false);
        Assert.Equal(new[] { 1, 1, 2, 2 }, yd.Dimensions.ToArray());
        Assert.Equal(new double[] { 5.0, 6.0, 7.0, 8.0 }, yd.ToArray());
    }

    static float[] NaiveIm2Col(float[] src, int C, int H, int W, int kH, int kW, int dH, int dW, int sH, int sW, int pT, int pL, int pB, int pR)
    {
        int effH = (kH - 1) * dH + 1, effW = (kW - 1) * dW + 1;
        int outH = (H + pT + pB - effH) / sH + 1, outW = (W + pL + pR - effW) / sW + 1;
        var patch = new float[C * kH * kW * outH * outW];
        int o = 0;
        for (int c = 0; c < C; c++)
            for (int kh = 0; kh < kH; kh++)
                for (int kw = 0; kw < kW; kw++)
                    for (int oh = 0; oh < outH; oh++)
                        for (int ow = 0; ow < outW; ow++)
                        {
                            int ih = kh * dH - pT + oh * sH, iw = kw * dW - pL + ow * sW;
                            patch[o++] = (ih < 0 || ih >= H || iw < 0 || iw >= W) ? 0f : src[(c * H + ih) * W + iw];
                        }
        return patch;
    }

    static double[] NaiveIm2Col(double[] src, int C, int H, int W, int kH, int kW, int dH, int dW, int sH, int sW, int pT, int pL, int pB, int pR)
    {
        int effH = (kH - 1) * dH + 1, effW = (kW - 1) * dW + 1;
        int outH = (H + pT + pB - effH) / sH + 1, outW = (W + pL + pR - effW) / sW + 1;
        var patch = new double[C * kH * kW * outH * outW];
        int o = 0;
        for (int c = 0; c < C; c++)
            for (int kh = 0; kh < kH; kh++)
                for (int kw = 0; kw < kW; kw++)
                    for (int oh = 0; oh < outH; oh++)
                        for (int ow = 0; ow < outW; ow++)
                        {
                            int ih = kh * dH - pT + oh * sH, iw = kw * dW - pL + ow * sW;
                            patch[o++] = (ih < 0 || ih >= H || iw < 0 || iw >= W) ? 0.0 : src[(c * H + ih) * W + iw];
                        }
        return patch;
    }

    static float[] Im2ColValues(float[] src, int C, int H, int W, int kH, int kW, int dH, int dW, int sH, int sW, int pT, int pL, int pB, int pR)
    {
        int effH = (kH - 1) * dH + 1, effW = (kW - 1) * dW + 1;
        int outH = (H + pT + pB - effH) / sH + 1, outW = (W + pL + pR - effW) / sW + 1;
        var patch = new float[C * kH * kW * outH * outW];
        unsafe
        {
            fixed (float* s = src)
            fixed (float* p = patch)
            {
                MathOps.Im2col(s, C, H, W, kH, kW, dH, dW, sH, sW, pT, pL, pB, pR, p);
            }
        }
        return patch;
    }

    static double[] Im2ColValues(double[] src, int C, int H, int W, int kH, int kW, int dH, int dW, int sH, int sW, int pT, int pL, int pB, int pR)
    {
        int effH = (kH - 1) * dH + 1, effW = (kW - 1) * dW + 1;
        int outH = (H + pT + pB - effH) / sH + 1, outW = (W + pL + pR - effW) / sW + 1;
        var patch = new double[C * kH * kW * outH * outW];
        unsafe
        {
            fixed (double* s = src)
            fixed (double* p = patch)
            {
                MathOps.Im2col(s, C, H, W, kH, kW, dH, dW, sH, sW, pT, pL, pB, pR, p);
            }
        }
        return patch;
    }

    static float[] SweepValues(int n)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = ((i * 37 + 11) % 17 - 8) * 0.5f;
        return a;
    }

    static double[] SweepValuesD(int n)
    {
        var a = new double[n];
        for (int i = 0; i < n; i++) a[i] = ((i * 37 + 11) % 17 - 8) * 0.5;
        return a;
    }

    [Fact]
    public void Im2Col_MatchesNaiveLayout()
    {
        var shapes = new[]
        {
            (C: 1, H: 4, W: 4, kH: 2, kW: 2, dH: 1, dW: 1, sH: 1, sW: 1, pT: 0, pL: 0, pB: 0, pR: 0),
            (C: 2, H: 5, W: 4, kH: 3, kW: 2, dH: 1, dW: 1, sH: 2, sW: 1, pT: 1, pL: 0, pB: 1, pR: 0),
            (C: 3, H: 7, W: 7, kH: 3, kW: 3, dH: 2, dW: 2, sH: 1, sW: 1, pT: 1, pL: 1, pB: 1, pR: 1),
            (C: 1, H: 3, W: 5, kH: 1, kW: 1, dH: 1, dW: 1, sH: 2, sW: 3, pT: 0, pL: 0, pB: 0, pR: 0),
            (C: 2, H: 4, W: 6, kH: 2, kW: 3, dH: 1, dW: 1, sH: 1, sW: 2, pT: 0, pL: 1, pB: 0, pR: 2),
            (C: 1, H: 5, W: 5, kH: 3, kW: 3, dH: 1, dW: 1, sH: 1, sW: 1, pT: 2, pL: 2, pB: 2, pR: 2),
        };
        foreach (var g in shapes)
        {
            var src = SweepValues(g.C * g.H * g.W);
            Assert.Equal(
                NaiveIm2Col(src, g.C, g.H, g.W, g.kH, g.kW, g.dH, g.dW, g.sH, g.sW, g.pT, g.pL, g.pB, g.pR),
                Im2ColValues(src, g.C, g.H, g.W, g.kH, g.kW, g.dH, g.dW, g.sH, g.sW, g.pT, g.pL, g.pB, g.pR));
        }
        var dsrc = SweepValuesD(2 * 4 * 4);
        Assert.Equal(
            NaiveIm2Col(dsrc, 2, 4, 4, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1),
            Im2ColValues(dsrc, 2, 4, 4, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1));
        var dsrc2 = SweepValuesD(1 * 5 * 6);
        Assert.Equal(
            NaiveIm2Col(dsrc2, 1, 5, 6, 3, 2, 2, 1, 1, 2, 0, 1, 2, 0),
            Im2ColValues(dsrc2, 1, 5, 6, 3, 2, 2, 1, 1, 2, 0, 1, 2, 0));
    }
}
