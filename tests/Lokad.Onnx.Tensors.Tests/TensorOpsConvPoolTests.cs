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
        var y = Tensor<float>.Conv2D(x, w, 1, MathOps.PadType.Value, padvalue: 1);
        var ye = DenseTensor<float>.OfValues(new float[1, 1, 5, 5] { { {
             {12.0f, 21.0f, 27.0f, 33.0f, 24.0f}, {33.0f, 54.0f, 63.0f, 72.0f, 51.0f}, {63.0f, 99.0f, 108.0f, 117.0f, 81.0f},  {93.0f, 144.0f, 153.0f, 162.0f, 111.0f}, {72.0f, 111.0f, 117.0f, 123.0f, 84.0f}
            } } });
        Assert.Equal(ye, y);

        y = Tensor<float>.Conv2D(x, w, 1, MathOps.PadType.Valid);
        ye = DenseTensor<float>.OfValues(new float[1, 1, 3, 3] { { {
             {54.0f, 63.0f, 72.0f}, {99.0f, 108.0f, 117.0f},  {144.0f, 153.0f, 162.0f}
            } } });
        Assert.Equal(y.Dimensions.ToArray(), w.Dimensions.ToArray());
        Assert.Equal(ye, y);

        x = Tensor<float>.Arange(0.0f, 35.0f).Reshape(1, 1, 7, 5);
        y = Tensor<float>.Conv2D(x, w, 1, MathOps.PadType.Value, strides: new int[] { 2, 2 }, padvalue: 1);
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
        var y = Tensor<double>.Conv2D(x, w, 1, MathOps.PadType.Valid, bias: bias);

        Assert.Equal(new[] { 1, 1, 2, 2 }, y.Dimensions.ToArray());
        Assert.Equal(8.0, y[0, 0, 0, 0], 10);
        Assert.Equal(12.0, y[0, 0, 0, 1], 10);
    }

    [Fact]
    public void MaxPool2D_Float_Int_Double()
    {
        var x = DenseTensor<float>.OfValues(new float[1, 1, 4, 4] { { {
            {12.0f, 20.0f, 30.0f, 0.0f  }, { 8.0f, 12.0f, 2.0f, 0.0f }, { 34.0f, 70.0f, 37.0f, 4.0f }, { 112.0f, 100.0f, 25.0f, 12.0f } } } });
        var y = Tensor<float>.MaxPool2D(x, new int[] { 2, 2 }, MathOps.PadType.Value, padvalue: 0, strides: new int[2] { 2, 2 });
        Assert.Equal(DenseTensor<float>.OfValues(new float[2, 2] { { 20.0f, 30.0f }, { 112.0f, 37.0f } }), y);
        y = Tensor<float>.MaxPool2D(x, new int[] { 2, 2 }, MathOps.PadType.Value, padvalue: 0, strides: new int[2] { 1, 1 });
        Assert.Equal(DenseTensor<float>.OfValues(new float[3, 3] { { 20.0f, 30.0f, 30.0f }, { 70.0f, 70.0f, 37.0f }, { 112.0f, 100.0f, 37.0f } }), y);

        var n = DenseTensor<int>.OfValues(new int[1, 1, 4, 4]
        { {
            { { 1, 1, 2, 4 }, { 5, 6, 7, 8 }, { 3, 2, 1, 0 }, { 1, 2, 3, 4 } }
        } });
        var y2 = Tensor<int>.MaxPool2D(n, new int[] { 2, 2 });
        Assert.Equal(DenseTensor<int>.OfValues(new int[2, 2] { { 6, 8 }, { 3, 4 } }), y2);

        var d = DenseTensor<double>.OfValues(new double[1, 1, 3, 3] { { {
            { 1.0, 5.0, 2.0 }, { 4.0, 3.0, 6.0 }, { 7.0, 0.0, 8.0 }
        } } });
        var yd = Tensor<double>.MaxPool2D(d, new int[] { 2, 2 });
        Assert.Equal(new[] { 1, 1, 1, 1 }, yd.Dimensions.ToArray());
        Assert.Equal(5.0, yd[0, 0, 0, 0], 10);
    }
}
