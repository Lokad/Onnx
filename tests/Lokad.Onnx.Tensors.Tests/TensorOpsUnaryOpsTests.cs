using System;

namespace Lokad.Onnx.Tensors.Tests;

public class TensorOpsUnaryOpsTests
{
    [Fact]
    public void Negate_Abs_Square_Sqrt_Float()
    {
        var a = DenseTensor<float>.OfValues(new float[] { -4f, -1f, 0f, 3f });

        var neg = Tensor<float>.Negate(a);
        var abs = Tensor<float>.Abs(a);
        var sqr = Tensor<float>.Square(a);
        var sqrt = Tensor<float>.Sqrt(DenseTensor<float>.OfValues(new float[] { 4f, 9f, 16f }));

        Assert.Equal(4f, neg[0], 5);
        Assert.Equal(1f, abs[1], 5);
        Assert.Equal(9f, sqr[3], 5);
        Assert.Equal(2f, sqrt[0], 5);
        Assert.Equal(3f, sqrt[1], 5);
        Assert.Equal(4f, sqrt[2], 5);
    }

    [Fact]
    public void Negate_Abs_Square_Sqrt_Double()
    {
        var a = DenseTensor<double>.OfValues(new double[] { -9d, -2d, 0d, 5d });

        var neg = Tensor<double>.Negate(a);
        var abs = Tensor<double>.Abs(a);
        var sqr = Tensor<double>.Square(a);
        var sqrt = Tensor<double>.Sqrt(DenseTensor<double>.OfValues(new double[] { 1d, 4d, 25d }));

        Assert.Equal(9d, neg[0], 10);
        Assert.Equal(2d, abs[1], 10);
        Assert.Equal(25d, sqr[3], 10);
        Assert.Equal(1d, sqrt[0], 10);
        Assert.Equal(2d, sqrt[1], 10);
        Assert.Equal(5d, sqrt[2], 10);
    }

    [Fact]
    public void Pow_Relu_Erf_Float()
    {
        var a = DenseTensor<float>.OfValues(new float[] { -2f, -1f, 0f, 3f });
        var b = DenseTensor<float>.OfValues(new float[] { 2f, 3f, 1f, 2f });

        var pow = Tensor<float>.Pow(a, b);
        var relu = Tensor<float>.Relu(a);
        var erf = Tensor<float>.Erf(DenseTensor<float>.OfValues(new float[] { 0f, 1f }));

        Assert.Equal(4f, pow[0], 5);
        Assert.Equal(-1f, pow[1], 5);
        Assert.Equal(0f, relu[1], 5);
        Assert.Equal(3f, relu[3], 5);
        Assert.Equal(0f, erf[0], 5);
        Assert.Equal(0.8427f, erf[1], 3);
    }

    [Fact]
    public void Pow_Relu_Erf_Double()
    {
        var a = DenseTensor<double>.OfValues(new double[] { -2d, -1d, 0d, 3d });
        var b = DenseTensor<double>.OfValues(new double[] { 2d, 3d, 1d, 2d });

        var pow = Tensor<double>.Pow(a, b);
        var relu = Tensor<double>.Relu(a);
        var erf = Tensor<double>.Erf(DenseTensor<double>.OfValues(new double[] { 0d, 1d }));

        Assert.Equal(4d, pow[0], 10);
        Assert.Equal(-1d, pow[1], 10);
        Assert.Equal(0d, relu[1], 10);
        Assert.Equal(3d, relu[3], 10);
        Assert.InRange(Math.Abs(erf[0]), 0d, 1e-6d);
        Assert.Equal(0.8427d, erf[1], 3);
    }
}
