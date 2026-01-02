using System;

namespace Lokad.Onnx.Tensors.Tests;

public class TensorOpsElementwiseTests
{
    [Fact]
    public void Add_Subtract_Multiply_Divide_IntTensorAndScalar()
    {
        var a = DenseTensor<int>.OfValues(new int[,] { { 8, 6 }, { 4, 2 } });
        var b = DenseTensor<int>.OfValues(new int[,] { { 1, 2 }, { 3, 4 } });

        var add = Tensor<int>.Add(a, b);
        var sub = Tensor<int>.Subtract(a, b);
        var mul = Tensor<int>.Multiply(a, b);
        var div = Tensor<int>.Divide(a, b);

        Assert.Equal(9, add[0, 0]);
        Assert.Equal(6, add[1, 1]);
        Assert.Equal(7, sub[0, 0]);
        Assert.Equal(-2, sub[1, 1]);
        Assert.Equal(8, mul[0, 0]);
        Assert.Equal(8, mul[1, 1]);
        Assert.Equal(8, div[0, 0]);
        Assert.Equal(0, div[1, 1]);

        var addScalar = Tensor<int>.Add(a, 3);
        var subScalar = Tensor<int>.Subtract(a, 3);
        var mulScalar = Tensor<int>.Multiply(a, 3);
        var divScalar = Tensor<int>.Divide(a, 2);

        Assert.Equal(11, addScalar[0, 0]);
        Assert.Equal(-1, subScalar[1, 1]);
        Assert.Equal(12, mulScalar[1, 0]);
        Assert.Equal(3, divScalar[0, 1]);
    }

    [Fact]
    public void Add_Subtract_Multiply_Divide_FloatTensorAndScalar()
    {
        var a = DenseTensor<float>.OfValues(new float[,] { { 1.5f, -2f }, { 3f, 4f } });
        var b = DenseTensor<float>.OfValues(new float[,] { { 0.5f, 1f }, { -1f, 2f } });

        var add = Tensor<float>.Add(a, b);
        var sub = Tensor<float>.Subtract(a, b);
        var mul = Tensor<float>.Multiply(a, b);
        var div = Tensor<float>.Divide(a, b);

        Assert.Equal(2.0f, add[0, 0], 5);
        Assert.Equal(-1.0f, add[0, 1], 5);
        Assert.Equal(1.0f, sub[0, 0], 5);
        Assert.Equal(-3.0f, sub[0, 1], 5);
        Assert.Equal(0.75f, mul[0, 0], 5);
        Assert.Equal(-2.0f, mul[0, 1], 5);
        Assert.Equal(2.0f, div[1, 1], 5);

        var addScalar = Tensor<float>.Add(a, 0.5f);
        var subScalar = Tensor<float>.Subtract(a, 1.0f);
        var mulScalar = Tensor<float>.Multiply(a, 2.0f);
        var divScalar = Tensor<float>.Divide(a, 2.0f);

        Assert.Equal(2.0f, addScalar[0, 0], 5);
        Assert.Equal(3.0f, subScalar[1, 1], 5);
        Assert.Equal(6.0f, mulScalar[1, 0], 5);
        Assert.Equal(-1.0f, divScalar[0, 1], 5);
    }

    [Fact]
    public void Add_Subtract_Multiply_Divide_DoubleTensorAndScalar()
    {
        var a = DenseTensor<double>.OfValues(new double[,] { { 2d, -4d }, { 6d, 8d } });
        var b = DenseTensor<double>.OfValues(new double[,] { { 1d, 2d }, { 3d, 4d } });

        var add = Tensor<double>.Add(a, b);
        var sub = Tensor<double>.Subtract(a, b);
        var mul = Tensor<double>.Multiply(a, b);
        var div = Tensor<double>.Divide(a, b);

        Assert.Equal(3d, add[0, 0], 10);
        Assert.Equal(-2d, add[0, 1], 10);
        Assert.Equal(1d, sub[0, 0], 10);
        Assert.Equal(-6d, sub[0, 1], 10);
        Assert.Equal(2d, mul[0, 0], 10);
        Assert.Equal(-8d, mul[0, 1], 10);
        Assert.Equal(2d, div[1, 1], 10);

        var addScalar = Tensor<double>.Add(a, 1d);
        var subScalar = Tensor<double>.Subtract(a, 2d);
        var mulScalar = Tensor<double>.Multiply(a, 2d);
        var divScalar = Tensor<double>.Divide(a, 4d);

        Assert.Equal(3d, addScalar[0, 0], 10);
        Assert.Equal(6d, subScalar[1, 1], 10);
        Assert.Equal(12d, mulScalar[1, 0], 10);
        Assert.Equal(-1d, divScalar[0, 1], 10);
    }

    [Fact]
    public void Add_Subtract_Multiply_Divide_ByteTensor()
    {
        var a = DenseTensor<byte>.OfValues(new byte[,] { { 4, 8 }, { 12, 16 } });
        var b = DenseTensor<byte>.OfValues(new byte[,] { { 1, 2 }, { 3, 4 } });

        var add = Tensor<byte>.Add(a, b);
        var sub = Tensor<byte>.Subtract(a, b);
        var mul = Tensor<byte>.Multiply(a, b);
        var div = Tensor<byte>.Divide(a, b);

        Assert.Equal((byte)5, add[0, 0]);
        Assert.Equal((byte)9, sub[1, 0]);
        Assert.Equal((byte)16, mul[0, 1]);
        Assert.Equal((byte)4, div[1, 1]);
    }
}
