using System;
using System.Linq;

namespace Lokad.Onnx.Tensors.Tests;

public class TensorOpsMatMulTests
{
    [Fact]
    public void MatMul_Int_BroadcastsBatch()
    {
        var a = Tensor<int>.Ones(1, 1, 5, 6);
        var b = Tensor<int>.Ones(3, 6, 7);
        var c = Tensor<int>.MatMul(a, b);
        Assert.Equal(new int[] { 1, 3, 5, 7 }, c.Dimensions.ToArray());

        a = Tensor<int>.Ones(2, 3, 5, 6);
        b = Tensor<int>.Ones(3, 6, 7);
        c = Tensor<int>.MatMul(a, b);
        Assert.Equal(new int[] { 2, 3, 5, 7 }, c.Dimensions.ToArray());

        a = Tensor<int>.Ones(4, 1, 5, 6);
        b = Tensor<int>.Ones(4, 2, 6, 7);
        c = Tensor<int>.MatMul(a, b);
        Assert.Equal(new int[] { 4, 2, 5, 7 }, c.Dimensions.ToArray());

        a = Tensor<int>.Arange(0, 2 * 2 * 4).Reshape(2, 2, 4);
        b = Tensor<int>.Arange(0, 2 * 2 * 4).Reshape(2, 4, 2);
        c = Tensor<int>.MatMul(a, b);
        Assert.Equal(98, c[0, 1, 1]);
    }

    [Fact]
    public void MatMul_Float_BroadcastsBatch()
    {
        var a = Tensor<float>.Ones(1, 1, 5, 6);
        var b = Tensor<float>.Ones(3, 6, 7);
        var c = Tensor<float>.MatMul(a, b);
        Assert.Equal(new int[] { 1, 3, 5, 7 }, c.Dimensions.ToArray());

        a = Tensor<float>.Ones(2, 3, 5, 6);
        b = Tensor<float>.Ones(3, 6, 7);
        c = Tensor<float>.MatMul(a, b);
        Assert.Equal(new int[] { 2, 3, 5, 7 }, c.Dimensions.ToArray());

        a = Tensor<float>.Ones(4, 1, 5, 6);
        b = Tensor<float>.Ones(4, 2, 6, 7);
        c = Tensor<float>.MatMul(a, b);
        Assert.Equal(new int[] { 4, 2, 5, 7 }, c.Dimensions.ToArray());

        a = Tensor<float>.Arange(0.0f, 2.0f * 2 * 4).Reshape(2, 2, 4);
        b = Tensor<float>.Arange(0.0f, 2.0f * 4 * 8).Reshape(2, 4, 8);
        c = Tensor<float>.MatMul(a, b);
        Assert.Equal(326f, c[0, 1, 1], 5);
    }

    [Fact]
    public void MatMul_Int_VectorVectorReturnsScalarDot()
    {
        var x = DenseTensor<int>.OfValues(new[] { 1, 2, 3 });
        var y = DenseTensor<int>.OfValues(new[] { 4, 5, 6 });
        var c = Tensor<int>.MatMul(x, y);
        Assert.Equal(new int[0], c.Dimensions.ToArray());
        Assert.Equal(32, c.GetValue(0));
    }

    [Fact]
    public void MatMul_ThrowsOnRankZero()
    {
        var scalar = new DenseTensor<int>(new int[1], Array.Empty<int>());
        var matrix = Tensor<int>.Ones(2, 2);

        Assert.Throws<ArgumentException>(() => Tensor<int>.MatMul(scalar, matrix));
        Assert.Throws<ArgumentException>(() => Tensor<int>.MatMul(matrix, scalar));
    }
}
