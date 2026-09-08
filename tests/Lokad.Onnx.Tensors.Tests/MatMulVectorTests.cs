using System;

namespace Lokad.Onnx.Tensors.Tests;

public class MatMulVectorTests
{
    [Fact]
    public void VectorVector_ReturnsScalarDot()
    {
        var x = DenseTensor<float>.OfValues(new float[] { 1f, 2f });
        var y = DenseTensor<float>.OfValues(new float[] { 3f, 4f });
        var c = Tensor<float>.MatMul(x, y);
        Assert.Equal(new int[0], c.Dimensions.ToArray());
        Assert.Equal(11f, c.GetValue(0), 5);
    }

    [Fact]
    public void VectorVector_Int_ReturnsScalarDot()
    {
        var x = DenseTensor<int>.OfValues(new int[] { 1, 2, 3 });
        var y = DenseTensor<int>.OfValues(new int[] { 4, 5, 6 });
        var c = Tensor<int>.MatMul(x, y);
        Assert.Equal(new int[0], c.Dimensions.ToArray());
        Assert.Equal(32, c.GetValue(0));
    }

    [Fact]
    public void MatrixVector_ReturnsVector()
    {
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f } });
        var y = DenseTensor<float>.OfValues(new float[] { 5f, 6f });
        var c = Tensor<float>.MatMul(x, y);
        Assert.Equal(new int[] { 2 }, c.Dimensions.ToArray());
        Assert.Equal(17f, c.GetValue(0), 5);
        Assert.Equal(39f, c.GetValue(1), 5);
    }

    [Fact]
    public void VectorMatrix_ReturnsVector()
    {
        var x = DenseTensor<float>.OfValues(new float[] { 1f, 2f });
        var y = DenseTensor<float>.OfValues(new float[,] { { 3f, 4f, 5f }, { 6f, 7f, 8f } });
        var c = Tensor<float>.MatMul(x, y);
        Assert.Equal(new int[] { 3 }, c.Dimensions.ToArray());
        Assert.Equal(15f, c.GetValue(0), 5);
        Assert.Equal(18f, c.GetValue(1), 5);
        Assert.Equal(21f, c.GetValue(2), 5);
    }

    [Fact]
    public void BatchedMatrixVector_MatchesPerBatch()
    {
        var x = DenseTensor<float>.OfShape(2, 2, 2);
        for (int i = 0; i < x.Length; i++) x.SetValue(i, (float)(i + 1));
        var y = DenseTensor<float>.OfValues(new float[] { 1f, 0f });
        var c = Tensor<float>.MatMul(x, y);
        Assert.Equal(new int[] { 2, 2 }, c.Dimensions.ToArray());
        Assert.Equal(1f, c[0, 0], 5);
        Assert.Equal(3f, c[0, 1], 5);
        Assert.Equal(5f, c[1, 0], 5);
        Assert.Equal(7f, c[1, 1], 5);
    }

    [Fact]
    public void ZeroExtentMatrixVector_ReturnsEmpty()
    {
        var x = DenseTensor<float>.OfShape(0, 2);
        var y = DenseTensor<float>.OfValues(new float[] { 1f, 2f });
        var c = Tensor<float>.MatMul(x, y);
        Assert.Equal(new int[] { 0 }, c.Dimensions.ToArray());
        Assert.Equal(0, c.Length);
    }

    [Fact]
    public void MatrixVector_Double_ReturnsVector()
    {
        var x = DenseTensor<double>.OfValues(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });
        var y = DenseTensor<double>.OfValues(new double[] { 5.0, 6.0 });
        var c = Tensor<double>.MatMul(x, y);
        Assert.Equal(new int[] { 2 }, c.Dimensions.ToArray());
        Assert.Equal(17.0, c.GetValue(0), 9);
        Assert.Equal(39.0, c.GetValue(1), 9);
    }

    [Fact]
    public void DestinationOverload_WritesSqueezedResult()
    {
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f } });
        var y = DenseTensor<float>.OfValues(new float[] { 5f, 6f });
        var dest = DenseTensor<float>.OfShape(2);
        var c = Tensor<float>.MatMul(x, y, dest, TensorExecutionOptions.Scalar);
        Assert.Same(dest, c);
        Assert.Equal(17f, c.GetValue(0), 5);
        Assert.Equal(39f, c.GetValue(1), 5);
    }
}
