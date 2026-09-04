using System;

namespace Lokad.Onnx.Tensors.Tests;

public class TensorParallelMatMulTests
{
    static Tensor<float> Seq(int[] dims, int start = 1)
    {
        var t = new DenseTensor<float>(dims);
        var sp = t.Buffer.Span;
        for (int i = 0; i < sp.Length; i++) sp[i] = start + i;
        return t;
    }

    static void AssertTensorsBitwiseEqual(Tensor<float> expected, Tensor<float> actual)
    {
        Assert.Equal(expected.Dimensions.ToArray(), actual.Dimensions.ToArray());
        Assert.Equal(expected.ToArray(), actual.ToArray());
    }

    [Fact]
    public void BatchedParallelMatchesSequentialBitwise()
    {
        var a = Seq(new[] { 4, 8, 8 });
        var b = Seq(new[] { 4, 8, 8 }, 100);
        var expected = Tensor<float>.MatMul(a, b);
        foreach (int dop in new[] { 1, 2, 4, 16 })
            AssertTensorsBitwiseEqual(expected, Tensor<float>.MatMul(a, b, TensorExecutionOptions.Parallel(dop)));
    }

    [Fact]
    public void UnbatchedIgnoresParallelism()
    {
        var a = Seq(new[] { 8, 8 });
        var b = Seq(new[] { 8, 8 }, 100);
        var expected = Tensor<float>.MatMul(a, b);
        AssertTensorsBitwiseEqual(expected, Tensor<float>.MatMul(a, b, TensorExecutionOptions.Parallel(8)));
    }

    [Fact]
    public void ParallelFactoryRejectsNonPositive()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => TensorExecutionOptions.Parallel(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => TensorExecutionOptions.Parallel(-2));
    }
}
