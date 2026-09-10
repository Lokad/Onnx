using System;

namespace Lokad.Onnx.Tensors.Tests;

public class TensorParallelMatMulTests
{
    static Tensor<float> Seq(int[] dims, int start)
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
        var a = Seq(new[] { 4, 8, 8 }, 1);
        var b = Seq(new[] { 4, 8, 8 }, 100);
        var expected = Tensor<float>.MatMul(a, b);
        foreach (int dop in new[] { 1, 2, 4, 16 })
            AssertTensorsBitwiseEqual(expected, Tensor<float>.MatMul(a, b, TensorExecutionOptions.Parallel(dop)));
    }

    [SkippableFact]
    public void UnbatchedIgnoresParallelism()
    {
        Skip.If(!System.Runtime.Intrinsics.X86.Fma.IsSupported, "x86 FMA not available on this machine.");
        var a = Seq(new[] { 8, 8 }, 1);
        var b = Seq(new[] { 8, 8 }, 100);
        var expected = Tensor<float>.MatMul(a, b);
        AssertTensorsBitwiseEqual(expected, Tensor<float>.MatMul(a, b, TensorExecutionOptions.Parallel(8)));
    }

    [SkippableFact]
    public void RowSplitMatchesSequentialBitwise()
    {
        Skip.If(!System.Runtime.Intrinsics.X86.Fma.IsSupported, "x86 FMA not available on this machine.");
        var a = Seq(new[] { 257, 128 }, 1);
        var b = Seq(new[] { 128, 64 }, 100);
        var expected = Tensor<float>.MatMul2D(a, b, TensorExecutionOptions.Intrinsics);
        foreach (int dop in new[] { 1, 2, 4, 8, 16 })
            AssertTensorsBitwiseEqual(expected, Tensor<float>.MatMul2D(a, b, new TensorExecutionOptions(true, true, dop)));
    }

    [Fact]
    public void RowSplitMatchesSequentialInScalarAndSimd()
    {
        var a = Seq(new[] { 130, 64 }, 1);
        var b = Seq(new[] { 64, 32 }, 100);
        foreach (var mode in new[] { TensorExecutionOptions.Scalar, TensorExecutionOptions.Simd })
        {
            var expected = Tensor<float>.MatMul2D(a, b, mode);
            AssertTensorsBitwiseEqual(expected, Tensor<float>.MatMul2D(a, b, new TensorExecutionOptions(mode.UseSimd, mode.UseIntrinsics, 4)));
        }
    }

    [SkippableFact]
    public void BelowRowThresholdIgnoresParallelism()
    {
        Skip.If(!System.Runtime.Intrinsics.X86.Fma.IsSupported, "x86 FMA not available on this machine.");
        var a = Seq(new[] { 30, 32 }, 1);
        var b = Seq(new[] { 32, 16 }, 100);
        var expected = Tensor<float>.MatMul2D(a, b, TensorExecutionOptions.Intrinsics);
        AssertTensorsBitwiseEqual(expected, Tensor<float>.MatMul2D(a, b, new TensorExecutionOptions(true, true, 8)));
    }

    [Fact]
    public void ParallelFactoryRejectsNonPositive()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => TensorExecutionOptions.Parallel(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => TensorExecutionOptions.Parallel(-2));
    }
}
