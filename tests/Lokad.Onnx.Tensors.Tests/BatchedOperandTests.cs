using System;
using System.Linq;
using System.Numerics;

namespace Lokad.Onnx.Tensors.Tests;

public class BatchedOperandTests
{
    static Tensor<T> Multiply<T>(Tensor<T> a, Tensor<T> b, TensorExecutionOptions options) where T : unmanaged =>
        typeof(T) == typeof(double) ? (Tensor<T>)(object)Tensor<double>.MatMul((Tensor<double>)(object)a, (Tensor<double>)(object)b, options) :
        typeof(T) == typeof(int) ? (Tensor<T>)(object)Tensor<int>.MatMul((Tensor<int>)(object)a, (Tensor<int>)(object)b, options) :
        (Tensor<T>)(object)Tensor<float>.MatMul((Tensor<float>)(object)a, (Tensor<float>)(object)b, options);

    static Tensor<T> Values<T>(int[] shape, int layout) where T : unmanaged, INumber<T>
    {
        var parentShape = (int[])shape.Clone();
        if (layout == 3) parentShape[^1] += 2;
        var parent = new DenseTensor<T>(parentShape, layout is 1 or 2);
        var slices = Enumerable.Repeat(SliceIndex.All, shape.Length).ToArray();
        if (layout == 3) slices[^1] = new SliceIndex(1, shape[^1] + 1);
        if (layout == 4) slices[^1] = new SliceIndex(shape[^1] - 1, null, -1);
        Tensor<T> result = layout >= 2 ? parent.Slice(slices) : parent;
        int value = 0;
        foreach (var coordinate in result.GetDimensionsIterator()) result[coordinate] = T.CreateChecked(value++ % 7 - 3);
        return result;
    }

    static void Check<T>(int layout) where T : unmanaged, INumber<T>
    {
        foreach (var shapes in new[]
        {
            (new[] { 2, 3, 5, 7 }, new[] { 7, 3 }),
            (new[] { 2, 1, 5, 7 }, new[] { 1, 3, 7, 3 }),
            (new[] { 2, 0, 7 }, new[] { 1, 7, 3 })
        })
        {
            var a = Values<T>(shapes.Item1, layout);
            var b = Values<T>(shapes.Item2, layout);
            int rank = Math.Max(a.Rank, b.Rank);
            var shape = new int[rank];
            for (int d = 0; d < rank - 2; d++)
            {
                int ad = d < rank - a.Rank ? 1 : a.Dimensions[d - (rank - a.Rank)];
                int bd = d < rank - b.Rank ? 1 : b.Dimensions[d - (rank - b.Rank)];
                shape[d] = ad == 1 ? bd : ad;
            }
            shape[^2] = a.Dimensions[^2]; shape[^1] = b.Dimensions[^1];
            var expected = new DenseTensor<T>(shape);
            var ac = new int[a.Rank]; var bc = new int[b.Rank];
            foreach (var coordinate in expected.GetDimensionsIterator())
            {
                for (int d = 0; d < a.Rank - 2; d++) ac[d] = a.Dimensions[d] == 1 ? 0 : coordinate[d + rank - a.Rank];
                for (int d = 0; d < b.Rank - 2; d++) bc[d] = b.Dimensions[d] == 1 ? 0 : coordinate[d + rank - b.Rank];
                ac[^2] = coordinate[^2]; bc[^1] = coordinate[^1];
                T sum = T.Zero;
                for (int n = 0; n < a.Dimensions[^1]; n++)
                {
                    ac[^1] = n; bc[^2] = n;
                    sum = unchecked(sum + a[ac] * b[bc]);
                }
                expected[coordinate] = sum;
            }
            foreach (var options in new[] { TensorExecutionOptions.Scalar, TensorExecutionOptions.Auto })
            {
                var actual = Multiply(a, b, options);
                Assert.Equal(shape, actual.Dimensions.ToArray());
                Assert.Equal(expected.ToArray(), actual.ToArray());
            }
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void BroadcastLayouts_MatchIndependentReference(int layout)
    {
        Check<double>(layout); Check<int>(layout); Check<float>(layout);
    }

    static void CheckCopies<T>(int batches, int itemSize) where T : unmanaged, INumber<T>
    {
        var a = Values<T>(new[] { batches, 1, 64 }, 0);
        var b = Values<T>(new[] { 64, 64 }, 0);
        var copies = new CopyAccountant();
        Multiply(a, b, TensorExecutionOptions.Scalar with { CopyReporter = copies });
        Assert.Equal(0, copies.TotalCopyBytes);
        for (int i = 0; i < 32; i++) Multiply(a, b, TensorExecutionOptions.Scalar);
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 5; i++) GC.KeepAlive(Multiply(a, b, TensorExecutionOptions.Scalar));
        long extra = (GC.GetAllocatedBytesForCurrentThread() - before) / 5 - batches * 64 * itemSize;
        Assert.True(extra < 16384, $"Metadata/scratch allocation was {extra} bytes beyond the output payload.");
    }

    [Theory]
    [InlineData(16)]
    [InlineData(32)]
    public void BroadcastWeights_AreSharedAcrossBatches(int batches)
    {
        CheckCopies<double>(batches, sizeof(double));
        CheckCopies<int>(batches, sizeof(int));
    }

    [Fact]
    public void CroppedOperands_NormalizeBeforeBroadcastExpansion()
    {
        var a = Values<double>(new[] { 8, 2, 7 }, 3);
        var b = Values<double>(new[] { 1, 7, 3 }, 3);
        var copies = new CopyAccountant();
        var actual = Multiply(a, b, TensorExecutionOptions.Scalar with { CopyReporter = copies });
        var expected = Multiply(new DenseTensor<double>(a.ToArray(), a.Dimensions), new DenseTensor<double>(b.ToArray(), b.Dimensions), TensorExecutionOptions.Scalar);
        Assert.Equal(expected.ToArray(), actual.ToArray());
        Assert.Equal((a.Length + b.Length) * sizeof(double), copies.TotalCopyBytes);
    }

    [Fact]
    public void VectorPromotionAndIntegerWrappingRemainUnchanged()
    {
        var a = new DenseTensor<int>(new[] { int.MaxValue, 2, -3, int.MaxValue, 2, -3 }, new[] { 2, 1, 3 });
        var b = new DenseTensor<int>(new[] { 2, 3, 4 }, new[] { 3 });
        Assert.Equal(new[] { -8, -8 }, Multiply(a, b, TensorExecutionOptions.Scalar).ToArray());
        var d = new DenseTensor<double>(new[] { double.NaN, 2.0, 3.0, double.PositiveInfinity, 2.0, 3.0 }, new[] { 2, 1, 3 });
        var e = new DenseTensor<double>(new[] { 1.0, 2.0, 3.0 }, new[] { 3 });
        var result = Multiply(d, e, TensorExecutionOptions.Scalar).ToArray();
        Assert.True(double.IsNaN(result[0])); Assert.Equal(double.PositiveInfinity, result[1]);
    }
}
