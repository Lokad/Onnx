using System;
using System.Linq;
using System.Numerics;

namespace Lokad.Onnx.Tensors.Tests;

public class KernelRowMajorTests
{
    static Tensor<T> Input<T>(int layout) where T : unmanaged, INumber<T>
    {
        var parent = new DenseTensor<T>(new[] { 2, layout == 3 ? 5 : 3 }, layout != 0);
        Tensor<T> input = layout switch
        {
            2 => parent.Slice(SliceIndex.All, SliceIndex.All),
            3 => parent.Slice(SliceIndex.All, new SliceIndex(1, 4)),
            4 => parent.Slice(SliceIndex.All, new SliceIndex(2, null, -1)),
            _ => parent
        };
        int value = 1;
        foreach (var coordinate in input.GetDimensionsIterator()) input[coordinate] = T.CreateChecked(value++);
        Assert.Equal(Enumerable.Range(1, 6).Select(T.CreateChecked), input.ToArray());
        return input;
    }

    static DenseTensor<T> Identity<T>() where T : unmanaged, INumber<T>
        => new(Enumerable.Range(0, 9).Select(i => i % 4 == 0 ? T.One : T.Zero).ToArray(), new[] { 3, 3 });

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void MatMul_UsesLogicalOrderAndCountsOneCopy(int layout)
    {
        var input = Input<float>(layout);
        foreach (var options in new[] { TensorExecutionOptions.Scalar, TensorExecutionOptions.Auto })
        {
            var copies = new CopyAccountant();
            Assert.Equal(input.ToArray(), Tensor<float>.MatMul2D(input, Identity<float>(), options with { CopyReporter = copies }).ToArray());
            Assert.Equal(layout == 0 ? 0 : 24, copies.TotalCopyBytes);
            Assert.Equal(input.ToArray(), Tensor<float>.MatMul(input, Identity<float>(), options).ToArray());
            var doubles = Input<double>(layout);
            Assert.Equal(doubles.ToArray(), Tensor<double>.MatMul2D(doubles, Identity<double>(), options).ToArray());
            var integers = Input<int>(layout);
            Assert.Equal(integers.ToArray(), Tensor<int>.MatMul2D(integers, Identity<int>(), options).ToArray());
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void Reductions_UseLogicalRows(int layout)
    {
        var axes = DenseTensor<int>.OfValues(new[] { 1 });
        var floats = Input<float>(layout);
        Assert.Equal(new float[] { 6, 15 }, Tensor<float>.ReduceSum(floats, axes, false, false).ToArray());
        Assert.Equal(new float[] { 2, 5 }, Tensor<float>.ReduceMean(floats, axes, false, false).ToArray());
        Assert.Equal(new float[] { 3, 6 }, Tensor<float>.ReduceMax(floats, axes, false, false).ToArray());
        var doubles = Input<double>(layout);
        Assert.Equal(new double[] { 6, 15 }, Tensor<double>.ReduceSum(doubles, axes, false, false).ToArray());
        Assert.Equal(new double[] { 2, 5 }, Tensor<double>.ReduceMean(doubles, axes, false, false).ToArray());
        Assert.Equal(new double[] { 3, 6 }, Tensor<double>.ReduceMax(doubles, axes, false, false).ToArray());
        var integers = Input<int>(layout);
        Assert.Equal(new[] { 6, 15 }, Tensor<int>.ReduceSum(integers, axes, false, false).ToArray());
        Assert.Equal(new[] { 2, 5 }, Tensor<int>.ReduceMean(integers, axes, false, false).ToArray());
        Assert.Equal(new[] { 3, 6 }, Tensor<int>.ReduceMax(integers, axes, false, false).ToArray());
        var longs = Input<long>(layout);
        Assert.Equal(new long[] { 6, 15 }, Tensor<long>.ReduceSum(longs, axes, false, false).ToArray());
        Assert.Equal(new long[] { 2, 5 }, Tensor<long>.ReduceMean(longs, axes, false, false).ToArray());
        Assert.Equal(new long[] { 3, 6 }, Tensor<long>.ReduceMax(longs, axes, false, false).ToArray());
        Assert.Equal(new float[] { 5, 7, 9 }, Tensor<float>.ReduceSum(floats, DenseTensor<int>.OfValues(new[] { 0 }), false, false).ToArray());
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void Normalization_UsesLogicalOrder(int layout)
    {
        Assert.Equal(Tensor<float>.LayerNormalization(Input<float>(0), DenseTensor<float>.Ones(3), null, 1, 1e-5f).ToArray(),
            Tensor<float>.LayerNormalization(Input<float>(layout), DenseTensor<float>.Ones(3), null, 1, 1e-5f).ToArray());
        Assert.Equal(Tensor<double>.LayerNormalization(Input<double>(0), DenseTensor<double>.Ones(3), null, 1, 1e-5).ToArray(),
            Tensor<double>.LayerNormalization(Input<double>(layout), DenseTensor<double>.Ones(3), null, 1, 1e-5).ToArray());
        var input = Input<float>(layout);
        Assert.Equal(input.ToArray(), Tensor<float>.RotaryEmbedding(input, DenseTensor<float>.Ones(3), DenseTensor<float>.Zeros(3), 1, 1, 1).ToArray());
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void IdentityResize_UsesLogicalOrder(int layout)
    {
        var shape = new[] { 1, 1, 2, 3 };
        var floats = Input<float>(layout).InsertDim(0).InsertDim(0);
        Assert.Equal(new float[] { 1, 2, 3, 4, 5, 6 }, Tensor<float>.Resize(floats, shape, MathOps.ResizeMode.Nearest, MathOps.ResizeCoordinateTransformation.Asymmetric, MathOps.ResizeNearestMode.Floor, -.75f, null).ToArray());
        var doubles = Input<double>(layout).InsertDim(0).InsertDim(0);
        Assert.Equal(new double[] { 1, 2, 3, 4, 5, 6 }, Tensor<double>.Resize(doubles, shape, MathOps.ResizeMode.Nearest, MathOps.ResizeCoordinateTransformation.Asymmetric, MathOps.ResizeNearestMode.Floor, -.75, null).ToArray());
        var integers = Input<int>(layout).InsertDim(0).InsertDim(0);
        Assert.Equal(new[] { 1, 2, 3, 4, 5, 6 }, Tensor<int>.Resize(integers, shape, MathOps.ResizeMode.Nearest, MathOps.ResizeCoordinateTransformation.Asymmetric, MathOps.ResizeNearestMode.Floor, -.75f, null).ToArray());
    }
}
