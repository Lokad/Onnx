using System;
using System.Linq;
using System.Numerics;

namespace Lokad.Onnx.Tensors.Tests;

public class FlatKernelLayoutTests
{
    static Tensor<T> Input<T>(int layout) where T : unmanaged, INumber<T>
    {
        var root = new DenseTensor<T>(new[] { 2, layout == 3 ? 5 : 3 }, layout != 0);
        Tensor<T> view = layout switch
        {
            2 => root.Slice(SliceIndex.All, SliceIndex.All),
            3 => root.Slice(SliceIndex.All, new SliceIndex(1, 4)),
            4 => root.Slice(SliceIndex.All, new SliceIndex(2, null, -1)),
            _ => root
        };
        int index = 0;
        int[] values = { -3, -1, 0, 1, 2, 4 };
        foreach (var coordinate in view.GetDimensionsIterator()) view[coordinate] = T.CreateChecked(values[index++]);
        return view;
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void Softmax_AxesAndVersionsReadLogicalValues(int layout)
    {
        foreach (int version in new[] { 11, 13 })
        foreach (int axis in new[] { 0, 1, -1 })
        foreach (var options in new[] { TensorExecutionOptions.Scalar, TensorExecutionOptions.Auto })
        {
            Assert.Equal(Tensor<float>.Softmax(Input<float>(0), axis, options, version).ToArray(), Tensor<float>.Softmax(Input<float>(layout), axis, options, version).ToArray());
            Assert.Equal(Tensor<double>.Softmax(Input<double>(0), axis, options, version).ToArray(), Tensor<double>.Softmax(Input<double>(layout), axis, options, version).ToArray());
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void Convolution_IdentityReadsLogicalImage(int layout)
    {
        var floats = Input<float>(layout).InsertDim(0).InsertDim(0);
        var doubles = Input<double>(layout).InsertDim(0).InsertDim(0);
        Assert.Equal(floats.ToArray(), Tensor<float>.Conv2D(floats, DenseTensor<float>.Ones(1, 1, 1, 1), 1, MathOps.PadType.Valid, null, null, null, null, null, TensorExecutionOptions.Scalar).ToArray());
        Assert.Equal(doubles.ToArray(), Tensor<double>.Conv2D(doubles, DenseTensor<double>.Ones(1, 1, 1, 1), 1, MathOps.PadType.Valid, null, null, null, null, null, TensorExecutionOptions.Scalar).ToArray());
    }

    [Fact]
    public void Convolution_ReadsReversedSliceWeight()
    {
        var input = new DenseTensor<float>(Enumerable.Range(1, 9).Select(i => (float)i).ToArray(), new[] { 1, 1, 3, 3 });
        var root = new DenseTensor<float>(new[] { 1, 1, 2, 2 }, true);
        var weight = root.Slice(SliceIndex.All, SliceIndex.All, SliceIndex.All, SliceIndex.All);
        int value = 1;
        foreach (var coordinate in weight.GetDimensionsIterator()) weight[coordinate] = value++;
        Assert.Equal(new float[] { 37, 47, 67, 77 }, Tensor<float>.Conv2D(input, weight, 1, MathOps.PadType.Valid, null, null, null, null, null, TensorExecutionOptions.Scalar).ToArray());
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void GeluTanh_ReadsLogicalValues(int layout)
    {
        Assert.Equal(Tensor<float>.GeluTanh(Input<float>(0), null).ToArray(), Tensor<float>.GeluTanh(Input<float>(layout), null).ToArray());
        Assert.Equal(Tensor<double>.GeluTanh(Input<double>(0), null).ToArray(), Tensor<double>.GeluTanh(Input<double>(layout), null).ToArray());
    }
}
