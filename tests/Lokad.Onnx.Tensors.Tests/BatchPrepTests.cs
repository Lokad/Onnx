using System;
using System.Linq;
using Xunit;

namespace Lokad.Onnx.Tensors.Tests;

public class BatchPrepTests
{
    static DenseTensor<float> RandFloat(System.Random rnd, params int[] dims)
    {
        int n = 1;
        foreach (var d in dims) n *= d;
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)rnd.NextDouble();
        return new DenseTensor<float>(a, dims);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    public void BatchedValues_ParityAcrossModes(int dop)
    {
        var rnd = new System.Random(11);
        foreach (var (xd, yd) in new[] { (new[] { 3, 4, 5 }, new[] { 5, 2 }), (new[] { 4, 2, 3 }, new[] { 3, 2 }) })
        {
            var opts = new TensorExecutionOptions(true, true, dop);
            var a = RandFloat(rnd, xd);
            var b = RandFloat(rnd, yd);
            var seq = Tensor<float>.MatMul(a, b, TensorExecutionOptions.Scalar);
            var par = Tensor<float>.MatMul(a, b, opts);
            Assert.Equal(seq.Dimensions.ToArray(), par.Dimensions.ToArray());
            Assert.Equal(seq.ToArray(), par.ToArray());
        }
        var xi = Tensor<int>.Ones(3, 4, 5);
        var yi = Tensor<int>.Ones(3, 5, 2);
        var ise = Tensor<int>.MatMul(xi, yi, TensorExecutionOptions.Scalar);
        var ipa = Tensor<int>.MatMul(xi, yi, new TensorExecutionOptions(true, true, dop));
        Assert.Equal(ise.ToArray(), ipa.ToArray());
        var xd2 = Tensor<double>.Ones(3, 4, 5);
        var yd2 = Tensor<double>.Ones(3, 5, 2);
        var dse = Tensor<double>.MatMul(xd2, yd2, TensorExecutionOptions.Scalar);
        var dpa = Tensor<double>.MatMul(xd2, yd2, new TensorExecutionOptions(true, true, dop));
        Assert.Equal(dse.ToArray(), dpa.ToArray());
    }

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    public void BatchPlanning_AllocationBound(int dop)
    {
        var rnd = new System.Random(13);
        var x = RandFloat(rnd, 2000, 1, 4);
        var y = RandFloat(rnd, 2000, 4, 1);
        var opts = new TensorExecutionOptions(true, true, dop);
        for (int i = 0; i < 3; i++) Tensor<float>.MatMul(x, y, opts);
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 50; i++) Tensor<float>.MatMul(x, y, opts);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.True(allocated < 2000000L, $"batch planning allocated {allocated} bytes for 50 small-batch products");
    }

    [Fact]
    public void SliceOffsetEntries_Agree()
    {
        var data = new DenseTensor<float>(Enumerable.Range(0, 24).Select(i => (float)i).ToArray(), new[] { 2, 3, 4 });
        var slice = new TensorSlice<float>(data, new SliceIndex[] { new SliceIndex(0, 2), 1 });
        Assert.Equal(new int[] { 2, 4 }, slice.Dimensions.ToArray());
        for (int r = 0; r < 2; r++)
            for (int c = 0; c < 4; c++)
            {
                float expected = 4 * (r * 3 + 1) + c;
                Assert.Equal(expected, slice.GetValue(r * 4 + c));
                Assert.Equal(expected, slice[new int[] { r, c }]);
                Assert.Equal(expected, data.GetValue(slice.GetOffset(r, c)));
            }
    }

    [Fact]
    public void SliceOffset_AllocationBound()
    {
        var data = new DenseTensor<float>(Enumerable.Range(0, 120).Select(i => (float)i).ToArray(), new[] { 2, 3, 4, 5 });
        var slice = new TensorSlice<float>(data, new SliceIndex[] { new SliceIndex(0, 2), 1, new SliceIndex(1, 4) });
        var coords = new int[] { 1, 2, 3 };
        for (int i = 0; i < 100; i++) slice.GetOffset(coords);
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 1000; i++) slice.GetOffset(coords);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.True(allocated < 64000L, $"slice offsets allocated {allocated} bytes for 1000 translations");
    }

    [Fact]
    public void SliceOffset_OverlongCoords_Throw()
    {
        var data = DenseTensor<float>.Zeros(2, 3, 4);
        var slice = new TensorSlice<float>(data, new SliceIndex[] { new SliceIndex(0, 2) });
        Assert.Throws<ArgumentOutOfRangeException>(() => slice.GetOffset(new int[] { 0, 0, 0, 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => slice.GetOffset(0, 0, 0, 0));
    }
}
