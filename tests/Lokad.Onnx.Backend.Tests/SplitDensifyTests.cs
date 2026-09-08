using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

public class SplitDensifyTests
{
    [Fact]
    public void SplitViewInput_Values_And_Allocation()
    {
        var view = new TensorSlice<float>(
            new DenseTensor<float>(Enumerable.Range(0, 48).Select(i => (float)i).ToArray(), new[] { 4, 12 }),
            new SliceIndex[] { new SliceIndex(0, 4), new SliceIndex(2, 10) });
        var sizes = DenseTensor<long>.OfValues(new long[] { 4, 4 });
        var r = CPU.Split(view, sizes, 1, null, null, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(2, r.Outputs!.Length);
        var first = (Tensor<float>)r.Outputs[0];
        Assert.Equal(new[] { 4, 4 }, first.Dimensions.ToArray());
        Assert.Equal(2f, first[0, 0], 5);
        var bigView = new TensorSlice<float>(
            new DenseTensor<float>(Enumerable.Range(0, 96000).Select(i => (float)(i % 13)).ToArray(), new[] { 8, 12000 }),
            new SliceIndex[] { new SliceIndex(0, 8), new SliceIndex(0, 10000) });
        var bigSizes = DenseTensor<long>.OfValues(new long[] { 5000, 5000 });
        for (int i = 0; i < 3; i++) CPU.Split(bigView, bigSizes, 1, null, null, null, null);
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 50; i++) CPU.Split(bigView, bigSizes, 1, null, null, null, null);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.True(allocated < 40000000L, $"view split allocated {allocated} bytes for 50 calls (outputs plus one densify)");
    }

    [Fact]
    public void SplitShapes_NonzeroAxis_EmptyAndUneven()
    {
        var x = DenseTensor<float>.OfValues(Enumerable.Range(0, 24).Select(i => (float)i).ToArray());
        var xx = new DenseTensor<float>(Enumerable.Range(0, 24).Select(i => (float)i).ToArray(), new[] { 2, 3, 4 });
        var uneven = CPU.Split(xx, DenseTensor<long>.OfValues(new long[] { 1, 2, 1 }), 2, null, null, null, null);
        Assert.Equal(OpStatus.Success, uneven.Status);
        Assert.Equal(3, uneven.Outputs!.Length);
        Assert.Equal(new[] { 2, 3, 1 }, ((Tensor<float>)uneven.Outputs[0]).Dimensions.ToArray());
        Assert.Equal(new[] { 2, 3, 2 }, ((Tensor<float>)uneven.Outputs[1]).Dimensions.ToArray());
        Assert.Equal(0f, ((Tensor<float>)uneven.Outputs[0])[0, 0, 0], 5);
        Assert.Equal(1f, ((Tensor<float>)uneven.Outputs[1])[0, 0, 0], 5);
        var withEmpty = CPU.Split(xx, DenseTensor<long>.OfValues(new long[] { 0, 4 }), 2, null, null, null, null);
        Assert.Equal(OpStatus.Success, withEmpty.Status);
        Assert.Equal(0, ((Tensor<float>)withEmpty.Outputs[0]).Dimensions[2]);
        Assert.Equal(new[] { 2, 3, 4 }, ((Tensor<float>)withEmpty.Outputs[1]).Dimensions.ToArray());
        var negAxis = CPU.Split(xx, DenseTensor<long>.OfValues(new long[] { 1, 1 }), 0, null, null, null, null);
        Assert.Equal(OpStatus.Success, negAxis.Status);
        var threeWay = CPU.Split(
            new DenseTensor<float>(Enumerable.Range(0, 8 * 2304).Select(i => (float)(i % 7)).ToArray(), new[] { 8, 2304 }),
            DenseTensor<long>.OfValues(new long[] { 768, 768, 768 }), 1, null, null, null, null);
        Assert.Equal(OpStatus.Success, threeWay.Status);
        Assert.Equal(3, threeWay.Outputs!.Length);
        Assert.Equal(new[] { 8, 768 }, ((Tensor<float>)threeWay.Outputs[2]).Dimensions.ToArray());
    }

    [Fact]
    public void SplitToSequence_ViewInput_Values_And_Allocation()
    {
        var view = new TensorSlice<float>(
            new DenseTensor<float>(Enumerable.Range(0, 24).Select(i => (float)i).ToArray(), new[] { 2, 12 }),
            new SliceIndex[] { new SliceIndex(0, 2), new SliceIndex(0, 12) });
        var r = CPU.SplitToSequence(view, DenseTensor<int>.OfValues(new int[] { 6, 6 }), 1, 1, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var seq = (TensorSequence)r.Outputs![0];
        Assert.Equal(2, seq.Dims[0]);
        var bigView = new TensorSlice<float>(
            new DenseTensor<float>(Enumerable.Range(0, 48000).Select(i => (float)(i % 13)).ToArray(), new[] { 4, 12000 }),
            new SliceIndex[] { new SliceIndex(0, 4), new SliceIndex(0, 10000) });
        var bigSizes = DenseTensor<int>.OfValues(new int[] { 5000, 5000 });
        for (int i = 0; i < 3; i++) CPU.SplitToSequence(bigView, bigSizes, 1, 1, null);
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 50; i++) CPU.SplitToSequence(bigView, bigSizes, 1, 1, null);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.True(allocated < 20000000L, $"sequence split allocated {allocated} bytes for 50 calls (outputs plus one densify)");
    }

    [Fact]
    public void LargeSplitCount_Values()
    {
        var x = new DenseTensor<float>(Enumerable.Range(0, 64).Select(i => (float)i).ToArray(), new[] { 64 });
        var sizes = DenseTensor<long>.OfValues(Enumerable.Repeat(1L, 64).ToArray());
        var r = CPU.Split(x, sizes, 0, null, null, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(64, r.Outputs!.Length);
        Assert.Equal(42f, ((Tensor<float>)r.Outputs[42]).GetValue(0), 5);
    }

}
