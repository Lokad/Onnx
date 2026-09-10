using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

public class SplitDensifyTests
{
    [Fact]
    public void OutOfRangeAxis_FailsCleanly()
    {
        // ORT 1.29 refuses axis=5 on rank 2 at load.
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f } });
        var sizes = DenseTensor<long>.OfValues(new long[] { 2L, 2L });
        Assert.Equal(OpStatus.Failure, CPU.Split(x, sizes, 5, null, null, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Split(x, sizes, -3, null, null, null, null).Status);
    }

    [Fact]
    public void SequenceMismatchedSizes_FailsCleanly()
    {
        // ORT 1.29 fails the run (sizes [2,1] sum to 3 on dim 4).
        var x = DenseTensor<float>.OfValues(Enumerable.Range(0, 8).Select(v => (float)v).ToArray(), new[] { 2, 4 });
        var r = CPU.SplitToSequence(x, DenseTensor<long>.OfValues(new long[] { 2L, 1L }), 1, 1, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }

    [Fact]
    public void SequenceScalarChunk_PartialLast()
    {
        // ORT 1.29: chunk 2 over dim 5 -> [(2,),(2,),(1,)].
        var x = DenseTensor<float>.OfValues(Enumerable.Range(0, 5).Select(v => (float)v).ToArray());
        var chunk = new DenseTensor<long>(new long[] { 2L }, Array.Empty<int>());
        var r = CPU.SplitToSequence(x, chunk, 0, 1, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var items = ((TensorSequence)r.Outputs![0]).Items;
        Assert.Equal(3, items.Count);
        Assert.Equal(new float[] { 4f }, ((Tensor<float>)items[2]).ToArray());
    }

    [Fact]
    public void SequenceKeepdimsZero_DropsSingletons()
    {
        // ORT 1.29: chunk 1 over [2,2] axis 0 keepdims=0 -> [(2,),(2,)].
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f } });
        var one = new DenseTensor<long>(new long[] { 1L }, Array.Empty<int>());
        var r = CPU.SplitToSequence(x, one, 0, 0, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var items = ((TensorSequence)r.Outputs![0]).Items;
        Assert.Equal(new int[] { 2 }, ((Tensor<float>)items[0]).Dimensions.ToArray());
        Assert.Equal(new float[] { 3f, 4f }, ((Tensor<float>)items[1]).ToArray());
        // Non-singleton chunks cannot drop the axis: ORT fails the run too.
        var two = new DenseTensor<long>(new long[] { 2L }, Array.Empty<int>());
        Assert.Equal(OpStatus.Failure, CPU.SplitToSequence(x, two, 0, 0, null).Status);
    }

    [Fact]
    public void MismatchedSplitSizes_FailsCleanly()
    {
        // ORT 1.29 fails the run (sizes [2,1] sum to 3 on dim 4).
        var x = DenseTensor<float>.OfValues(Enumerable.Range(0, 8).Select(v => (float)v).ToArray(), new[] { 2, 4 });
        var r = CPU.Split(x, DenseTensor<long>.OfValues(new long[] { 2L, 1L }), 1, null, null, null, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }

    [Fact]
    public void UnevenNumOutputs_DistributesRemainderLast()
    {
        // ORT 1.29: 5/2 -> [3,2], 5/3 -> [2,2,1] (ceil-sized leading).
        var x = DenseTensor<float>.OfValues(Enumerable.Range(0, 5).Select(v => (float)v).ToArray());
        var r2 = CPU.Split(x, null, 0, null, 2, null, null);
        Assert.Equal(OpStatus.Success, r2.Status);
        Assert.Equal(new float[] { 0f, 1f, 2f }, ((Tensor<float>)r2.Outputs![0]).ToArray());
        Assert.Equal(new float[] { 3f, 4f }, ((Tensor<float>)r2.Outputs![1]).ToArray());
        var r3 = CPU.Split(x, null, 0, null, 3, null, null);
        Assert.Equal(OpStatus.Success, r3.Status);
        Assert.Equal(3, r3.Outputs!.Length);
        Assert.Equal(new float[] { 4f }, ((Tensor<float>)r3.Outputs![2]).ToArray());
    }

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

    [Fact]
    public void NumOutputs_SplitsEvenly_MatchingOrt()
    {
        // ORT 1.29: [[1,2],[5,6]] and [[3,4],[7,8]] along axis 1.
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f } });
        var r = CPU.Split(x, null, 1, null, 2, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(2, r.Outputs!.Length);
        Assert.Equal(new float[] { 1f, 2f, 5f, 6f }, ((Tensor<float>)r.Outputs[0]).ToArray());
        Assert.Equal(new float[] { 3f, 4f, 7f, 8f }, ((Tensor<float>)r.Outputs[1]).ToArray());
    }

    [Fact]
    public void NumOutputs_WithSplitInput_Fails()
    {
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f } });
        var sizes = DenseTensor<long>.OfValues(new long[] { 2, 2 });
        var r = CPU.Split(x, sizes, 1, null, 2, null, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("num_outputs", r.Message ?? "");
    }

}
