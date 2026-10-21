namespace Lokad.Onnx.Tensors.Tests;

public class ElementwiseViewAllocationTests
{
    static Tensor<float> View(int side, int layout)
    {
        if (layout == 0) return DenseTensor<float>.Ones(1, side).BroadcastDim(0, side);
        var parent = new DenseTensor<float>(new[] { side, side + 1 }, layout >= 2);
        parent.Fill(1);
        return parent.Slice(SliceIndex.All, new SliceIndex(1, side + 1));
    }

    static long Allocation(Tensor<float> input, Tensor<float> right, ExecutionOptions options)
    {
        for (int i = 0; i < 2; i++) _ = CPUExecutionProvider.Add(input, right, options, null);
        long before = GC.GetAllocatedBytesForCurrentThread();
        var result = CPUExecutionProvider.Add(input, right, options, null);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.All(((Tensor<float>)result.Outputs[0]).ToArray(), value => Assert.Equal(2, value));
        return allocated;
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void BuiltInViews_DoNotAllocateElementSizedInputCopies(int layout)
    {
        foreach (int side in new[] { 16, 128 })
        foreach (var options in new[] { ExecutionOptions.Default, ExecutionOptions.Scalar })
        {
            var right = DenseTensor<float>.Ones(side, side);
            long dense = Allocation(DenseTensor<float>.Ones(side, side), right, options);
            long view = Allocation(View(side, layout), right, options);
            Assert.True(view - dense < 8192, $"View overhead was {view - dense} bytes for {side}x{side}.");
        }
    }

    [Fact]
    public void NegativeStepsAndMemoryWindows_UseComposedRootOffsets()
    {
        var storage = new float[38];
        var parent = new DenseTensor<float>(storage.AsMemory(7, 24), new[] { 4, 6 });
        for (int i = 0; i < 24; i++) parent.SetValue(i, i + 1);
        var input = parent.Slice(new SliceIndex(1, 4), new SliceIndex(5, null, -2));
        var before = (float[])storage.Clone();
        var result = CPUExecutionProvider.Add(input, DenseTensor<float>.Ones(3, 3), null, null);
        Assert.Equal(new float[] { 13, 11, 9, 19, 17, 15, 25, 23, 21 }, ((Tensor<float>)result.Outputs[0]).ToArray());
        Assert.Equal(before, storage);
    }

    [Fact]
    public void CustomMappings_KeepVirtualReadsWithoutDensification()
    {
        var adjusted = new AdjustedDense();
        for (int i = 0; i < 4; i++) adjusted.SetValue(i, i + 1);
        var result = CPUExecutionProvider.Add(adjusted, DenseTensor<float>.Ones(2, 2), null, null);
        Assert.Equal(new float[] { 1002, 1003, 1004, 1005 }, ((Tensor<float>)result.Outputs[0]).ToArray());
        var parent = new DenseTensor<float>(new float[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 });
        var view = new IndexedOnlySlice(parent);
        result = CPUExecutionProvider.Add(view, DenseTensor<float>.Ones(2, 2), null, null);
        Assert.Equal(new float[] { 2, 3, 5, 6 }, ((Tensor<float>)result.Outputs[0]).ToArray());
    }

    sealed class AdjustedDense : DenseTensor<float>
    {
        public AdjustedDense() : base(new[] { 2, 2 }) { }
        public override float GetValue(int index) => base.GetValue(index) + 1000;
    }

    sealed class IndexedOnlySlice : TensorSlice<float>
    {
        public IndexedOnlySlice(Tensor<float> parent) : base(parent, new[] { SliceIndex.All, new SliceIndex(0, 2) }) { }
        public override DenseTensor<float> ToDenseTensor() => throw new InvalidOperationException("Use indexed reads.");
    }
}
