using System;
using System.Linq;

namespace Lokad.Onnx.Backend.Tests;

public class SplitTypedLayoutTests
{
    static void Check<T>(Func<int, T> convert) where T : unmanaged
    {
        var values = Enumerable.Range(1, 6).Select(convert).ToArray();
        foreach (int layout in new[] { 0, 1, 2, 3, 4 })
        foreach (int axis in new[] { 0, 1, -1 })
        foreach (bool pooled in new[] { false, true })
        {
            var parent = new DenseTensor<T>(new[] { 2, layout == 3 ? 5 : 3 }, layout != 0);
            Tensor<T> input = layout switch
            {
                2 => parent.Slice(SliceIndex.All, SliceIndex.All),
                3 => parent.Slice(SliceIndex.All, new SliceIndex(1, 4)),
                4 => parent.Slice(SliceIndex.All, new SliceIndex(2, null, -1)),
                _ => parent
            };
            int position = 0;
            foreach (var coordinate in input.GetDimensionsIterator()) input[coordinate] = values[position++];
            var pool = pooled ? new TensorBufferPool() : null;
            int[] sizes = axis == 0 ? new[] { 1, 0, 1 } : new[] { 1, 0, 2 };
            var result = CPUExecutionProvider.Split(input, null, axis, sizes, null, ExecutionOptions.Memory, 3, pool);
            Assert.Equal(OpStatus.Success, result.Status);
            Assert.Equal(3, result.Outputs.Length);
            Assert.Empty(((Tensor<T>)result.Outputs[1]).ToArray());
            Assert.Equal(axis == 0 ? values[..3] : new[] { values[0], values[3] }, ((Tensor<T>)result.Outputs[0]).ToArray());
            Assert.Equal(axis == 0 ? values[3..] : new[] { values[1], values[2], values[4], values[5] }, ((Tensor<T>)result.Outputs[2]).ToArray());
            if (pool is not null) Assert.Equal(typeof(T) == typeof(float) ? 3 : 0, pool.AllocatedNew);
            result.Outputs[0].SetValue(0, default(T));
            Assert.Equal(values, input.ToArray());
        }
    }

    [Theory]
    [InlineData(TensorElementType.Float)]
    [InlineData(TensorElementType.Double)]
    [InlineData(TensorElementType.Int32)]
    [InlineData(TensorElementType.Int64)]
    [InlineData(TensorElementType.UInt32)]
    [InlineData(TensorElementType.Int8)]
    [InlineData(TensorElementType.UInt8)]
    [InlineData(TensorElementType.Int16)]
    [InlineData(TensorElementType.UInt16)]
    [InlineData(TensorElementType.UInt64)]
    [InlineData(TensorElementType.Bool)]
    [InlineData(TensorElementType.Float16)]
    [InlineData(TensorElementType.BFloat16)]
    public void SharedExecution_PreservesTypesLayoutsOwnershipAndPooling(TensorElementType type)
    {
        switch (type)
        {
            case TensorElementType.Float: Check(i => (float)i); break;
            case TensorElementType.Double: Check(i => (double)i); break;
            case TensorElementType.Int32: Check(i => i); break;
            case TensorElementType.Int64: Check(i => (long)i); break;
            case TensorElementType.UInt32: Check(i => (uint)i); break;
            case TensorElementType.Int8: Check(i => (sbyte)i); break;
            case TensorElementType.UInt8: Check(i => (byte)i); break;
            case TensorElementType.Int16: Check(i => (short)i); break;
            case TensorElementType.UInt16: Check(i => (ushort)i); break;
            case TensorElementType.UInt64: Check(i => (ulong)i); break;
            case TensorElementType.Bool: Check(i => i % 2 != 0); break;
            case TensorElementType.Float16: Check(i => (Half)i); break;
            case TensorElementType.BFloat16: Check(i => (BFloat16)(float)i); break;
        }
    }

    static long SequenceAllocation(Tensor<float> input)
    {
        OpResult Run() => CPUExecutionProvider.SplitToSequence(input, null, 1, 1, null);
        for (int i = 0; i < 4; i++) Run();
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 3; i++) GC.KeepAlive(Run());
        return (GC.GetAllocatedBytesForCurrentThread() - before) / 3;
    }

    [Fact]
    public void SequenceSplit_NormalizesReversedViewOnce()
    {
        var root = new DenseTensor<float>(new[] { 128, 128 }, true);
        var input = root.Slice(SliceIndex.All, SliceIndex.All);
        var dense = DenseTensor<float>.OfShape(128, 128);
        long baseline = SequenceAllocation(dense);
        long view = SequenceAllocation(input);
        Assert.True(view - baseline < 2 * 128 * 128 * sizeof(float), $"Reversed sequence split allocated {view - baseline} extra bytes.");
        var small = new DenseTensor<float>(new float[] { 1, 4, 2, 5, 3, 6 }, new[] { 2, 3 }, true).Slice(SliceIndex.All, SliceIndex.All);
        var result = CPUExecutionProvider.SplitToSequence(small, null, 1, 1, null);
        Assert.Equal(OpStatus.Success, result.Status);
        var sequence = Assert.IsType<TensorSequence>(result.Outputs[0]);
        Assert.Equal(new float[] { 1, 4 }, ((Tensor<float>)sequence.Items[0]).ToArray());
        Assert.Equal(new float[] { 2, 5 }, ((Tensor<float>)sequence.Items[1]).ToArray());
        Assert.Equal(new float[] { 3, 6 }, ((Tensor<float>)sequence.Items[2]).ToArray());
    }
}
