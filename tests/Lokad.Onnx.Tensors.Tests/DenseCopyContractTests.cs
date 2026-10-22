using System.Buffers;

namespace Lokad.Onnx.Tensors.Tests;

public class DenseCopyContractTests
{
    sealed class CopyTensor : DenseTensor<int>
    {
        public CopyTensor(int[] values, int[] shape, bool reversed = false) : base(values, shape, reversed) { }
        public CopyTensor(Memory<int> values, int[] shape) : base(values, shape) { }
        public void Assign(Tensor<int> source) => base.CopyFrom(source);
        public int ReadCalls { get; private set; }
        public override int GetValue(int index)
        {
            ReadCalls++;
            return base.GetValue(index);
        }
    }

    sealed class InspectionFailure(int[] values) : MemoryManager<int>
    {
        public Exception? Failure { get; set; }
        public bool AttemptedFailure { get; private set; }
        public override Span<int> GetSpan() => values;
        protected override bool TryGetArray(out ArraySegment<int> segment)
        {
            if (Failure is { } failure)
            {
                AttemptedFailure = true;
                throw failure;
            }
            segment = values;
            return true;
        }
        public override MemoryHandle Pin(int elementIndex = 0) => throw new NotSupportedException();
        public override void Unpin() { }
        protected override void Dispose(bool disposing) { }
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void FatalStorageInspectionFailuresPropagateBeforeCopy(bool stackExhaustion, bool sliceAssignment)
    {
        // Controlled exceptions exercise the policy without exhausting resources.
        Exception failure = stackExhaustion ? new StackOverflowException("controlled") : new OutOfMemoryException("controlled");
        using var memory = new InspectionFailure(new[] { 11, 22 });
        var source = new CopyTensor(memory.Memory, new[] { 2 });
        var destination = new CopyTensor(new[] { -1, -2 }, new[] { 2 });
        memory.Failure = failure;

        Assert.Same(failure, Record.Exception(() =>
        {
            if (sliceAssignment) destination[SliceIndex.All] = source;
            else destination.Assign(source);
        }));
        Assert.True(memory.AttemptedFailure);
        Assert.Equal(0, source.ReadCalls);
        Assert.Equal(new[] { -1, -2 }, destination.ToArray());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NonfatalStorageInspectionFailureSnapshotsOverlappingSource(bool sliceAssignment)
    {
        int[] values = { 1, 2, 3, 4, 5 };
        using var memory = new InspectionFailure(values);
        var source = new CopyTensor(memory.Memory[..4], new[] { 4 });
        var destination = new CopyTensor(values.AsMemory(1, 4), new[] { 4 });
        memory.Failure = new NotSupportedException("controlled unknown backing store");

        if (sliceAssignment) destination[SliceIndex.All] = source;
        else destination.Assign(source);

        Assert.True(memory.AttemptedFailure);
        Assert.Equal(4, source.ReadCalls);
        Assert.Equal(new[] { 1, 1, 2, 3, 4 }, values);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ProtectedCopy_UsesLogicalCoordinates(bool destinationReversed)
    {
        var destination = new CopyTensor(new int[4], new[] { 2, 2 }, destinationReversed);
        foreach (bool sourceReversed in new[] { false, true })
        {
            var source = new DenseTensor<int>(sourceReversed ? new[] { 1, 3, 2, 4 } : new[] { 1, 2, 3, 4 }, new[] { 2, 2 }, sourceReversed);
            destination.Assign(source);
            Assert.Equal(new[] { 1, 2, 3, 4 }, destination.ToArray());
            Assert.Equal(new[] { 1, 2, 3, 4 }, source.ToArray());
        }
    }

    [Fact]
    public void ShapeMismatch_RejectsBeforeChangingDestination()
    {
        var destination = new CopyTensor(new[] { 1, 2, 3, 4 }, new[] { 2, 2 });
        foreach (var source in new[] { DenseTensor<int>.OfValues(new[] { 99 }), new DenseTensor<int>(new[] { 9, 8, 7, 6 }, new[] { 4 }) })
        {
            Assert.Throws<ArgumentException>(() => destination.Assign(source));
            Assert.Equal(new[] { 1, 2, 3, 4 }, destination.ToArray());
        }
    }

    [Fact]
    public void ShiftedOverlap_ObservesOriginalSourceValues()
    {
        int[] values = { 1, 2, 3, 4, 5 };
        var source = new DenseTensor<int>(values.AsMemory(0, 4), new[] { 4 });
        var destination = new CopyTensor(values.AsMemory(1, 4), new[] { 4 });
        destination.Assign(source);
        Assert.Equal(new[] { 1, 2, 3, 4 }, destination.ToArray());
    }
}
