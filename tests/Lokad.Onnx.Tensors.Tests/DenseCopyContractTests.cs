namespace Lokad.Onnx.Tensors.Tests;

public class DenseCopyContractTests
{
    sealed class CopyTensor : DenseTensor<int>
    {
        public CopyTensor(int[] values, int[] shape, bool reversed = false) : base(values, shape, reversed) { }
        public CopyTensor(Memory<int> values, int[] shape) : base(values, shape) { }
        public void Assign(Tensor<int> source) => base.CopyFrom(source);
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
