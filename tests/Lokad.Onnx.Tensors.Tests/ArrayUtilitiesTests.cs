namespace Lokad.Onnx.Tensors.Tests;

public class ArrayUtilitiesTests
{
    [Theory]
    [InlineData(false, 23, 0, 1, 2, 3)]
    [InlineData(true, 23, 0, 1, 2, 3)]
    [InlineData(false, 11, 1, -1, 2, 3)]
    [InlineData(true, 5, 1, 1, 2, -1)]
    public void IndexDecompositionPreservesLayoutAndSkippedAxes(bool reverse, int index, int start, int x, int y, int z)
    {
        int[] strides = reverse ? [1, 2, 6] : [12, 4, 1];
        int[] indices = [-1, -1, -1];
        ArrayUtilities.GetIndices(strides, reverse, index, indices, start);
        Assert.Equal(new[] { x, y, z }, indices);

        int[] guarded = [-10, -1, -1, -1, -20];
        ArrayUtilities.GetIndices(strides, reverse, index, guarded.AsSpan(1, 3), start);
        Assert.Equal(new[] { -10, x, y, z, -20 }, guarded);
    }

    [Fact]
    public void EmptyDestinationsAcceptScalarLayout()
    {
        ArrayUtilities.GetIndices(ReadOnlySpan<int>.Empty, false, 0, Array.Empty<int>(), 0);
        ArrayUtilities.GetIndices(ReadOnlySpan<int>.Empty, false, 0, Span<int>.Empty, 0);
    }

    [Fact]
    public void NullArrayRemainsAnError()
    {
        Assert.Throws<NullReferenceException>(() =>
            ArrayUtilities.GetIndices(ReadOnlySpan<int>.Empty, false, 0, (int[])null!, 0));
    }
}
