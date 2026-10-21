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

    [Fact]
    public void UnsafeCopy_SupportsHalfValues()
    {
        AssertCopied(new[] { (Half)1.5f, (Half)(-2), Half.PositiveInfinity });
    }

    [Fact]
    public void UnsafeCopy_SupportsBFloat16BitPatterns()
    {
        var source = new[] { new BFloat16(0x3FC0), new BFloat16(0x8000), new BFloat16(0x7FC1) };
        var destination = new BFloat16[source.Length];
        ArrayUtilities.UnsafeCopy(source, ref destination);
        Assert.Equal(source.Select(v => v.value), destination.Select(v => v.value));
    }

    readonly record struct CopyPair(long First, double Second);

    [Fact]
    public void UnsafeCopy_SupportsUnmanagedStructures()
    {
        AssertCopied(new[] { new CopyPair(long.MaxValue, 1.5), new CopyPair(-1, double.NegativeInfinity) });
    }

    [Fact]
    public void UnsafeCopy_PreservesPrimitiveEmptyAndSelfCopies()
    {
        AssertCopied(new[] { int.MinValue, 0, int.MaxValue });
        AssertCopied(new[] { true, false, true });
        AssertCopied(Array.Empty<Half>());
        var source = new[] { 1, 2, 3 };
        var destination = source;
        ArrayUtilities.UnsafeCopy(source, ref destination);
        Assert.Same(source, destination);
        Assert.Equal(new[] { 1, 2, 3 }, destination);
    }

    [Fact]
    public void UnsafeCopy_RejectsLengthMismatchBeforeWriting()
    {
        var destination = new[] { 10 };
        Assert.Throws<ArgumentException>(() => ArrayUtilities.UnsafeCopy(new[] { 1, 2 }, ref destination));
        Assert.Equal(new[] { 10 }, destination);
    }

    static void AssertCopied<T>(T[] source) where T : unmanaged
    {
        var destination = new T[source.Length];
        var originalDestination = destination;
        ArrayUtilities.UnsafeCopy(source, ref destination);
        Assert.Same(originalDestination, destination);
        Assert.Equal(source, destination);
    }
}
