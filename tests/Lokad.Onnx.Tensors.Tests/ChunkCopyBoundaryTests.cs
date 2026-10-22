namespace Lokad.Onnx.Tensors.Tests;

public class ChunkCopyBoundaryTests
{
    [Theory]
    [InlineData(int.MaxValue, 1)]
    [InlineData(int.MaxValue, int.MaxValue)]
    [InlineData(1, int.MaxValue)]
    [InlineData(-1, 0)]
    [InlineData(0, -1)]
    [InlineData(1, 0)]
    [InlineData(0, 1)]
    public void ImpossibleChunksRejectEvenWithEmptyPayload(int start, int length)
    {
        var source = DenseTensor<int>.OfShape(0, 0);
        Assert.Throws<ArgumentException>(() => Tensor<int>.ChunkCopy(source, 0, start, length));
    }

    [Fact]
    public void NegativeAxisCopiesExpectedValuesIntoIndependentStorage()
    {
        var source = DenseTensor<int>.OfValues(new[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var copy = Tensor<int>.ChunkCopy(source, -1, 1, 2);
        Assert.Equal(new[] { 2, 2 }, copy.Dimensions.ToArray());
        Assert.Equal(new[] { 2, 3, 5, 6 }, copy.ToArray());
        copy[0, 0] = 99;
        Assert.Equal(2, source[0, 1]);
    }

    [Fact]
    public void EmptyChunkAtAxisEndIsValid()
    {
        var source = DenseTensor<int>.OfShape(2, 3);
        var copy = Tensor<int>.ChunkCopy(source, 1, 3, 0);
        Assert.Equal(new[] { 2, 0 }, copy.Dimensions.ToArray());
        Assert.Empty(copy.ToArray());
        Assert.Empty(Tensor<int>.ChunkCopy(DenseTensor<int>.OfShape(0, 0), 0, 0, 0).ToArray());
    }
}
