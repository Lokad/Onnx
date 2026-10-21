namespace Lokad.Onnx.Backend.Tests;

public class TileProviderBoundaryTests
{
    [Theory]
    [InlineData(1431655766)]
    [InlineData(1073741824)]
    public void Int32RepeatOverflow_RejectsUnsupportedDimensions(int repeat)
    {
        var input = DenseTensor<int>.OfShape(repeat == 1073741824 ? 4 : 3);
        Assert.Throws<ArgumentException>(() => CPUExecutionProvider.Tile(input, DenseTensor<int>.OfValues(new[] { repeat }), null));
    }

    [Theory]
    [InlineData(3, 4611686018427387906L)]
    [InlineData(4, 4611686018427387904L)]
    [InlineData(1, 2147483648L)]
    public void Int64RepeatBounds_AreCheckedBeforeMultiplicationOrNarrowing(int dimension, long repeat)
    {
        Assert.Throws<ArgumentException>(() => CPUExecutionProvider.Tile(DenseTensor<int>.OfShape(dimension),
            DenseTensor<long>.OfValues(new[] { repeat }), null));
    }

    [Fact]
    public void Int64HugeRepeatOfEmptyAxis_RemainsEmpty()
    {
        var result = CPUExecutionProvider.Tile(DenseTensor<int>.OfShape(0, 3),
            DenseTensor<long>.OfValues(new[] { long.MaxValue, 1L }), null);
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.Equal(new[] { 0, 3 }, result.Outputs[0].Dims);
        Assert.Empty(((Tensor<int>)result.Outputs[0]).ToArray());
        Assert.Throws<ArgumentException>(() => CPUExecutionProvider.Tile(DenseTensor<int>.OfShape(0),
            DenseTensor<long>.OfValues(new[] { -1L }), null));
        Assert.Throws<ArgumentException>(() => CPUExecutionProvider.Tile(DenseTensor<int>.OfShape(1),
            DenseTensor<long>.OfValues(new[] { 1L, 1L }), null));
    }
}
