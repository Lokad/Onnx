namespace Lokad.Onnx.Tensors.Tests;

public class TileBoundaryTests
{
    [Theory]
    [InlineData(3, 1431655766)]
    [InlineData(4, 1073741824)]
    [InlineData(2, int.MaxValue)]
    public void UnsupportedDimensions_ThrowInsteadOfWrapping(int dimension, int repeat)
    {
        var input = DenseTensor<int>.OfShape(dimension);
        var error = Assert.Throws<ArgumentException>(() => Tensor<int>.Tile(input, new[] { repeat }));
        Assert.Equal("repeats", error.ParamName);
        Assert.Equal(dimension, input.Length);
    }

    [Fact]
    public void ValidViewAndEmptyBoundary_PreserveLogicalValues()
    {
        var input = new DenseTensor<int>(new[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 });
        var view = input.Slice(SliceIndex.All, new SliceIndex(1, 3));
        var result = Tensor<int>.Tile(view, new[] { 1, 2 });
        Assert.Equal(new[] { 2, 4 }, result.Dimensions.ToArray());
        Assert.Equal(new[] { 2, 3, 2, 3, 5, 6, 5, 6 }, result.ToArray());
        Assert.Equal(new[] { 1, 2, 3, 4, 5, 6 }, input.ToArray());
        var empty = Tensor<int>.Tile(DenseTensor<int>.OfShape(0, 1), new[] { 1, int.MaxValue });
        Assert.Equal(new[] { 0, int.MaxValue }, empty.Dimensions.ToArray());
        Assert.Empty(empty.ToArray());
        Assert.Equal(new[] { 5 }, Tensor<int>.Tile(DenseTensor<int>.Scalar(5), Array.Empty<int>()).ToArray());
        Assert.Throws<ArgumentException>(() => Tensor<int>.Tile(view, new[] { 1 }));
        Assert.Throws<ArgumentException>(() => Tensor<int>.Tile(view, new[] { 1, -1 }));
    }
}
