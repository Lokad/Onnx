namespace Lokad.Onnx.Tensors.Tests;

public class CoordinateBoundsTests
{
    static Tensor<int> Layout(int layout)
    {
        if (layout == 3) return DenseTensor<int>.OfValues(new[] { 1, 2, 3 }).Reshape(1, 3).BroadcastDim(0, 2);
        var data = new DenseTensor<int>(new[] { 2, layout == 2 ? 4 : 3 }, layout == 1);
        Tensor<int> result = layout == 2 ? data.Slice(SliceIndex.All, new SliceIndex(1, 4)) : data;
        int value = 1;
        foreach (var coordinate in result.GetDimensionsIterator()) result[coordinate] = value++;
        return result;
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void InvalidCoordinates_RejectBeforeReadingOrWriting(int layout)
    {
        var tensor = Layout(layout);
        var original = tensor.ToArray();
        foreach (var coordinate in new[] { new[] { 0, 3 }, new[] { 1, -1 }, new[] { -1, 1 },
            new[] { 2, 0 }, new[] { 0 }, new[] { 0, 0, 0 } })
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => tensor[coordinate]);
            Assert.Throws<ArgumentOutOfRangeException>(() => tensor[coordinate] = 99);
            Assert.Throws<ArgumentOutOfRangeException>(() => tensor.GetStorageIndex(coordinate));
            Assert.Equal(original, tensor.ToArray());
        }
        int last = tensor[1, 2];
        tensor[1, 2] = last + 10;
        Assert.Equal(last + 10, tensor[1, 2]);
    }

    [Fact]
    public void ScalarCoordinates_KeepEmptyAndIteratorConventions()
    {
        foreach (var scalar in new Tensor<int>[] { DenseTensor<int>.Scalar(7),
            DenseTensor<int>.OfValues(new[] { 0, 7 }).Slice(SliceIndex.Index(1)) })
        {
            Assert.Equal(7, scalar[Array.Empty<int>()]);
            Assert.Equal(7, scalar[new[] { 0 }]);
            foreach (var coordinate in new[] { new[] { 1 }, new[] { -1 }, new[] { 0, 0 } })
            {
                Assert.Throws<ArgumentOutOfRangeException>(() => scalar[coordinate]);
                Assert.Throws<ArgumentOutOfRangeException>(() => scalar[coordinate] = 99);
            }
            Assert.Equal(new[] { 7 }, scalar.ToArray());
        }
    }

    [Fact]
    public void EmptyAndReducedViews_EnforceTheirOwnDimensions()
    {
        var empty = new DenseTensor<int>(new[] { 0, 3 });
        Assert.Throws<ArgumentOutOfRangeException>(() => empty[0, 0]);
        var parent = new DenseTensor<int>(new[] { 2, 3 });
        var view = parent.Slice(SliceIndex.Index(1), SliceIndex.All);
        foreach (var coordinate in new[] { new[] { 3 }, new[] { -1 }, new[] { 0, 0 } })
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => view.GetOffset(coordinate));
            Assert.Throws<ArgumentOutOfRangeException>(() => view.GetOffset((ReadOnlySpan<int>)coordinate));
        }
        Assert.Equal(5, view.GetOffset(2));
    }
}
