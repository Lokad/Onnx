namespace Lokad.Onnx.Tensors.Tests;

public class BroadcastViewContractTests
{
    static Tensor<int> Source(int axis, int layout)
    {
        var shape = new List<int> { 2, 3 };
        shape.Insert(axis, 1);
        var backingShape = shape.ToArray();
        if (layout == 2) backingShape[^1]++;
        Tensor<int> source = new DenseTensor<int>(backingShape, layout == 1);
        if (layout == 2)
        {
            var selectors = Enumerable.Repeat(SliceIndex.All, source.Rank).ToArray();
            selectors[^1] = new SliceIndex(1, backingShape[^1]);
            source = source.Slice(selectors);
        }
        int value = 1;
        foreach (var coordinate in source.GetDimensionsIterator()) source[coordinate] = value++;
        return source;
    }

    [Theory]
    [InlineData(0, 0)]
    [InlineData(0, 1)]
    [InlineData(0, 2)]
    [InlineData(1, 0)]
    [InlineData(1, 1)]
    [InlineData(1, 2)]
    [InlineData(2, 0)]
    [InlineData(2, 1)]
    [InlineData(2, 2)]
    public void RemovingOnlyBroadcastSingleton_PreservesValuesAndSharedWrites(int axis, int layout)
    {
        var source = Source(axis, layout);
        var broadcast = source.BroadcastDim(axis, 1);
        var reduced = broadcast.RemoveDim(axis);
        Assert.Equal(new[] { 2, 3 }, reduced.Dimensions.ToArray());
        Assert.Equal(new[] { 1, 2, 3, 4, 5, 6 }, reduced.ToArray());
        Assert.Equal(source.Storage, reduced.Storage);

        var originalCoordinate = new List<int> { 1, 2 };
        originalCoordinate.Insert(axis, 0);
        reduced[1, 2] = 99;
        Assert.Equal(99, source[originalCoordinate.ToArray()]);
        Assert.Equal(new[] { 1, 2, 3, 4, 5, 99 }, source.ToArray());
        source[originalCoordinate.ToArray()] = 100;
        Assert.Equal(new[] { 1, 2, 3, 4, 5, 100 }, reduced.ToArray());
        Assert.Equal(source.Dimensions.ToArray(), broadcast.Dimensions.ToArray());
        Assert.Equal(source.ToArray(), broadcast.ToArray());
    }

    [Fact]
    public void RemovingLastBroadcastSingleton_ReturnsSharingScalar()
    {
        var source = DenseTensor<int>.OfValues(new[] { 7 });
        var reduced = source.BroadcastDim(0, 1).RemoveDim(0);
        Assert.Equal(0, reduced.Rank);
        Assert.Equal(7, reduced.GetValue(0));
        Assert.Equal(source.Storage, reduced.Storage);
        reduced.SetValue(0, 99);
        Assert.Equal(99, source[0]);
    }

    [Fact]
    public void RemovingBroadcastSingleton_RetainsOtherBroadcastAxesAndWriteTargets()
    {
        var source = DenseTensor<int>.OfValues(new[] { 1, 2, 3, 4, 5, 6 }).Reshape(1, 2, 1, 1, 3);
        var broadcast = source.BroadcastDim(0, 2).BroadcastDim(2, 1).BroadcastDim(3, 4);
        var reduced = broadcast.RemoveDim(2);
        int[] oneBatch = { 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
                           4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6 };
        Assert.Equal(new[] { 2, 2, 4, 3 }, reduced.Dimensions.ToArray());
        Assert.Equal(oneBatch.Concat(oneBatch), reduced.ToArray());
        Assert.Equal(source.Storage, reduced.Storage);
        reduced[1, 1, 3, 2] = 99;
        Assert.Equal(new[] { 1, 2, 3, 4, 5, 99 }, source.ToArray());
        Assert.Equal(99, reduced[0, 1, 0, 2]);
        Assert.Equal(3, reduced[0, 0, 0, 2]);
    }

    [Theory]
    [InlineData(0, 0)]
    [InlineData(0, 1)]
    [InlineData(0, 2)]
    [InlineData(1, 0)]
    [InlineData(1, 1)]
    [InlineData(1, 2)]
    [InlineData(2, 0)]
    [InlineData(2, 1)]
    [InlineData(2, 2)]
    [InlineData(3, 0)]
    [InlineData(3, 1)]
    [InlineData(3, 2)]
    [InlineData(4, 0)]
    [InlineData(4, 1)]
    [InlineData(4, 2)]
    public void EmptyBroadcast_ConvertsClonesAndReshapes(int geometry, int entrypoint)
    {
        Tensor<int> source = geometry switch
        {
            1 => DenseTensor<int>.OfShape(1, 2, 0, 3),
            2 => DenseTensor<int>.OfShape(2, 1),
            3 => new DenseTensor<int>(new[] { 1, 0 }, true),
            4 => DenseTensor<int>.OfShape(1, 1).Slice(SliceIndex.All, new SliceIndex(0, 0)),
            _ => DenseTensor<int>.OfShape(1, 0)
        };
        int axis = geometry == 2 ? 1 : 0;
        int[] shape = source.Dimensions.ToArray();
        shape[axis] = geometry == 2 ? 0 : 2;
        var view = entrypoint switch
        {
            1 => Tensor<int>.Expand(source, shape),
            2 => Tensor<int>.BroadcastTo(source, shape),
            _ => source.BroadcastDim(axis, shape[axis])
        };
        Assert.Equal(0, view.Length);
        Assert.Equal(shape, view.Dimensions.ToArray());
        var converted = view.ToDenseTensor();
        Assert.Equal(shape, converted.Dimensions.ToArray());
        Assert.Empty(converted.Buffer.ToArray());
        var clone = view.Clone();
        Assert.NotSame(view, clone);
        Assert.Equal(shape, clone.Dimensions.ToArray());
        Assert.Empty(clone.ToArray());
        var reshaped = view.Reshape(0);
        Assert.Equal(new[] { 0 }, reshaped.Dimensions.ToArray());
        Assert.Empty(reshaped.ToArray());
    }
}
