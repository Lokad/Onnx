namespace Lokad.Onnx.Tensors.Tests;

public class GatherLayoutTests
{
    static Tensor<int> Layout(int[] values, int[] dimensions, int layout)
    {
        if (layout == 1 && dimensions.Length == 0)
            return DenseTensor<int>.OfValues(new[] { -99, values[0] }).Slice(SliceIndex.Index(1));
        var storageShape = (int[])dimensions.Clone();
        if (layout == 1) storageShape[^1]++;
        var dense = new DenseTensor<int>(storageShape, layout == 2);
        Tensor<int> tensor = layout == 1
            ? dense.Slice(dimensions.Select(d => new SliceIndex(0, d)).ToArray()) : dense;
        int i = 0;
        foreach (var coordinate in tensor.GetDimensionsIterator()) tensor[coordinate] = values[i++];
        Assert.Equal(values, tensor.ToArray());
        return tensor;
    }

    public static IEnumerable<object[]> Cases()
    {
        var shapes = new (int[] data, int[] indices, int[] indexShape, int axis, int[] resultShape, int[] expected)[]
        {
            (new[] { 2, 3 }, new[] { 1 }, Array.Empty<int>(), 0, new[] { 3 }, new[] { 4, 5, 6 }),
            (new[] { 2, 3 }, new[] { -1 }, Array.Empty<int>(), -1, new[] { 2 }, new[] { 3, 6 }),
            (new[] { 4 }, new[] { 2 }, Array.Empty<int>(), 0, Array.Empty<int>(), new[] { 3 }),
            (new[] { 2, 2, 2 }, new[] { 1 }, Array.Empty<int>(), 1, new[] { 2, 2 }, new[] { 3, 4, 7, 8 }),
            (new[] { 2, 2, 2 }, new[] { 0 }, Array.Empty<int>(), 2, new[] { 2, 2 }, new[] { 1, 3, 5, 7 }),
            (new[] { 2, 3 }, new[] { 1, 0 }, new[] { 2 }, 0, new[] { 2, 3 }, new[] { 4, 5, 6, 1, 2, 3 }),
            (new[] { 4 }, new[] { 0, 2, 1, 3 }, new[] { 2, 2 }, 0, new[] { 2, 2 }, new[] { 1, 3, 2, 4 }),
            (new[] { 2, 3 }, Array.Empty<int>(), new[] { 0 }, 1, new[] { 2, 0 }, Array.Empty<int>())
        };
        foreach (var item in shapes)
            for (int dataLayout = 0; dataLayout < 3; dataLayout++)
                for (int indexLayout = 0; indexLayout < 3; indexLayout++)
                    yield return new object[] { item.data, item.indices, item.indexShape, item.axis,
                        item.resultShape, item.expected, dataLayout, indexLayout };
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void Gather_UsesLogicalCoordinatesAcrossLayouts(int[] dimensions, int[] values, int[] indexShape,
        int axis, int[] resultShape, int[] expected, int dataLayout, int indexLayout)
    {
        var inputValues = Enumerable.Range(1, dimensions.Aggregate(1, (a, b) => a * b)).ToArray();
        var data = Layout(inputValues, dimensions, dataLayout);
        var indices = Layout(values, indexShape, indexLayout);
        var result = Tensor<int>.Gather(data, indices, axis);
        Assert.Equal(resultShape, result.Dimensions.ToArray());
        Assert.Equal(expected, result.ToArray());
        Assert.Equal(inputValues, data.ToArray());
        Assert.Equal(values, indices.ToArray());
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void Gather_RejectsInvalidAxesAndIndicesAcrossLayouts(int layout)
    {
        var data = Layout(new[] { 1, 2, 3 }, new[] { 3 }, layout);
        Assert.Throws<ArgumentException>(() => Tensor<int>.Gather(data, DenseTensor<int>.Scalar(0), -2));
        foreach (int index in new[] { 3, -4 })
            Assert.Throws<ArgumentOutOfRangeException>(() => Tensor<int>.Gather(data, DenseTensor<int>.Scalar(index), 0));
    }
}
