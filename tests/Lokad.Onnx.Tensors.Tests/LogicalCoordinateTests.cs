namespace Lokad.Onnx.Tensors.Tests;

public class LogicalCoordinateTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CoordinateOrderDependsOnShapeRatherThanStorage(bool reversed)
    {
        var values = Enumerable.Range(0, 24).ToArray();
        var input = new DenseTensor<int>(new Memory<int>(values), new[] { 2, 1, 3, 4 }, reversed);
        AssertCoordinates(input);
        var slice = new TensorSlice<int>(input, new SliceIndex[] { SliceIndex.All, 0, new SliceIndex(1, 3), SliceIndex.All });
        AssertCoordinates(slice);
        var broadcast = DenseTensor<int>.Ones(1, 3).BroadcastDim(0, 2);
        AssertCoordinates(broadcast);
        // Physical access remains physical for reversed dense storage.
        Assert.Equal(Enumerable.Range(0, 24), Enumerable.Range(0, 24).Select(input.GetValue));
    }

    [Fact]
    public void ScalarHasOneEmptyCoordinateAndEmptyShapesHaveNone()
    {
        var scalar = DenseTensor<int>.OfShape();
        Assert.Empty(scalar.GetCoordinates(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => scalar.GetCoordinates(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => scalar.GetCoordinates(1));
        foreach (var empty in new[] { DenseTensor<int>.OfShape(2, 0, 3), DenseTensor<int>.OfShape(0) })
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => empty.GetCoordinates(0));
            Assert.Throws<ArgumentOutOfRangeException>(() => empty.GetCoordinates(-1));
        }
    }

    static void AssertCoordinates(Tensor<int> tensor)
    {
        var dimensions = tensor.Dimensions.ToArray();
        var expected = new List<int[]>();
        Visit(new int[dimensions.Length], 0);
        for (int i = 0; i < expected.Count; i++) Assert.Equal(expected[i], tensor.GetCoordinates(i));
        Assert.Throws<ArgumentOutOfRangeException>(() => tensor.GetCoordinates(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => tensor.GetCoordinates(expected.Count));

        void Visit(int[] coordinates, int axis)
        {
            if (axis == dimensions.Length) { expected.Add((int[])coordinates.Clone()); return; }
            for (int i = 0; i < dimensions[axis]; i++)
            {
                coordinates[axis] = i;
                Visit(coordinates, axis + 1);
            }
        }
    }
}
