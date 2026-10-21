namespace Lokad.Onnx.Tensors.Tests;

public class SharedShapeRulesTests
{
    static DenseTensor<int> Values(int[] shape)
    {
        var result = new DenseTensor<int>(shape);
        for (int i = 0; i < result.Length; i++) result.SetValue(i, i + 1);
        return result;
    }

    [Theory]
    [InlineData(new int[] { }, new int[] { 3 }, new int[] { 3 })]
    [InlineData(new int[] { 3 }, new int[] { }, new int[] { 3 })]
    [InlineData(new int[] { 2, 1 }, new int[] { 1, 3 }, new int[] { 2, 3 })]
    [InlineData(new int[] { 0, 3 }, new int[] { 1, 3 }, new int[] { 0, 3 })]
    [InlineData(new int[] { }, new int[] { }, new int[] { })]
    public void TypedAndUntypedBroadcast_AgreeOnShapeValuesAndOwnership(int[] leftShape, int[] rightShape, int[] expectedShape)
    {
        var left = Values(leftShape);
        var right = Values(rightShape);
        var typed = Tensor<int>.Broadcast(left, right);
        var untyped = ITensor.Broadcast(left, right);
        for (int side = 0; side < 2; side++)
        {
            Assert.Equal(expectedShape, typed[side].Dimensions.ToArray());
            Assert.Equal(expectedShape, untyped[side].Dims);
            Assert.Equal(typed[side].ToArray(), ((Tensor<int>)untyped[side]).ToArray());
            Assert.Equal((side == 0 ? left : right).Storage, typed[side].Storage);
        }
    }

    [Fact]
    public void BroadcastMismatch_PreservesExistingFailureConventions()
    {
        var left = Values(new[] { 2, 3 });
        var right = Values(new[] { 4 });
        Assert.Empty(Tensor<int>.Broadcast(left, right));
        Assert.Empty(ITensor.Broadcast(left, right));
        Assert.False(Tensor<int>.Broadcast(left, right, out var typedLeft, out var typedRight));
        Assert.Null(typedLeft);
        Assert.Null(typedRight);
        Assert.False(ITensor.Broadcast(left, right, out var untypedLeft, out var untypedRight));
        Assert.Same(left, untypedLeft);
        Assert.Same(right, untypedRight);
    }

    static long ScalarBroadcastAllocation(int length)
    {
        var scalar = DenseTensor<int>.Scalar(7);
        var vector = Values(new[] { length });
        _ = Tensor<int>.Broadcast(scalar, vector);
        long before = GC.GetAllocatedBytesForCurrentThread();
        var result = Tensor<int>.Broadcast(scalar, vector);
        long allocation = GC.GetAllocatedBytesForCurrentThread() - before;
        scalar.SetValue(0, 99);
        Assert.Equal(99, result[0][length - 1]);
        return allocation;
    }

    [Fact]
    public void ScalarBroadcast_AllocatesMetadataAndRetainsSharing()
    {
        long small = ScalarBroadcastAllocation(16), large = ScalarBroadcastAllocation(65536);
        Assert.True(large <= small + 2048, $"Scalar broadcast allocation grew from {small} to {large} bytes.");
    }

    [Fact]
    public void TransposeShapeAndValues_UseIdenticalPermutationRules()
    {
        var data = Values(new[] { 2, 3, 4 });
        int[] permutation = { 2, 0, 1 };
        Assert.Equal(new[] { 4, 2, 3 }, Tensor<int>.TransposedShape(data.Dimensions, permutation));
        Assert.Equal(new[] { 4, 2, 3 }, Tensor<int>.Transpose(data, permutation).Dimensions.ToArray());
        Assert.Equal(new[] { 2, 0, 1 }, permutation);
        foreach (int[] invalid in new[] { new[] { 0, 1 }, new[] { 0, 1, 1 }, new[] { -1, 0, 1 }, new[] { 0, 1, 3 } })
        {
            Assert.Throws<ArgumentException>(() => Tensor<int>.TransposedShape(data.Dimensions, invalid));
            Assert.Throws<ArgumentException>(() => Tensor<int>.Transpose(data, invalid));
        }
    }
}
