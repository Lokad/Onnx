namespace Lokad.Onnx.Tensors.Tests;

public class TensorCoordinateTests
{
    [Fact]
    public void SliceLinearAccess_ReducedDims()
    {
        Tensor<int> t = Tensor<int>.Arange(0, 24).Reshape(2, 3, 4);
        var s = t[1, ..];
        Assert.Equal(new int[] { 3, 4 }, s.Dimensions.ToArray());
        Assert.Equal(t[1, 0, 0], s.GetValue(0));
        Assert.Equal(t[1, 2, 3], s.GetValue(11));
        Assert.Equal(t[1, 1, 2], s.GetValue(6));
        s.SetValue(0, 99);
        Assert.Equal(99, t[1, 0, 0]);
        s.SetValue(0, 12);
        Assert.Equal(12, t[1, 0, 0]);
    }

    [Fact]
    public void SliceSpanIndexer_ReducedDims()
    {
        Tensor<int> t = Tensor<int>.Arange(0, 24).Reshape(2, 3, 4);
        var s = t[1, ..];
        Assert.Equal(t[1, 2, 3], s[(System.ReadOnlySpan<int>)new int[] { 2, 3 }]);
        s[(System.ReadOnlySpan<int>)new int[] { 0, 0 }] = -5;
        Assert.Equal(-5, t[1, 0, 0]);
        s[(System.ReadOnlySpan<int>)new int[] { 0, 0 }] = 12;
    }

    [Fact]
    public void SliceSpanIndexer_TooManyIndices_Throws()
    {
        Tensor<int> t = Tensor<int>.Arange(0, 6).Reshape(2, 3);
        var s = t[1, ..];
        Assert.Equal(1, s.Rank);
        Assert.Throws<System.ArgumentOutOfRangeException>(() => s[(System.ReadOnlySpan<int>)new int[] { 0, 0 }]);
    }

    [Fact]
    public void SliceBroadcast_StrideZero_Coords()
    {
        Tensor<int> t = Tensor<int>.Arange(0, 3).Reshape(1, 3);
        var b = t.BroadcastDim(0, 2);
        Assert.Equal(new int[] { 2, 3 }, b.Dimensions.ToArray());
        var s = b[0..2, 1];
        Assert.Equal(1, s.Rank);
        Assert.Equal(1, s.GetValue(0));
        Assert.Equal(1, s.GetValue(1));
    }

    [Fact]
    public void SliceTrailingIndex_ReducedLastDim()
    {
        Tensor<int> t = Tensor<int>.Arange(0, 24).Reshape(2, 3, 4);
        var s = t[.., 2];
        Assert.Equal(new int[] { 2, 3 }, s.Dimensions.ToArray());
        Assert.Equal(t[0, 0, 2], s.GetValue(0));
        Assert.Equal(t[1, 2, 2], s.GetValue(5));
        Assert.Equal(t[1, 0, 2], s[(System.ReadOnlySpan<int>)new int[] { 1, 0 }]);
    }
}
