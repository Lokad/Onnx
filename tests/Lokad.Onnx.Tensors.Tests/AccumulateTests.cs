namespace Lokad.Onnx.Tensors.Tests;

public class AccumulateTests
{
    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void FoldUsesLogicalOrderAcrossStorageAndSlices(bool reversed, bool sliced)
    {
        var dense = new DenseTensor<int>(sliced ? new[] { 2, 3 } : new[] { 2, 2 }, reversed);
        Tensor<int> source = sliced ? dense.Slice(SliceIndex.All, new SliceIndex(0, 2)).Slice(SliceIndex.All, SliceIndex.All) : dense;
        source[0, 0] = 1; source[0, 1] = 2; source[1, 0] = 3; source[1, 1] = 4;
        Assert.Equal(new[] { 1, 2, 3, 4 }, source.ToArray());
        Assert.Equal(1234, source.Accumulate((state, value) => state * 10 + value, 0));
        Assert.Equal(91234, source.Accumulate((state, value) => state * 10 + value, 9));
        Assert.Equal(new[] { 1, 2, 3, 4 }, source.ToArray());
        if (reversed && !sliced)
            Assert.Equal(new[] { 1, 3, 2, 4 }, Enumerable.Range(0, 4).Select(source.GetValue));
    }

    [Fact]
    public void FoldUsesLogicalOrderForBroadcasts()
    {
        var row = DenseTensor<int>.OfValues(new[] { 1, 2 }, new[] { 1, 2 });
        var broadcast = row.BroadcastDim(0, 2);
        Assert.Equal(new[] { 1, 2, 1, 2 }, broadcast.ToArray());
        Assert.Equal(1212, broadcast.Accumulate((state, value) => state * 10 + value, 0));
    }

    [Fact]
    public void ScalarAndEmptyFoldsPreserveStateAndVisitCounts()
    {
        int calls = 0;
        int Fold(int state, int value) { calls++; return state * 10 + value; }
        Assert.Equal(97, DenseTensor<int>.Scalar(7).Accumulate(Fold, 9));
        Assert.Equal(1, calls);
        calls = 0;
        Assert.Equal(9, DenseTensor<int>.OfShape(2, 0).Accumulate(Fold, 9));
        Assert.Equal(0, calls);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void NullCallbackRejectsBeforeTraversal(int count)
    {
        Assert.Throws<ArgumentNullException>(() => DenseTensor<int>.OfShape(count).Accumulate(null!, 0));
    }
}
