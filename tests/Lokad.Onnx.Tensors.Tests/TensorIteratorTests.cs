namespace Lokad.Onnx.Tensors.Tests;

public class TensorIteratorTests
{
    [Fact]
    public void CanIterateDims()
    {
        var a = new DenseTensor<int>(new[] { 256, 212, 3, });
        var di = a.GetDimensionsIterator(0..^1);
        di = a.GetDimensionsIterator();
        while (di.Next() != null)
        {
            var i = di.Index;
        }
    }

    static System.Collections.Generic.List<int[]> Collect(System.Collections.Generic.IEnumerable<int[]> it)
    {
        var seen = new System.Collections.Generic.List<int[]>();
        foreach (var coords in it) seen.Add((int[])coords.Clone());
        return seen;
    }

    [Fact]
    public void ZeroExtents_YieldNothing()
    {
        Assert.Empty(Collect(new TensorDimensionsIterator(new int[] { 0, 3 })));
        Assert.Empty(Collect(new TensorDimensionsIterator(new int[] { 2, 0 })));
        var t = DenseTensor<int>.OfShape(0, 3);
        var seen = new System.Collections.Generic.List<int[]>();
        foreach (var coords in t.GetDimensionsIterator()) seen.Add(coords);
        Assert.Empty(seen);
    }

    [Fact]
    public void Scalar_YieldsSingleZero()
    {
        var seen = Collect(new TensorDimensionsIterator(new int[0]));
        Assert.Single(seen);
        Assert.Equal(new int[] { 0 }, seen[0]);
    }

    [Fact]
    public void NegativeExtents_Throw()
    {
        Assert.Throws<System.ArgumentOutOfRangeException>(() => new TensorDimensionsIterator(new int[] { 2, -1 }));
    }

    [Fact]
    public void RepeatedEnumeration_Matches()
    {
        var it = new TensorDimensionsIterator(new int[] { 2, 2 });
        var first = Collect(it);
        var second = Collect(it);
        Assert.Equal(4, first.Count);
        Assert.Equal(4, second.Count);
        for (int i = 0; i < 4; i++) Assert.Equal(first[i], second[i]);
    }

    [Fact]
    public void AdvanceThenEnumerate_Restarts()
    {
        var it = new TensorDimensionsIterator(new int[] { 2, 2 });
        Assert.NotNull(it.Next());
        Assert.NotNull(it.Next());
        var seen = Collect(it);
        Assert.Equal(4, seen.Count);
        Assert.Equal(new int[] { 0, 0 }, seen[0]);
    }

    [Fact]
    public void Reset_Reenumerates()
    {
        var it = new TensorDimensionsIterator(new int[] { 2, 2 });
        int n = 0;
        foreach (var _ in it) n++;
        Assert.Equal(4, n);
        it.Reset();
        var seen = Collect(it);
        Assert.Equal(4, seen.Count);
        Assert.Equal(new int[] { 0, 0 }, seen[0]);
        Assert.Equal(new int[] { 1, 1 }, seen[3]);
    }

    [Fact]
    public void CurrentArray_IsBorrowed()
    {
        var it = new TensorDimensionsIterator(new int[] { 2, 2 });
        var e = it.GetEnumerator();
        Assert.True(e.MoveNext());
        var first = e.Current;
        Assert.True(e.MoveNext());
        Assert.Same(first, e.Current);
    }

    [Fact]
    public void FixedDimensions_Reset_Reenumerates()
    {
        var it = new TensorFixedDimensionsIterator(new int[] { 7 }, new int[] { 2, 2 });
        var first = Collect(it);
        Assert.Equal(4, first.Count);
        Assert.Equal(new int[] { 7, 0, 0 }, first[0]);
        it.Reset();
        var second = Collect(it);
        Assert.Equal(4, second.Count);
        for (int i = 0; i < 4; i++) Assert.Equal(first[i], second[i]);
    }

    [Fact]
    public void FixedDimensions_ZeroExtent_YieldsNothing()
    {
        var it = new TensorFixedDimensionsIterator(new int[] { 7 }, new int[] { 0, 2 });
        Assert.Empty(Collect(it));
    }

    [Fact]
    public void EmptyTensor_Clone_Works()
    {
        Tensor<int> t = Tensor<int>.Arange(0, 6).Reshape(2, 3);
        var s = t[0..0, ..];
        Assert.Equal(new int[] { 0, 3 }, s.Dimensions.ToArray());
        var c = s.Clone();
        Assert.Equal(new int[] { 0, 3 }, c.Dimensions.ToArray());
        Assert.Equal(0, c.Length);
        var d = s.ToDenseTensor();
        Assert.Equal(new int[] { 0, 3 }, d.Dimensions.ToArray());
        Assert.Equal(0, d.Length);
    }
}
