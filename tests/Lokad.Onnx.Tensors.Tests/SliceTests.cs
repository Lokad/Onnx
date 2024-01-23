namespace Lokad.Onnx.Tensors.Tests;


public class SliceTests
{
    [Fact]
    public void CanSlice()
    {
        Tensor<int> t = Tensor<int>.Arange(0, 12).Reshape(3, 4);
        var s1 = t[.., 2..4];
        Assert.Equal(2, s1.Rank);
        Assert.Equal(6, s1.Length);
        Assert.Equal(3, s1.Dimensions[0]);
        Assert.Equal(2, s1.Dimensions[1]);
        Assert.Equal(11, s1[2, 1]);
        Assert.Equal(t[0, 2], s1[0, 0]);
        var sd = t[1..3, 2..4];
        Assert.NotNull(sd);

        Assert.Equal(11, t[^0, 3]);
        

        var t2 = t.Reshape(2, 2, 3);
        var se = t2[.., 2];
        Assert.NotNull(se);
        Assert.Equal(2, se.Rank);
        Assert.Equal(2, se.Dimensions[1]);
        var ac = Tensor<int>.Arange(20, 24).Reshape(2, 2);
        t2[.., 2] = ac;
        Assert.Equal(2, ac.Rank);
        ac = ac.Reshape(4, 1);
        Assert.Throws<ArgumentException>(() => t2[.., 2] = ac);

        se = t2[..3, 1, ..];
        Assert.Equal(se.Dimensions.ToArray(), new int[] { 2, 3 });
    }

    [Fact]
    public void CanSliceWithRange()
    {
        var x = Tensor<int>.Arange(0, 10);
        Assert.Equal(new[] { 5, 6, 7, 8, 9 }, x[5..]);
        Assert.Equal(new[] { 0, 1, 2 }, x[..3]);
        Assert.Equal(new[] { 3, 4, 5, 6, 7, 8, 9 }, x[3..]);
        Assert.Equal(new[] { 8, 9 }, x[^2..10]);
        x = x.Reshape(2, 5);
        Assert.Equal(8, x[1, 3]);
        Assert.Equal(9, x[1, ^1]);
        x = Tensor<int>.Arange(1, 7).Reshape(2, 3, 1);
        var r = x[1..2];
        Assert.Equal(3, r.Length);
        Assert.Equal(4, r[0, 0, 0]);
        Assert.Equal(5, r[0, 1, 0]);
        Assert.Equal(6, r[0, 2, 0]);
        r = x[.., 0];
        Assert.Equal(6, r.Length);
        Assert.Equal(2, r.Rank);
        Assert.Equal(3, r[0, 2]);
        Assert.Equal(3, r[0, ^0]);
        var p = x.Dimensions[^2];
    }
}
