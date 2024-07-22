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
        var se = t2[1, ..];
        Assert.NotNull(se);
        Assert.Equal(2, se.Rank);
        Assert.Equal(3, se.Dimensions[1]);
        var ac = Tensor<int>.Arange(20, 24).Reshape(2, 2);
        t2[.., 1] = ac;
        Assert.Equal(2, ac.Rank);
        ac = ac.Reshape(4, 1);
        Assert.Throws<IndexOutOfRangeException>(() => t2[.., 2] = ac);

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

    [Fact]
    public void CanGather()
    {
        var data = DenseTensor<double>.OfValues(new double[3, 2]
        {{ 1.0, 1.2 }, { 2.3, 3.4 }, { 4.5, 5.7 }});
        var indices = DenseTensor<int>.OfValues(new int[2, 2]
        {
            { 0, 1 },
            { 1, 2},
        });
        var output = Tensor<double>.Gather(data, indices);
        Assert.NotNull(output);
        var data2 = DenseTensor<double>.OfValues(new double[3, 3] {
        { 1.0, 1.2, 1.9 },
        { 2.3, 3.4, 3.9 },
        { 4.5, 5.7, 5.9 },
            });
        var indices2 = DenseTensor<int>.OfValues(new int[1,2] { { 0, 2 } });
        output = Tensor<double>.Gather(data2, indices2, 1);
        Assert.NotNull(output);
    }

    [Fact]
    public void CanSliceOp()
    {
        Tensor<int> data = new int[2, 4] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } }.ToTensor<int>();
        var axes = new int[2] { 0, 1 }.ToTensor<int>();
        var start = new int[2] { 1, 0 }.ToTensor<int>();
        var ends = new int[2] { 2, 3 }.ToTensor<int>();
        var steps = new int[2] { 1, 2 }.ToTensor<int>();
        var output = Tensor<int>.Slice(data, start, ends, axes, steps);
        Assert.NotNull(output);

        data = new int[2, 4] {{ 1, 2, 3, 4 },{ 5, 6, 7, 8 }}.ToTensor<int>();
        start = new int[2] { 0, 1 }.ToTensor<int>();
        ends = new int[2] { -1, 1000 }.ToTensor<int>();

        output = Tensor<int>.Slice(data, start, ends);
        Assert.NotNull(output);

        axes = new int[2] { 0, 2 }.ToTensor<int>();
        start = new int[2] { 0, -7 }.ToTensor<int>();
        ends = new int[2] {-8, 20 }.ToTensor<int>();

        output = Tensor<int>.Slice(Tensor<int>.Ones(10, 10, 10), start, ends, axes);
        Assert.NotNull(output);

        axes = new int[1] {1}.ToTensor<int>();
        start = new int[1] {0}.ToTensor<int>();
        ends = new int[1] {-1}.ToTensor<int>();

        output = Tensor<int>.Slice(Tensor<int>.Ones(20, 10, 5), start, ends, axes);
        Assert.NotNull(output);
    }
}
