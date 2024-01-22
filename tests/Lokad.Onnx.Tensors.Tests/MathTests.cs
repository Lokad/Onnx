namespace Lokad.Onnx.Tensors.Tests;


public class MathTests
{
    [Fact]
    public void CanIterateDims()
    {
        var t = Tensor<int>.Ones(3, 4, 3);
        var id = t.GetDimensionsIterator(0..);
    }

    [Fact]
    public void CanMatMul2D()
    {
        var a = Tensor<int>.Ones(9, 5, 7, 4);
        var b = Tensor<int>.Ones(9, 5, 4, 3);
        var a0 = a[0, 0, ..];
        Assert.Equal(28, a0.Length);
        Assert.Equal(2, a0.Rank);
        var b0 = b[0, 0, ..];
        Assert.Equal(12, b0.Length);
        Assert.Equal(2, b0.Rank);
        var c = Tensor<int>.MatMul2D(a0, b0);
        Assert.Equal(2, c.Rank);
        Assert.Equal(21, c.Length);
    }
}

