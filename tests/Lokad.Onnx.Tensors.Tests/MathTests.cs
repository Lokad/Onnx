namespace Lokad.Onnx.Tensors.Tests;

using System.Linq;
public class MathTests
{
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

        a = new DenseTensor<int>(new[] { 1, 0, 0, 1 }, new[] { 2, 2, });
        b = new DenseTensor<int>(new[] { 4, 1, 2, 2 }, new[] { 2, 2, });
        c = Tensor<int>.MatMul2D(a, b);
        Assert.Equal(1, c[0, 1]);

        a = Tensor<int>.Arange(0, 2 * 2 * 4).Reshape(2, 2, 4);
        b = Tensor<int>.Arange(0, 2 * 2 * 4).Reshape(2, 4, 2);
        c = new DenseTensor<int>(new int[] { 2, 2, 2 });
        var la = a.Dimensions[^2..];
        var di = a.GetDimensionsIterator(0..^2);
        foreach (var _ in di)
        {
            Assert.Equal(2, a[di[..]].Rank);
            Assert.Equal(2, b[di[..]].Rank);
            c[di[..]] = Tensor<int>.MatMul2D(a[di[..]], b[di[..]]);
        }
        Assert.Equal(98, c[0, 1, 1]);
    }

    [Fact]  
    public void CanMatMul()
    {
        var a = Tensor<int>.Ones(1, 1, 5, 6);
        var b = Tensor<int>.Ones(3, 6, 7);
        var c = Tensor<int>.MatMul(a, b);
        Assert.Equal(new int[] {1,3,5,7}, c.Dimensions.ToArray());
        a = Tensor<int>.Ones(2, 3, 5, 6);
        b = Tensor<int>.Ones(3, 6, 7);
        c = Tensor<int>.MatMul(a, b);
        Assert.Equal(new int[] { 2, 3, 5, 7 }, c.Dimensions.ToArray());
        a = Tensor<int>.Ones(4, 1, 5, 6);
        b = Tensor<int>.Ones(4, 2, 6, 7);
        c = Tensor<int>.MatMul(a, b);
        Assert.Equal(new int[] { 4, 2, 5, 7 }, c.Dimensions.ToArray());
        a = Tensor<int>.Arange(0, 2 * 2 * 4).Reshape(2, 2, 4);
        b = Tensor<int>.Arange(0, 2 * 2 * 4).Reshape(2, 4, 2);
        c = Tensor<int>.MatMul(a, b);
        Assert.Equal(98, c[0, 1, 1]);
    }

}