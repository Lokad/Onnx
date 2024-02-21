namespace Lokad.Onnx.Tensors.Tests;

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
        Assert.Equal(new int[] { 1, 3, 5, 7 }, c.Dimensions.ToArray());
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

    // Based on https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv
    [Fact]
    public void CanConv2D()
    {
        var X = Tensor<float>.Arange(0.0f, 25.0f).Reshape(1, 1, 5, 5);
        var W = Tensor<float>.Ones(1, 1, 3, 3);
        var Y = Tensor<float>.Conv2D(X, W, 1, MathOps.PadType.SameLower, strides: new int[] { 2, 2 });
        Assert.Equal(Y.Dimensions.ToArray(), W.Dimensions.ToArray());
        var Ye = DenseTensor<float>.OfValues(new float[1, 1, 3, 3] { { {
                    { 12.0f, 27.0f, 24.0f }, { 63.0f, 108.0f, 81.0f }, { 72.0f, 117.0f, 84.0f }
                } } });
        Assert.Equal(Ye, Y);


        X = Tensor<float>.Arange(0.0f, 35.0f).Reshape(1, 1, 7, 5);
        Y = Tensor<float>.Conv2D(X, W, 1, MathOps.PadType.Value, strides: new int[] { 2, 2 }, padvalue: 1);
        Ye = DenseTensor<float>.OfValues(new float[1, 1, 4, 3] { { {
                { 12.0f, 27.0f, 24.0f }, { 63.0f, 108.0f, 81.0f },{ 123.0f, 198.0f, 141.0f }, { 112.0f, 177.0f, 124.0f },
            } } });
        Assert.Equal(Ye, Y);
    }

    [Fact]
    public void CanMaxPool2D()
    {
        var X = DenseTensor<float>.OfValues(new float[1, 1, 4, 4] { { {
            {12.0f, 20.0f, 30.0f, 0.0f  }, { 8.0f, 12.0f, 2.0f, 0.0f }, { 34.0f, 70.0f, 37.0f, 4.0f }, { 112.0f, 100.0f, 25.0f, 12.0f } } } });
        var Y = Tensor<float>.MaxPool2D(X, new int[] { 2, 2 }, MathOps.PadType.Value, padvalue: 0, strides: new int[2] { 2,2});
        Assert.Equal(DenseTensor<float>.OfValues(new float[2, 2] { { 20.0f, 30.0f }, { 112.0f, 37.0f } }), Y);
        Y = Tensor<float>.MaxPool2D(X, new int[] { 2, 2 }, MathOps.PadType.Value, padvalue: 0, strides: new int[2] { 1, 1 });
        Assert.Equal(DenseTensor<float>.OfValues(new float[3, 3] { { 20.0f, 30.0f, 30.0f }, { 70.0f, 70.0f, 37.0f }, { 112.0f, 100.0f, 37.0f } }), Y);
        var N = DenseTensor<int>.OfValues(new int[1, 1, 4, 4]
        {{{
            {1,1,2,4 }, {5, 6, 7, 8 }, {3, 2, 1, 0 }, { 1, 2, 3, 4}
        }}});
        var Y2 = Tensor<int>.MaxPool2D(N, new int[] { 2, 2 });
        Assert.Equal(DenseTensor<int>.OfValues(new int[2, 2] { { 6, 8 }, { 3, 4 } }), Y2);

    }
}