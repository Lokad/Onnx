namespace Lokad.Onnx.Tensors.Tests;

public class TensorConcatUnsqueezeTests
{
    [Fact]
    public void CanConcat()
    {
        var x = DenseTensor<float>.OfValues(new float[2, 3] { { 0.6580f, -1.0969f, -0.4614f }, { -0.1034f, -0.5790f, 0.149f } });
        var y0 = Tensor<float>.Concat(x, x, 0);
        Assert.Equal(new[] { 4, 3 }, y0.Dimensions.ToArray());
        Assert.Equal(0.6580f, y0[0, 0], 5);
        Assert.Equal(0.149f, y0[1, 2], 5);
        Assert.Equal(0.6580f, y0[2, 0], 5);
        Assert.Equal(0.149f, y0[3, 2], 5);

        var y1 = Tensor<float>.Concat(x, x, 1);
        Assert.Equal(new[] { 2, 6 }, y1.Dimensions.ToArray());
        Assert.Equal(-1.0969f, y1[0, 1], 5);
        Assert.Equal(0.6580f, y1[0, 3], 5);
        Assert.Equal(0.149f, y1[1, 5], 5);
    }

    [Fact]
    public void CanUnsqueeze()
    {
        var X = (ITensor) DenseTensor<int>.Ones(3, 4, 5);
        var tX = X.Unsqueeze(new int[] { 1 });
        Assert.Equal(new[] { 3, 1, 4, 5 }, ((Tensor<int>)tX).Dimensions.ToArray());
        Assert.Equal(1, ((Tensor<int>)tX)[2, 0, 3, 4]);
    }

    [Fact]
    public void CanUnsqueezeNegativeAxis()
    {
        var d = DenseTensor<float>.OfShape(2, 3);
        for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++) d[i, j] = i * 10 + j;
        var u = (Tensor<float>)((ITensor)d).Unsqueeze(new int[] { -1 });
        Assert.Equal(new[] { 2, 3, 1 }, u.Dimensions.ToArray());
        Assert.Equal(12f, u[1, 2, 0], 5);
        var u2 = (Tensor<float>)((ITensor)d).Unsqueeze(new int[] { 0, -1 });
        Assert.Equal(new[] { 1, 2, 3, 1 }, u2.Dimensions.ToArray());
        Assert.Equal(12f, u2[0, 1, 2, 0], 5);
    }

    [Fact]
    public void CanConcatBroadcastViewMatchesDense()
    {
        var column = DenseTensor<float>.OfValues(new float[,] { { 1f }, { 2f } });
        var broadcast = Tensor<float>.Expand(column, new int[] { 2, 3 });
        var other = DenseTensor<float>.OfValues(new float[,] { { 7f, 8f, 9f }, { 10f, 11f, 12f } });
        var actual = Tensor<float>.Concat(broadcast, other, 0);
        Assert.Equal(new[] { 4, 3 }, actual.Dimensions.ToArray());
        Assert.Equal(1f, actual[0, 0], 5);
        Assert.Equal(2f, actual[1, 2], 5);
        Assert.Equal(7f, actual[2, 0], 5);
        Assert.Equal(12f, actual[3, 2], 5);
    }

    [Fact]
    public void CanConcatThreeInputs()
    {
        var first = DenseTensor<int>.OfValues(new int[,] { { 1, 2 } });
        var second = DenseTensor<int>.OfValues(new int[,] { { 3, 4 } });
        var third = DenseTensor<int>.OfValues(new int[,] { { 5, 6 } });
        var actual = Tensor<int>.Concat(new Tensor<int>[] { first, second, third }, 0);
        Assert.Equal(new[] { 3, 2 }, actual.Dimensions.ToArray());
        Assert.Equal(1, actual[0, 0]);
        Assert.Equal(4, actual[1, 1]);
        Assert.Equal(6, actual[2, 1]);

        var mixed = Tensor<int>.Concat(new Tensor<int>[] { first, Tensor<int>.Expand(second, new int[] { 1, 2 }) }, -1);
        Assert.Equal(new[] { 1, 4 }, mixed.Dimensions.ToArray());
        Assert.Equal(2, mixed[0, 1]);
        Assert.Equal(4, mixed[0, 3]);
    }

    [Fact]
    public void CanUnsqueezeRejectsInvalidAxes()
    {
        var data = (ITensor)DenseTensor<float>.OfShape(2, 3);
        Assert.Throws<ArgumentException>(() => data.Unsqueeze(new int[] { -5 }));
        Assert.Throws<ArgumentException>(() => data.Unsqueeze(new int[] { 4 }));
        Assert.Throws<ArgumentException>(() => data.Unsqueeze(new int[] { 2, -2 }));
    }
}
