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
}
