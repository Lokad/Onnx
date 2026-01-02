namespace Lokad.Onnx.Tensors.Tests;

public class TensorConcatUnsqueezeTests
{
    [Fact]
    public void CanConcat()
    {
        var x = DenseTensor<float>.OfValues(new float[2, 3] { { 0.6580f, -1.0969f, -0.4614f }, { -0.1034f, -0.5790f, 0.149f } });
        var y = Tensor<float>.Concat(x, x, 0);
        Assert.NotNull(y);
        y = Tensor<float>.Concat(x, x, 1);
        Assert.NotNull(y);
    }

    [Fact]
    public void CanUnsqueeze()
    {
        var X = (ITensor) DenseTensor<int>.Ones(3, 4, 5);
        var tX = X.Unsqueeze(new int[] { 1 });
        Assert.NotNull(tX);
    }
}

