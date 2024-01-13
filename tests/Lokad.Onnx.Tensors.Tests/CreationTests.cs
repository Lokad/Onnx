namespace Lokad.Onnx.Tensors.Tests;

public class CreationTests
{
    [Fact]
    public void CanCreateARange()
    {
        var t = Tensor<int>.Arange(0, 21);
        Assert.Equal(5, t[5]);
        var t2 = t.Reshape(7, 3);
        Assert.Equal(20, t2[6, 2]);
    }
}

