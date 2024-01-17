namespace Lokad.Onnx.Tensors.Tests;


public class MathTests
{
    [Fact]
    public void CanIterateDims()
    {
        var t = Tensor<int>.Ones(3, 4, 3);
        var id = t.GetDimensionsIterator(0..);
    }
}

