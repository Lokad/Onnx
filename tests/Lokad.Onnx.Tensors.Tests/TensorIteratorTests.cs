namespace Lokad.Onnx.Tensors.Tests;

public class TensorIteratorTests
{
    [Fact]
    public void CanIterateDims()
    {
        var a = new DenseTensor<int>(new[] { 256, 212, 3, });
        var di = a.GetDimensionsIterator(0..^1);
        di = a.GetDimensionsIterator();
        while (di.Next() != null)
        {
            var i = di.Index;
        }
    }
}

