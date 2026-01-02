namespace Lokad.Onnx.Tensors.Tests;

public class TensorPadTests
{
    [Fact]
    public void CanPadLeft()
    {
        var a = new DenseTensor<int>(new[] { 256, 256, 3, });
        var b = new DenseTensor<int>(new[] { 3, 1 });
        b[0, 0] = 1;
        b[1, 0] = 2;
        b[2, 0] = 3;
        var pb = b.PadLeft();
        Assert.Equal(3, pb.Rank);
        Assert.Equal(2, pb[0,1,0]);
    }
}

