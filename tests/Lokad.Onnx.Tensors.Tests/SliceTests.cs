namespace Lokad.Onnx.Tensors.Tests;


public class SliceTests
{
    [Fact]
    public void CanSlice()
    {
        var s = new SliceIndex(":2");
        Tensor<int> t = Tensor<int>.Arange(0, 12).Reshape(3, 4);
        var sd = t["1:3", "2:4"];
        Assert.NotNull(sd);
        Assert.Equal(3, t[0,0]);
        //var s2 = t.SliceDims("2", "3");
        //Assert.NotNull(s2);
    }
}
