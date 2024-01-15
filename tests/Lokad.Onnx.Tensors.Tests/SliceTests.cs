namespace Lokad.Onnx.Tensors.Tests;


public class SliceTests
{
    [Fact]
    public void CanSlice()
    {
        Tensor<int> t = Tensor<int>.Arange(0, 12).Reshape(3, 4);
        var s1 = t[":", "2:4"];
        Assert.Equal(6, s1.Length);
        Assert.Equal(11, s1[2, 1]);
        var sd = t["1:3", "2:4"];
        Assert.NotNull(sd);

        var s3 = t["0", "2"];
        Assert.NotNull(s3);

        var t2 = t.Reshape(2, 2, 3);
        var se = t2["...", 2];
        Assert.NotNull(se);

        var ac = Tensor<int>.Arange(20, 24).Reshape(2, 2);
        t2["...",2] = ac;
        ac = ac.Reshape(4, 1);
        Assert.Throws<ArgumentException>(() => t2["...",2] = ac);
        //Assert.Equal(3, t[0,0]);
        //var s2 = t.SliceDims("2", "3");
        //Assert.NotNull(s2);
    }
}
