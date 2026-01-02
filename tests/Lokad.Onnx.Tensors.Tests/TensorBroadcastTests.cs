namespace Lokad.Onnx.Tensors.Tests;

public class TensorBroadcastTests
{
    [Fact]
    public void CanBroadcastShape()
    {
        Assert.True(Tensor<int>.BroadcastShape(new int[] { 1, 1 }, new int[] { 2 }, out var b));
        Assert.Equal(b, new int[] { 1, 2 });

        Assert.True(Tensor<int>.BroadcastShape(new int[] { 4, 1, 1 }, new int[] { 2 }, out b));
        Assert.Equal(b, new int[] { 4, 1, 2 });

        Assert.True(Tensor<int>.BroadcastShape(new int[] { 1, 2 }, new int[] { 2 }, out b));
        Assert.Equal(b, new int[] { 1, 2 });

        Assert.False(Tensor<int>.BroadcastShape(new int[] { 1, 3 }, new int[] { 2 }, out b));
        Assert.False(Tensor<int>.BroadcastShape(new int[] { 4, 3, 3 }, new int[] { 2 }, out b));

        Assert.True(Tensor<int>.BroadcastShape(new int[] { 2, 1 }, new int[] { 2 }, out b));
        Assert.Equal(b.Append(5).Append(6), new int[] { 2, 2, 5, 6 });

        Assert.True(Tensor<int>.BroadcastShape(new int[] { 2, 1 }, new int[] { 2, 3 }, out b));
        Assert.Equal(b.Append(5).Append(6), new int[] { 2, 3, 5, 6 });
    }

    [Fact]
    public void CanBroadcast()
    {
        Tensor<int> a = new DenseTensor<int>(new[] { 256, 256, 3, });
        Tensor<int> b = new DenseTensor<int>(new[] { 3, 1 });
        b[0,0] = 1;
        b[1, 0] = 2;
        b[2, 0] = 3;
        var bc1 = b.BroadcastDim(1, 255);
        Assert.Equal(1, bc1[0, 204]);
        Assert.Equal(2, bc1[1, 254]);
        Assert.Equal(3, bc1[2, 164]);
        Assert.Throws<ArgumentOutOfRangeException>(() => bc1[3, 256]);

        var ba = Tensor<int>.Broadcast(Tensor<int>.Ones(1, 2), Tensor<int>.Ones(3, 1));
        Assert.Equal(2, ba.Length);
    }

    [Fact]
    public void CanBroadcastLargeShapes()
    {
        var a = new DenseTensor<int>(new[] { 256, 256, 3, });
        var c = new DenseTensor<int>(new[] { 22, 3 });
        var r = Tensor<int>.Broadcast(a, c);
        Assert.NotNull(r);

        r = Tensor<int>.Broadcast(a, new DenseTensor<int>(new[] { 256, 3 }));
        Assert.NotEmpty(r);
        r = Tensor<int>.Broadcast(a, new DenseTensor<int>(new[] { 1, 256, 3 }));
        Assert.NotEmpty(r);
        r = Tensor<int>.Broadcast(a, new DenseTensor<int>(new[] { 256, 1 }));
        Assert.NotEmpty(r);
    }
}

