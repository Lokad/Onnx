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
        Assert.Empty(Tensor<int>.Broadcast(a, c));
        var r = Tensor<int>.Broadcast(a, new DenseTensor<int>(new[] { 256, 3 }));
        Assert.Equal(2, r.Length);
        Assert.Equal(new[] { 256, 256, 3 }, r[0].Dimensions.ToArray());
        Assert.Equal(new[] { 256, 256, 3 }, r[1].Dimensions.ToArray());
        r = Tensor<int>.Broadcast(a, new DenseTensor<int>(new[] { 1, 256, 3 }));
        Assert.Equal(2, r.Length);
        Assert.Equal(new[] { 256, 256, 3 }, r[1].Dimensions.ToArray());
        r = Tensor<int>.Broadcast(a, new DenseTensor<int>(new[] { 256, 1 }));
        Assert.Equal(2, r.Length);
        Assert.Equal(new[] { 256, 256, 3 }, r[1].Dimensions.ToArray());
    }

    [Fact]
    public void BroadcastedViewDensifiesExactly()
    {
        var rnd = new System.Random(20260907);
        foreach (var (srcDims, dstDims) in new[] { (new[] { 384 }, new[] { 8, 384 }), (new[] { 1, 384 }, new[] { 8, 384 }), (new[] { 8, 1 }, new[] { 8, 384 }), (new[] { 1 }, new[] { 4, 5 }), (new[] { 4, 1, 6 }, new[] { 4, 5, 6 }) })
        {
            int n = 1;
            foreach (var d in srcDims) n *= d;
            var data = new float[n];
            for (int i = 0; i < n; i++) data[i] = (float)rnd.NextDouble();
            var src = new DenseTensor<float>(data, srcDims);
            var dst = new DenseTensor<float>(dstDims);
            INumericTensor.Broadcast(src, dst, out var bA, out var bB);
            var view = bA as BroadcastedTensor<float> ?? (BroadcastedTensor<float>)bB;
            var fast = view.ToDenseTensor().ToArray();
            for (int i = 0; i < fast.Length; i++) Assert.Equal((float)view.GetValue(i), fast[i]);
        }
    }

    [Fact]
    public void BroadcastShape_Allocates_ByRank_NotElements()
    {
        var x = new int[] { 1000, 1000 };
        var y = new int[] { 1, 1000 };
        Assert.True(Tensor<int>.BroadcastShape(x, y, out _));
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 10; i++) Assert.True(Tensor<int>.BroadcastShape(x, y, out var b));
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.True(Tensor<int>.BroadcastShape(x, y, out var shape));
        Assert.Equal(new int[] { 1000, 1000 }, shape);
        Assert.True(allocated < 4096L, $"shape broadcast allocated {allocated} bytes for rank-2 shapes");
    }

    [Fact]
    public void BroadcastTensorAgainstSpan_BuildsView_WithoutElementStorage()
    {
        var x = Tensor<int>.Ones(1, 1000);
        var target = new int[] { 1000, 1000 };
        Assert.True(Tensor<int>.Broadcast(x, target, out _));
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 10; i++) Assert.True(Tensor<int>.Broadcast(x, target, out var view));
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.True(Tensor<int>.Broadcast(x, target, out var result));
        Assert.Equal(target, result.Dimensions.ToArray());
        Assert.Equal(1, result.GetValue(0));
        Assert.Equal(1, result.GetValue(999999));
        Assert.True(allocated < 8192L, $"span broadcast allocated {allocated} bytes for rank-2 shapes");
    }

    [Fact]
    public void BroadcastShape_Incompatible_ReturnsFalse()
    {
        Assert.False(Tensor<int>.BroadcastShape(new int[] { 2, 3 }, new int[] { 4 }, out var b));
        Assert.Null(b);
    }


}


