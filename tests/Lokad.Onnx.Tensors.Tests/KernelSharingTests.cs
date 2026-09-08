using System;
using System.Linq;
using Xunit;

namespace Lokad.Onnx.Tensors.Tests;

public class KernelSharingTests
{
    [Fact]
    public void LayerNorm_AllocMatchesDestination()
    {
        var rnd = new System.Random(51);
        var data = new float[4 * 8];
        var scale = new float[8];
        var bias = new float[8];
        for (int i = 0; i < data.Length; i++) data[i] = (float)rnd.NextDouble() * 2f - 1f;
        for (int i = 0; i < 8; i++) { scale[i] = (float)rnd.NextDouble() + 0.5f; bias[i] = (float)rnd.NextDouble() - 0.5f; }
        var x = new DenseTensor<float>(data.ToArray(), new[] { 4, 8 });
        var s = new DenseTensor<float>(scale.ToArray(), new[] { 8 });
        var b = new DenseTensor<float>(bias.ToArray(), new[] { 8 });
        var alloc = Tensor<float>.LayerNormalization(x, s, b, -1, 1e-5f);
        var dest = DenseTensor<float>.Zeros(4, 8).ToDenseTensor();
        Tensor<float>.LayerNormalization(x, s, b, dest, -1, 1e-5f);
        Assert.Equal(alloc.ToArray(), dest.ToArray());
        var allocNb = Tensor<float>.LayerNormalization(x, s, null, -1, 1e-5f);
        var destNb = DenseTensor<float>.Zeros(4, 8).ToDenseTensor();
        Tensor<float>.LayerNormalization(x, s, null, destNb, -1, 1e-5f);
        Assert.Equal(allocNb.ToArray(), destNb.ToArray());
        var xd = new DenseTensor<double>(data.Select(v => (double)v).ToArray(), new[] { 4, 8 });
        var sd = new DenseTensor<double>(scale.Select(v => (double)v).ToArray(), new[] { 8 });
        var allocD = Tensor<double>.LayerNormalization(xd, sd, null, -1, 1e-5);
        var destD = DenseTensor<double>.Zeros(4, 8).ToDenseTensor();
        Tensor<double>.LayerNormalization(xd, sd, null, destD, -1, 1e-5);
        Assert.Equal(allocD.ToArray(), destD.ToArray());
    }

    [Fact]
    public void Transpose_DoesNotMutateCallerPerm()
    {
        var x = DenseTensor<float>.OfValues(Enumerable.Range(0, 24).Select(i => (float)i).ToArray());
        var xx = new DenseTensor<float>(Enumerable.Range(0, 24).Select(i => (float)i).ToArray(), new[] { 2, 3, 4 });
        var perm = new int[] { 2, 0, 1 };
        var snapshot = (int[])perm.Clone();
        var t = Tensor<float>.Transpose(xx, perm);
        Assert.Equal(snapshot, perm);
        Assert.Equal(new[] { 4, 2, 3 }, t.Dimensions.ToArray());
        Assert.Equal(xx[1, 2, 3], t[3, 1, 2]);
        var perm2 = new int[] { 2, 0, 1 };
        var dest = DenseTensor<float>.Zeros(4, 2, 3).ToDenseTensor();
        Tensor<float>.Transpose(xx, dest, perm2);
        Assert.Equal(snapshot, perm2);
        Assert.Equal(t.ToArray(), dest.ToArray());
        var v = Tensor<float>.Transpose(x, new int[] { 0 });
        Assert.Equal(x.ToArray(), v.ToArray());
    }

    [Fact]
    public void ResizeCubic_NoPerPixelTemps()
    {
        var x = new DenseTensor<float>(Enumerable.Range(0, 1 * 2 * 8 * 8).Select(i => (float)(i % 11)).ToArray(), new[] { 1, 2, 8, 8 });
        var sizes = new[] { 1, 2, 16, 16 };
        for (int i = 0; i < 3; i++) Tensor<float>.Resize(x, sizes, MathOps.ResizeMode.Cubic, MathOps.ResizeCoordinateTransformation.HalfPixel, MathOps.ResizeNearestMode.RoundPreferFloor, -0.75f, null);
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 10; i++) Tensor<float>.Resize(x, sizes, MathOps.ResizeMode.Cubic, MathOps.ResizeCoordinateTransformation.HalfPixel, MathOps.ResizeNearestMode.RoundPreferFloor, -0.75f, null);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        var y = Tensor<float>.Resize(x, sizes, MathOps.ResizeMode.Cubic, MathOps.ResizeCoordinateTransformation.HalfPixel, MathOps.ResizeNearestMode.RoundPreferFloor, -0.75f, null);
        Assert.Equal(sizes, y.Dimensions.ToArray());
        Assert.True(allocated < 200000L, $"cubic resize allocated {allocated} bytes for 10x512-float outputs");
    }
}
