using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Covers pooled convolution destinations: identical results with rented backing,
/// pool engagement and reuse, and the null-pool fallback across dispatch paths.
/// </summary>
public class ConvPooledOutputTests
{
    static float[] Range(float start, float step, int n)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = start + step * i;
        return v;
    }

    static Tensor<float> Run(Tensor<float> x, Tensor<float> w, Tensor<float>? b, int[] pads, bool fuseRelu, TensorBufferPool? pool)
    {
        return Tensor<float>.Conv2D(x, w, 1, pads, b, null, new[] { 1, 1 }, null, TensorExecutionOptions.Auto, fuseRelu, pool);
    }

    static void CheckAgreement(int n, int c, int h, int w, int m, int kh, bool bias, bool fuseRelu)
    {
        var x = new DenseTensor<float>(Range(-1f, 0.01f, n * c * h * w), new[] { n, c, h, w });
        var wt = new DenseTensor<float>(Range(-0.5f, 0.005f, m * c * kh * kh), new[] { m, c, kh, kh });
        var b = bias ? new DenseTensor<float>(Range(0.25f, -0.01f, m), new[] { m }) : null;
        var pads = new[] { kh / 2, kh / 2, kh / 2, kh / 2 };
        var pool = new TensorBufferPool();
        var pooled = Run(x, wt, b, pads, fuseRelu, pool);
        var plain = Tensor<float>.Conv2D(x, wt, 1, pads, b, null, new[] { 1, 1 }, null, TensorExecutionOptions.Auto, fuseRelu);
        Assert.Equal(plain.ToArray(), pooled.ToArray());
        Assert.Equal(1, pool.AllocatedNew);
    }

    [Fact]
    public void Pointwise_BitMatchesUnpooled()
    {
        CheckAgreement(1, 4, 4, 4, 4, 1, true, false);
    }

    [Fact]
    public void FullPatch_BitMatchesUnpooled()
    {
        CheckAgreement(1, 2, 6, 6, 3, 3, true, true);
    }

    [Fact]
    public void TiledPath_BitMatchesUnpooled()
    {
        CheckAgreement(1, 32, 32, 32, 32, 3, true, true);
    }

    [Fact]
    public void NoBiasNoRelu_BitMatchesUnpooled()
    {
        CheckAgreement(1, 2, 6, 6, 3, 3, false, false);
    }

    [Fact]
    public void PooledOutput_RecyclesAfterReturn()
    {
        var x = new DenseTensor<float>(Range(-1f, 0.1f, 16), new[] { 1, 1, 4, 4 });
        var wt = new DenseTensor<float>(Range(0.5f, -0.05f, 9), new[] { 1, 1, 3, 3 });
        var b = new DenseTensor<float>(new[] { 0.25f }, new[] { 1 });
        var pads = new[] { 1, 1, 1, 1 };
        var pool = new TensorBufferPool();
        var first = (DenseTensor<float>)Run(x, wt, b, pads, false, pool);
        Assert.True(MemoryMarshal.TryGetArray<float>(first.Buffer, out var window) && window.Array is not null);
        pool.Return(window.Array);
        var second = Run(x, wt, b, pads, false, pool);
        Assert.Equal(first.ToArray(), second.ToArray());
        Assert.Equal(1, pool.AllocatedNew);
        Assert.Equal(1, pool.Reused);
    }

    [Fact]
    public void NullPool_PreservesLegacyBehavior()
    {
        var x = new DenseTensor<float>(Range(-1f, 0.1f, 16), new[] { 1, 1, 4, 4 });
        var wt = new DenseTensor<float>(Range(0.5f, -0.05f, 9), new[] { 1, 1, 3, 3 });
        var y = Tensor<float>.Conv2D(x, wt, 1, new[] { 1, 1, 1, 1 }, null, null, new[] { 1, 1 }, null, TensorExecutionOptions.Auto, false);
        Assert.Equal(16, y.Length);
    }

    static bool TrySingle(Tensor<float> x, Tensor<float> w, Tensor<float>? b, int[] pads, int[] strides, bool fuseRelu, TensorBufferPool? pool, out Tensor<float>? y) =>
        Tensor<float>.TryConvSingleChannel2D(x, w, b, 1, pads, null, strides, null, TensorExecutionOptions.Auto, fuseRelu, pool, out y);

    static bool TryDw2(Tensor<float> x, Tensor<float> w, Tensor<float>? b, int[] pads, bool fuseRelu, TensorBufferPool? pool, out Tensor<float>? y) =>
        Tensor<float>.TryConvDepthwise2D(x, w, b, 4, pads, null, new[] { 1, 1 }, null, TensorExecutionOptions.Auto, fuseRelu, pool, out y);

    static bool TryDw1(Tensor<float> x, Tensor<float> w, Tensor<float>? b, int[] pads, bool fuseRelu, TensorBufferPool? pool, out Tensor<float>? y) =>
        Tensor<float>.TryConvDepthwise1D(x, w, b, 4, pads, null, new[] { 1 }, null, TensorExecutionOptions.Auto, fuseRelu, pool, out y);

    static void CheckDirectLane(System.Func<TensorBufferPool?, Tensor<float>> pooled, System.Func<Tensor<float>> plain, int expectedLength)
    {
        var pool = new TensorBufferPool();
        var yPooled = pooled(pool);
        var yPlain = plain();
        Assert.Equal(expectedLength, yPooled.Length);
        Assert.Equal(yPlain.ToArray(), yPooled.ToArray());
        Assert.Equal(1, pool.AllocatedNew);
    }

    [Fact]
    public void SingleChannelStem_BitMatchesUnpooled()
    {
        var x = new DenseTensor<float>(Range(-1f, 0.001f, 1 * 1 * 80 * 200), new[] { 1, 1, 80, 200 });
        var wt = new DenseTensor<float>(Range(-0.5f, 0.0005f, 32 * 1 * 3 * 3), new[] { 32, 1, 3, 3 });
        var b = new DenseTensor<float>(Range(0.25f, -0.01f, 32), new[] { 32 });
        var pads = new[] { 1, 1, 1, 1 };
        var strides = new[] { 1, 1 };
        Assert.True(TrySingle(x, wt, b, pads, strides, true, null, out _));
        CheckDirectLane(pool => { Assert.True(TrySingle(x, wt, b, pads, strides, true, pool, out var y)); return y!; },
            () => { Assert.True(TrySingle(x, wt, b, pads, strides, true, null, out var y)); return y!; },
            1 * 32 * 80 * 200);
    }

    [Fact]
    public void SingleChannel_StridedDilated_BitMatchesUnpooled()
    {
        var x = new DenseTensor<float>(Range(-1f, 0.05f, 1 * 1 * 10 * 10), new[] { 1, 1, 10, 10 });
        var wt = new DenseTensor<float>(Range(-0.5f, 0.02f, 4 * 1 * 3 * 3), new[] { 4, 1, 3, 3 });
        var pool = new TensorBufferPool();
        Assert.True(Tensor<float>.TryConvSingleChannel2D(x, wt, null, 1, new[] { 0, 0, 0, 0 }, null, new[] { 2, 2 }, new[] { 2, 2 }, TensorExecutionOptions.Auto, false, pool, out var yPooled));
        Assert.True(Tensor<float>.TryConvSingleChannel2D(x, wt, null, 1, new[] { 0, 0, 0, 0 }, null, new[] { 2, 2 }, new[] { 2, 2 }, TensorExecutionOptions.Auto, false, null, out var yPlain));
        Assert.Equal(yPlain!.ToArray(), yPooled!.ToArray());
        Assert.Equal(1, pool.AllocatedNew);
    }

    [Fact]
    public void Depthwise2D_BitMatchesUnpooled()
    {
        var x = new DenseTensor<float>(Range(-1f, 0.02f, 1 * 4 * 8 * 8), new[] { 1, 4, 8, 8 });
        var wt = new DenseTensor<float>(Range(-0.5f, 0.01f, 4 * 1 * 3 * 3), new[] { 4, 1, 3, 3 });
        var b = new DenseTensor<float>(Range(0.25f, -0.05f, 4), new[] { 4 });
        var pads = new[] { 1, 1, 1, 1 };
        Assert.True(TryDw2(x, wt, b, pads, true, null, out _));
        CheckDirectLane(pool => { Assert.True(TryDw2(x, wt, b, pads, true, pool, out var y)); return y!; },
            () => { Assert.True(TryDw2(x, wt, b, pads, true, null, out var y)); return y!; },
            1 * 4 * 8 * 8);
    }

    [Fact]
    public void Depthwise1D_BitMatchesUnpooled()
    {
        var x = new DenseTensor<float>(Range(-1f, 0.05f, 1 * 4 * 16), new[] { 1, 4, 16 });
        var wt = new DenseTensor<float>(Range(-0.5f, 0.02f, 4 * 1 * 5), new[] { 4, 1, 5 });
        var pads = new[] { 2, 2 };
        Assert.True(TryDw1(x, wt, null, pads, false, null, out _));
        CheckDirectLane(pool => { Assert.True(TryDw1(x, wt, null, pads, false, pool, out var y)); return y!; },
            () => { Assert.True(TryDw1(x, wt, null, pads, false, null, out var y)); return y!; },
            1 * 4 * 16);
    }

    [Fact]
    public void DirectLanes_DirtyBuffer_MatchesAllocating()
    {
        // Fill rented buffers with nonzero junk, return them, then run each
        // direct lane into the pool: reuse must overwrite every element, so
        // results match fresh allocation bit for bit (complete-overwrite proof).
        var pool = new TensorBufferPool();
        var sx = new DenseTensor<float>(Range(-1f, 0.05f, 1 * 1 * 10 * 10), new[] { 1, 1, 10, 10 });
        var sw = new DenseTensor<float>(Range(-0.5f, 0.02f, 4 * 1 * 3 * 3), new[] { 4, 1, 3, 3 });
        int sLen = 1 * 4 * 10 * 10;
        var dirty = pool.Rent<float>(sLen);
        for (int i = 0; i < dirty.Length; i++) dirty[i] = 7.5f;
        pool.Return(dirty);
        var dx = new DenseTensor<float>(Range(-1f, 0.02f, 1 * 4 * 8 * 8), new[] { 1, 4, 8, 8 });
        var dw = new DenseTensor<float>(Range(-0.5f, 0.01f, 4 * 1 * 3 * 3), new[] { 4, 1, 3, 3 });
        int dLen = 1 * 4 * 8 * 8;
        var dirtyDw = pool.Rent<float>(dLen);
        for (int i = 0; i < dirtyDw.Length; i++) dirtyDw[i] = -3.25f;
        pool.Return(dirtyDw);
        Assert.True(Tensor<float>.TryConvSingleChannel2D(sx, sw, null, 1, new[] { 1, 1, 1, 1 }, null, new[] { 1, 1 }, null, TensorExecutionOptions.Auto, false, pool, out var ySc));
        Assert.True(Tensor<float>.TryConvSingleChannel2D(sx, sw, null, 1, new[] { 1, 1, 1, 1 }, null, new[] { 1, 1 }, null, TensorExecutionOptions.Auto, false, null, out var yScPlain));
        Assert.Equal(yScPlain!.ToArray(), ySc!.ToArray());
        Assert.True(TryDw2(dx, dw, null, new[] { 1, 1, 1, 1 }, false, pool, out var yDw));
        Assert.True(TryDw2(dx, dw, null, new[] { 1, 1, 1, 1 }, false, null, out var yDwPlain));
        Assert.Equal(yDwPlain!.ToArray(), yDw!.ToArray());
        Assert.Equal(2, pool.Reused);
    }

    [Fact]
    public void DirectLanes_NullPool_PreservesBehavior()
    {
        var x = new DenseTensor<float>(Range(-1f, 0.05f, 1 * 1 * 10 * 10), new[] { 1, 1, 10, 10 });
        var wt = new DenseTensor<float>(Range(-0.5f, 0.02f, 4 * 1 * 3 * 3), new[] { 4, 1, 3, 3 });
        Assert.True(TrySingle(x, wt, null, new[] { 1, 1, 1, 1 }, new[] { 1, 1 }, false, null, out var y));
        Assert.Equal(1 * 4 * 10 * 10, y!.Length);
    }
}
