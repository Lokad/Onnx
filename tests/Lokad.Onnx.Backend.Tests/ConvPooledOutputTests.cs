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
}

