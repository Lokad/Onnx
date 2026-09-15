using System;
using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx.Backend.Tests;

// Overwrite-contract agreement (M4): panel-packed kernels with overwrite:true
// over garbage destinations must match cleared-plus-accumulate bit for bit,
// and every MatMul entry point must honor uninitialized destinations.
public class MatMulOverwriteAgreementTests
{
    static void Fill(Random rnd, float[] a, float scale)
    {
        for (int i = 0; i < a.Length; i++) a[i] = scale * (float)(rnd.NextDouble() * 2 - 1);
    }

    static void Garbage(Random rnd, float[] a)
    {
        float[] specials = new float[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity, -0f, float.MaxValue, float.MinValue, 12345.678f };
        for (int i = 0; i < a.Length; i++) a[i] = specials[i % specials.Length] * (0.5f + (float)rnd.NextDouble());
    }

    static void Pack(int n, int k, float[] b, float[] p)
    {
        unsafe
        {
            fixed (float* bp = b, pp = p)
                MathOps.PackPanelsB(n, k, bp, pp);
        }
    }

    static void AssertBits(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                what + "[" + i + "] differs: " + expected[i] + " vs " + actual[i] + ".");
    }

    static void CheckFull(string name, int m, int n, int k)
    {
        var rnd = new Random(m * 100003 + k * 101 + n);
        var a = new float[m * n];
        var b = new float[n * k];
        Fill(rnd, a, 1f);
        Fill(rnd, b, 0.25f);
        var p = new float[n * k];
        Pack(n, k, b, p);
        var cleared = new float[m * k];
        var garbage = new float[m * k];
        Garbage(rnd, garbage);
        unsafe
        {
            fixed (float* ap = a, pp = p)
            {
                fixed (float* cp = cleared)
                    Call(name, m, n, k, ap, pp, cp, false);
                fixed (float* gp = garbage)
                    Call(name, m, n, k, ap, pp, gp, true);
            }
        }
        AssertBits(cleared, garbage, name + "-m" + m + "-k" + k);
    }

    static unsafe void Call(string name, int m, int n, int k, float* a, float* p, float* c, bool overwrite)
    {
        if (name == "12")
            MathOps.mm_unsafe_vectorized_avx512_12x32packed(m, n, k, a, p, c, overwrite);
        else if (name == "8")
            MathOps.mm_unsafe_vectorized_avx512_8x32packed(m, n, k, a, p, c, overwrite);
        else if (name == "6")
            MathOps.mm_unsafe_vectorized_avx512_6x32packed(m, n, k, a, p, c, overwrite);
        else if (name == "3")
            MathOps.mm_unsafe_vectorized_intrinsics_3x4packed(m, n, k, a, p, c, overwrite);
        else
            MathOps.mm_unsafe_vectorized_intrinsics_2x4packed(m, n, k, a, p, c, overwrite);
    }

    [SkippableFact]
    public void Avx512FullKernels_OverwriteMatchesAccumulate()
    {
        Skip.If(!Avx512F.IsSupported || !Fma.IsSupported, "AVX512 packed kernels need AVX512F/FMA.");
        foreach (int k in new int[] { 32, 33, 34, 35, 36, 37, 38, 39, 40, 48, 56 })
        {
            CheckFull("12", 12, 64, k);
            CheckFull("8", 8, 64, k);
            CheckFull("6", 6, 64, k);
        }
        CheckFull("12", 24, 64, 40);
        CheckFull("8", 16, 64, 56);
        CheckFull("6", 12, 96, 33);
    }

    [SkippableFact]
    public void NarrowFullKernels_OverwriteMatchesAccumulate()
    {
        Skip.If(!Fma.IsSupported, "Narrow packed kernels need FMA.");
        foreach (int k in new int[] { 32, 33, 34, 35, 36, 37, 38, 39, 40, 48, 56 })
        {
            CheckFull("3", 6, 64, k);
            CheckFull("2", 8, 64, k);
        }
        CheckFull("3", 9, 64, 39);
        CheckFull("2", 10, 64, 37);
    }

    static float[] MatMulIntoGarbage(float[] x, int[] xd, float[] y, int[] yd)
    {
        var X = new DenseTensor<float>(x, xd);
        var Y = new DenseTensor<float>(y, yd);
        var shape = new int[] { xd[xd.Length - 2], yd[yd.Length - 1] };
        int len = shape[0] * shape[1];
        var g = new float[len];
        Garbage(new Random(99), g);
        var D = new DenseTensor<float>(new Memory<float>(g), shape);
        return ((Tensor<float>)Tensor<float>.MatMul(X, Y, D, TensorExecutionOptions.Auto)).ToArray();
    }

    static float[] MatMulFresh(float[] x, int[] xd, float[] y, int[] yd)
    {
        var X = new DenseTensor<float>(x, xd);
        var Y = new DenseTensor<float>(y, yd);
        return ((Tensor<float>)Tensor<float>.MatMul(X, Y, TensorExecutionOptions.Auto)).ToArray();
    }

    static void CheckMatMul(int m, int n, int k, int seed)
    {
        var rnd = new Random(seed);
        var x = new float[m * n];
        var y = new float[n * k];
        Fill(rnd, x, 1f);
        Fill(rnd, y, 0.25f);
        AssertBits(MatMulFresh(x, new int[] { m, n }, y, new int[] { n, k }),
            MatMulIntoGarbage(x, new int[] { m, n }, y, new int[] { n, k }), "matmul-" + m + "x" + n + "x" + k);
    }

    [Fact]
    public void MatMulInto_GarbageDestinationMatchesFresh()
    {
        CheckMatMul(16, 256, 520, 7);
        CheckMatMul(40, 256, 2048, 8);
        CheckMatMul(1, 64, 64, 9);
        CheckMatMul(5, 128, 257, 10);
        CheckMatMul(31, 128, 260, 11);
    }

    [Fact]
    public void PooledMatMul_ReusedGarbageMatchesFresh()
    {
        var rnd = new Random(21);
        int m = 16, n = 256, k = 520;
        var x = new float[m * n];
        var y = new float[n * k];
        Fill(rnd, x, 1f);
        Fill(rnd, y, 0.25f);
        var pool = new TensorBufferPool();
        var X = new DenseTensor<float>(x, new int[] { m, n });
        var Y = new DenseTensor<float>(y, new int[] { n, k });
        var first = ((Tensor<float>)Tensor<float>.MatMul(X, Y, TensorExecutionOptions.Auto, pool)).ToArray();
        var second = ((Tensor<float>)Tensor<float>.MatMul(X, Y, TensorExecutionOptions.Auto, pool)).ToArray();
        AssertBits(first, second, "pooled-reuse");
        AssertBits(MatMulFresh(x, new int[] { m, n }, y, new int[] { n, k }), second, "pooled-fresh");
    }

    [Fact]
    public void PooledTiledConv_ReusedGarbageMatchesFresh()
    {
        var x = new float[1 * 32 * 32 * 32];
        var w = new float[32 * 32 * 3 * 3];
        var b = new float[32];
        var rnd = new Random(33);
        Fill(rnd, x, 1f);
        Fill(rnd, w, 0.1f);
        Fill(rnd, b, 0.1f);
        var X = new DenseTensor<float>(x, new int[] { 1, 32, 32, 32 });
        var W = new DenseTensor<float>(w, new int[] { 32, 32, 3, 3 });
        var B = new DenseTensor<float>(b, new int[] { 32 });
        var pool = new TensorBufferPool();
        var first = ((Tensor<float>)Tensor<float>.Conv2D(X, W, 1, new int[] { 1, 1, 1, 1 }, B, new int[] { 3, 3 }, new int[] { 1, 1 }, null, TensorExecutionOptions.Auto, false, pool)).ToArray();
        var second = ((Tensor<float>)Tensor<float>.Conv2D(X, W, 1, new int[] { 1, 1, 1, 1 }, B, new int[] { 3, 3 }, new int[] { 1, 1 }, null, TensorExecutionOptions.Auto, false, pool)).ToArray();
        AssertBits(first, second, "conv-reuse");
    }
}
