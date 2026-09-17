using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// B3 P1 prototype guard: the prefetch tile variant must agree bitwise with
/// the plain 12-row tile on every covered path (direct tile, full 12-row
/// method with flag on/off including K tails, and dispatched MatMul2D on
/// both tiled-composer and legacy shapes). Prefetch is a hint only, so any
/// bit divergence is a bug. Delete with the prototype on a negative mirror
/// verdict; fold into the kernel tests on promotion.
/// </summary>
public class PackedPrefetchAgreementTests
{
    static float[] Range(float start, float step, int n)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = start + step * i;
        return v;
    }

    static void Pack(int n, int k, float[] b, float[] p)
    {
        unsafe
        {
            fixed (float* bp = b, pp = p)
            {
                MathOps.PackPanelsB(n, k, bp, pp);
            }
        }
    }

    static void AssertBitsEqual(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(
                System.BitConverter.SingleToInt32Bits(expected[i]),
                System.BitConverter.SingleToInt32Bits(actual[i]));
        }
    }

    static void RunFull12(int m, int n, int k, float[] a, float[] p, float[] c)
    {
        unsafe
        {
            fixed (float* ap = a, pp = p, cp = c)
            {
                MathOps.mm_unsafe_vectorized_avx512_12x32packed(m, n, k, ap, pp, cp, false);
            }
        }
    }

    static void RunTilePair(int n, int k, float[] a, float[] p, float[] cPlain, float[] cPref, int kb, int pOff)
    {
        unsafe
        {
            fixed (float* ap = a, pp = p, cp1 = cPlain, cp2 = cPref)
            {
                MathOps.mm_avx512_12x32packed_tile(12, n, ap, pp + pOff, cp1, k, kb, false);
                MathOps.mm_avx512_12x32packed_tile_pref(12, n, ap, pp + pOff, cp2, k, kb, false);
            }
        }
    }

    [SkippableFact]
    public void TilePref_MatchesPlainTile_Bitwise()
    {
        Skip.If(!Avx512F.IsSupported, "AVX512F not available on this machine.");
        const int n = 256;
        const int k = 512;
        var a = Range(-1f, 0.01f, 12 * n);
        var b = Range(-2f, 0.001f, n * k);
        var p = new float[n * k];
        Pack(n, k, b, p);
        var cPlain = new float[12 * k];
        var cPref = new float[12 * k];
        int tileStep = 32;
        int tiles = k / tileStep;
        for (int tb = 0; tb < tiles; tb++)
        {
            int kb = tb * tileStep;
            int pOff = tb * n * tileStep;
            RunTilePair(n, k, a, p, cPlain, cPref, kb, pOff);
        }
        AssertBitsEqual(cPlain, cPref);
    }

    [SkippableFact]
    public void FullMethod_FlagOnOff_Bitwise_InclKTail()
    {
        Skip.If(!Avx512F.IsSupported, "AVX512F not available on this machine.");
        var shapes = new[] { (12, 640, 64), (12, 640, 100), (24, 256, 520) };
        foreach (var shape in shapes)
        {
            int m = shape.Item1, n = shape.Item2, k = shape.Item3;
            var a = Range(-1f, 0.0007f, m * n);
            var b = Range(0.5f, -0.00013f, n * k);
            var p = new float[n * k];
            Pack(n, k, b, p);
            var cOff = new float[m * k];
            var cOn = new float[m * k];
            bool saved = MathOps.PackedSweepPrefetch;
            try
            {
                MathOps.PackedSweepPrefetch = false;
                RunFull12(m, n, k, a, p, cOff);
                MathOps.PackedSweepPrefetch = true;
                RunFull12(m, n, k, a, p, cOn);
            }
            finally { MathOps.PackedSweepPrefetch = saved; }
            AssertBitsEqual(cOff, cOn);
        }
    }

    [SkippableFact]
    public void DispatchedMatMul_FlagOnOff_Bitwise()
    {
        Skip.If(!Avx512F.IsSupported, "AVX512F not available on this machine.");
        var shapes = new[] { (40, 256, 2048), (12, 64, 64) };
        var rnd = new System.Random(9137);
        foreach (var shape in shapes)
        {
            int m = shape.Item1, n = shape.Item2, k = shape.Item3;
            var x = Tensor<float>.Zeros(m, n).ToDenseTensor();
            var w = Tensor<float>.Zeros(n, k).ToDenseTensor();
            for (int i = 0; i < x.Length; i++) x.SetValue(i, rnd.NextSingle());
            for (int i = 0; i < w.Length; i++) w.SetValue(i, rnd.NextSingle());
            bool saved = MathOps.PackedSweepPrefetch;
            float[] cOff, cOn;
            try
            {
                MathOps.PackedSweepPrefetch = false;
                cOff = Tensor<float>.MatMul2D(x, w).ToArray();
                MathOps.PackedSweepPrefetch = true;
                cOn = Tensor<float>.MatMul2D(x, w).ToArray();
            }
            finally { MathOps.PackedSweepPrefetch = saved; }
            Assert.Equal(cOff.Length, cOn.Length);
            for (int i = 0; i < cOff.Length; i++)
            {
                Assert.Equal(
                    System.BitConverter.SingleToInt32Bits(cOff[i]),
                    System.BitConverter.SingleToInt32Bits(cOn[i]));
            }
        }
    }
}
