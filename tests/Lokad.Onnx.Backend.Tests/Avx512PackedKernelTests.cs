namespace Lokad.Onnx.Backend.Tests;

using System.Runtime.Intrinsics.X86;

/// <summary>
/// Guards the 6-row AVX512 packed kernel: bitwise agreement with the 2-row
/// packed nest on panel-exact shapes, double-oracle agreement through masked
/// tails, no leakage from fault-suppressed lanes, and the row-count contract.
/// Skipped where AVX512F is unavailable; dispatched behavior is covered by
/// corpus lanes and graph tests.
/// </summary>
public class Avx512PackedKernelTests
{
    const int Seed = 5105;

    static DenseTensor<float> FillRect(int rows, int cols, Random rnd)
    {
        var t = Tensor<float>.Zeros(rows, cols).ToDenseTensor();
        for (int i = 0; i < t.Length; i++) t.SetValue(i, rnd.NextSingle());
        return t;
    }

    static unsafe void RunSix(int m, int n, int k, DenseTensor<float> a, DenseTensor<float> b, DenseTensor<float> p, DenseTensor<float> c)
    {
        using var pa = a.Buffer.Pin();
        using var pb = b.Buffer.Pin();
        using var pp = p.Buffer.Pin();
        using var pc = c.Buffer.Pin();
        MathOps.PackPanelsB(n, k, (float*)pb.Pointer, (float*)pp.Pointer);
        MathOps.mm_unsafe_vectorized_avx512_6x32packed(m, n, k, (float*)pa.Pointer, (float*)pp.Pointer, (float*)pc.Pointer);
    }

    static unsafe void RunTwo(int m, int n, int k, DenseTensor<float> a, DenseTensor<float> b, DenseTensor<float> p, DenseTensor<float> c)
    {
        using var pa = a.Buffer.Pin();
        using var pb = b.Buffer.Pin();
        using var pp = p.Buffer.Pin();
        using var pc = c.Buffer.Pin();
        MathOps.PackPanelsB(n, k, (float*)pb.Pointer, (float*)pp.Pointer);
        MathOps.mm_unsafe_vectorized_intrinsics_2x4packed(m, n, k, (float*)pa.Pointer, (float*)pp.Pointer, (float*)pc.Pointer);
    }
    static double[] Oracle(DenseTensor<float> a, DenseTensor<float> b, int m, int n, int k)
    {
        var aa = a.ToArray();
        var bb = b.ToArray();
        var c = new double[m * k];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < k; j++)
            {
                double acc = 0.0;
                for (int l = 0; l < n; l++) acc += (double)aa[i * n + l] * bb[l * k + j];
                c[i * k + j] = acc;
            }
        return c;
    }

    static void AssertNear(double[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            double tol = 2e-4 * (1.0 + System.Math.Abs(expected[i]));
            Assert.True(System.Math.Abs(actual[i] - expected[i]) <= tol, "index " + i);
        }
    }

    [SkippableFact]
    public unsafe void SixRowMatchesTwoRowBitwise()
    {
        Skip.If(!Avx512F.IsSupported, "AVX512F not available on this machine.");
        var rnd = new Random(Seed);
        foreach (var shape in new[] { (6, 64, 64), (12, 128, 96), (18, 64, 192) })
        {
            var a = FillRect(shape.Item1, shape.Item2, rnd);
            var b = FillRect(shape.Item2, shape.Item3, rnd);
            var p = Tensor<float>.Zeros(shape.Item2, shape.Item3).ToDenseTensor();
            var c1 = Tensor<float>.Zeros(shape.Item1, shape.Item3).ToDenseTensor();
            var c2 = Tensor<float>.Zeros(shape.Item1, shape.Item3).ToDenseTensor();
            RunSix(shape.Item1, shape.Item2, shape.Item3, a, b, p, c1);
            RunTwo(shape.Item1, shape.Item2, shape.Item3, a, b, p, c2);
            Assert.True(c1.Buffer.Span.SequenceEqual(c2.Buffer.Span),
                "6-row diverges bitwise from 2-row on " + shape.Item1 + "x" + shape.Item2 + "x" + shape.Item3);
        }
    }

    [SkippableFact]
    public unsafe void MaskedTailMatchesOracle()
    {
        Skip.If(!Avx512F.IsSupported, "AVX512F not available on this machine.");
        var rnd = new Random(Seed);
        foreach (var shape in new[] { (12, 128, 100), (6, 64, 20), (18, 256, 8) })
        {
            var a = FillRect(shape.Item1, shape.Item2, rnd);
            var b = FillRect(shape.Item2, shape.Item3, rnd);
            var p = Tensor<float>.Zeros(shape.Item2, shape.Item3).ToDenseTensor();
            var c = Tensor<float>.Zeros(shape.Item1, shape.Item3).ToDenseTensor();
            RunSix(shape.Item1, shape.Item2, shape.Item3, a, b, p, c);
            AssertNear(Oracle(a, b, shape.Item1, shape.Item2, shape.Item3), c.ToArray());
        }
    }

    [SkippableFact]
    public unsafe void RejectsNonMultipleOfSix()
    {
        Skip.If(!Avx512F.IsSupported, "AVX512F not available on this machine.");
        var a = FillRect(7, 8, new Random(Seed));
        var b = FillRect(8, 32, new Random(Seed));
        var p = Tensor<float>.Zeros(8, 32).ToDenseTensor();
        var c = Tensor<float>.Zeros(7, 32).ToDenseTensor();
        Assert.Throws<System.ArgumentException>(() =>
        {
            using var pa = a.Buffer.Pin();
            using var pb = b.Buffer.Pin();
            using var pp = p.Buffer.Pin();
            using var pc = c.Buffer.Pin();
            MathOps.mm_unsafe_vectorized_avx512_6x32packed(7, 8, 32, (float*)pa.Pointer, (float*)pp.Pointer, (float*)pc.Pointer);
        });
    }
    [SkippableFact]
    public unsafe void SixRowCompositionMatchesTwoRowBitwise()
    {
        // The dispatched composition (6-row heads with 2-row tails) must
        // agree bit-wise with single 2-row calls: rows are independent and
        // every piece shares panel layout and FMA order.
        Skip.If(!Avx512F.IsSupported, "AVX512F not available on this machine.");
        var rnd = new Random(Seed);
        foreach (var shape in new[] { (12, 64, 64), (16, 128, 96), (40, 64, 64) })
        {
            var a = FillRect(shape.Item1, shape.Item2, rnd);
            var b = FillRect(shape.Item2, shape.Item3, rnd);
            var p = Tensor<float>.Zeros(shape.Item2, shape.Item3).ToDenseTensor();
            var c1 = Tensor<float>.Zeros(shape.Item1, shape.Item3).ToDenseTensor();
            var c2 = Tensor<float>.Zeros(shape.Item1, shape.Item3).ToDenseTensor();
            using var pa = a.Buffer.Pin();
            using var pb = b.Buffer.Pin();
            using var pp = p.Buffer.Pin();
            using var pc1 = c1.Buffer.Pin();
            using var pc2 = c2.Buffer.Pin();
            MathOps.PackPanelsB(shape.Item2, shape.Item3, (float*)pb.Pointer, (float*)pp.Pointer);
            ComposedRows(shape.Item1, shape.Item2, shape.Item3, (float*)pa.Pointer, (float*)pp.Pointer, (float*)pc1.Pointer);
            RunTwo(shape.Item1, shape.Item2, shape.Item3, a, b, p, c2);
            Assert.True(c1.Buffer.Span.SequenceEqual(c2.Buffer.Span),
                "6-row composition diverges bitwise from 2-row on " + shape.Item1 + "x" + shape.Item2 + "x" + shape.Item3);
        }
    }

    static unsafe void ComposedRows(int m, int n, int k, float* x, float* packed, float* dest)
    {
        // Mirrors the dispatched AVX512 decomposition for the covered Ms.
        int main = (m / 6) * 6;
        int rem = m - main;
        if (rem == 1) { main -= 6; rem = 7; }
        if (main > 0) MathOps.mm_unsafe_vectorized_avx512_6x32packed(main, n, k, x, packed, dest);
        float* xr = x + main * n;
        float* dr = dest + main * k;
        switch (rem)
        {
            case 7:
                MathOps.mm_unsafe_vectorized_intrinsics_3x4packed(3, n, k, xr, packed, dr);
                MathOps.mm_unsafe_vectorized_intrinsics_2x4packed(2, n, k, xr + 3 * n, packed, dr + 3 * k);
                MathOps.mm_unsafe_vectorized_intrinsics_2x4packed(2, n, k, xr + 5 * n, packed, dr + 5 * k);
                break;
            case 5:
                MathOps.mm_unsafe_vectorized_intrinsics_3x4packed(3, n, k, xr, packed, dr);
                MathOps.mm_unsafe_vectorized_intrinsics_2x4packed(2, n, k, xr + 3 * n, packed, dr + 3 * k);
                break;
            case 4:
                MathOps.mm_unsafe_vectorized_intrinsics_2x4packed(4, n, k, xr, packed, dr);
                break;
            case 3:
                MathOps.mm_unsafe_vectorized_intrinsics_3x4packed(3, n, k, xr, packed, dr);
                break;
            case 2:
                MathOps.mm_unsafe_vectorized_intrinsics_2x4packed(2, n, k, xr, packed, dr);
                break;
        }
    }
    static unsafe void RunTwelve(int m, int n, int k, DenseTensor<float> a, DenseTensor<float> b, DenseTensor<float> p, DenseTensor<float> c)
    {
        using var pa = a.Buffer.Pin();
        using var pb = b.Buffer.Pin();
        using var pp = p.Buffer.Pin();
        using var pc = c.Buffer.Pin();
        MathOps.PackPanelsB(n, k, (float*)pb.Pointer, (float*)pp.Pointer);
        MathOps.mm_unsafe_vectorized_avx512_12x32packed(m, n, k, (float*)pa.Pointer, (float*)pp.Pointer, (float*)pc.Pointer);
    }

    static double[] Oracle12(DenseTensor<float> a, DenseTensor<float> b, int m, int n, int k)
    {
        var aa = a.ToArray();
        var bb = b.ToArray();
        var c = new double[m * k];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < k; j++)
            {
                double acc = 0.0;
                for (int l = 0; l < n; l++) acc += (double)aa[i * n + l] * bb[l * k + j];
                c[i * k + j] = acc;
            }
        return c;
    }

    static void AssertNear12(double[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            double tol = 2e-4 * (1.0 + System.Math.Abs(expected[i]));
            Assert.True(System.Math.Abs(actual[i] - expected[i]) <= tol, "index " + i);
        }
    }

    [SkippableFact]
    public unsafe void TwelveRowMatchesSixRowBitwise()
    {
        Skip.If(!Avx512F.IsSupported, "AVX512F not available on this machine.");
        var rnd = new Random(Seed);
        foreach (var shape in new[] { (12, 64, 64), (24, 128, 96) })
        {
            var a = FillRect(shape.Item1, shape.Item2, rnd);
            var b = FillRect(shape.Item2, shape.Item3, rnd);
            var p = Tensor<float>.Zeros(shape.Item2, shape.Item3).ToDenseTensor();
            var c1 = Tensor<float>.Zeros(shape.Item1, shape.Item3).ToDenseTensor();
            var c2 = Tensor<float>.Zeros(shape.Item1, shape.Item3).ToDenseTensor();
            RunTwelve(shape.Item1, shape.Item2, shape.Item3, a, b, p, c1);
            RunSix(shape.Item1, shape.Item2, shape.Item3, a, b, p, c2);
            Assert.True(c1.Buffer.Span.SequenceEqual(c2.Buffer.Span),
                "12-row diverges bitwise from 6-row on " + shape.Item1 + "x" + shape.Item2 + "x" + shape.Item3);
        }
    }

    [SkippableFact]
    public unsafe void TwelveRowMaskedTailMatchesOracle()
    {
        Skip.If(!Avx512F.IsSupported, "AVX512F not available on this machine.");
        var rnd = new Random(Seed);
        foreach (var shape in new[] { (12, 128, 100), (24, 64, 20) })
        {
            var a = FillRect(shape.Item1, shape.Item2, rnd);
            var b = FillRect(shape.Item2, shape.Item3, rnd);
            var p = Tensor<float>.Zeros(shape.Item2, shape.Item3).ToDenseTensor();
            var c = Tensor<float>.Zeros(shape.Item1, shape.Item3).ToDenseTensor();
            RunTwelve(shape.Item1, shape.Item2, shape.Item3, a, b, p, c);
            AssertNear12(Oracle12(a, b, shape.Item1, shape.Item2, shape.Item3), c.ToArray());
        }
    }

    [Fact]
    public void TransientPackedCoversOddRows()
    {
        // The transient per-call pack (M>=64 without persistent weights)
        // routes through the 12/6-row composer on AVX512 hardware and the
        // 2/3-row kernels elsewhere; odd row counts must cover every row
        // exactly once on either path, including the composer rem==1 peel
        // (M=73) and the exact-3-row branch (M=69).
        var rnd = new Random(Seed);
        foreach (var shape in new[] { (65, 64, 64), (69, 64, 64), (73, 64, 64), (77, 64, 96) })
        {
            var a = FillRect(shape.Item1, shape.Item2, rnd);
            var b = FillRect(shape.Item2, shape.Item3, rnd);
            var c = Tensor<float>.MatMul2D(a, b);
            AssertNear12(Oracle12(a, b, shape.Item1, shape.Item2, shape.Item3), c.ToArray());
        }
    }

    [SkippableFact]
    public unsafe void TwelveRowRejectsRemainder()
    {
        Skip.If(!Avx512F.IsSupported, "AVX512F not available on this machine.");
        var a = FillRect(7, 8, new Random(Seed));
        var b = FillRect(8, 32, new Random(Seed));
        var p = Tensor<float>.Zeros(8, 32).ToDenseTensor();
        var c = Tensor<float>.Zeros(7, 32).ToDenseTensor();
        Assert.Throws<System.ArgumentException>(() =>
        {
            using var pa = a.Buffer.Pin();
            using var pb = b.Buffer.Pin();
            using var pp = p.Buffer.Pin();
            using var pc = c.Buffer.Pin();
            MathOps.mm_unsafe_vectorized_avx512_12x32packed(7, 8, 32, (float*)pa.Pointer, (float*)pp.Pointer, (float*)pc.Pointer);
        });
    }

    static unsafe void RunEight(int m, int n, int k, DenseTensor<float> a, DenseTensor<float> b, DenseTensor<float> p, DenseTensor<float> c)
    {
        using var pa = a.Buffer.Pin();
        using var pb = b.Buffer.Pin();
        using var pp = p.Buffer.Pin();
        using var pc = c.Buffer.Pin();
        MathOps.PackPanelsB(n, k, (float*)pb.Pointer, (float*)pp.Pointer);
        MathOps.mm_unsafe_vectorized_avx512_8x32packed(m, n, k, (float*)pa.Pointer, (float*)pp.Pointer, (float*)pc.Pointer);
    }

    [SkippableFact]
    public unsafe void EightRowMatchesTwoRowBitwise()
    {
        Skip.If(!Avx512F.IsSupported, "AVX512F not available on this machine.");
        var rnd = new Random(Seed);
        foreach (var shape in new[] { (8, 64, 64), (16, 128, 96) })
        {
            var a = FillRect(shape.Item1, shape.Item2, rnd);
            var b = FillRect(shape.Item2, shape.Item3, rnd);
            var p = Tensor<float>.Zeros(shape.Item2, shape.Item3).ToDenseTensor();
            var c1 = Tensor<float>.Zeros(shape.Item1, shape.Item3).ToDenseTensor();
            var c2 = Tensor<float>.Zeros(shape.Item1, shape.Item3).ToDenseTensor();
            RunEight(shape.Item1, shape.Item2, shape.Item3, a, b, p, c1);
            RunTwo(shape.Item1, shape.Item2, shape.Item3, a, b, p, c2);
            Assert.True(c1.Buffer.Span.SequenceEqual(c2.Buffer.Span),
                "8-row diverges bitwise from 2-row on " + shape.Item1 + "x" + shape.Item2 + "x" + shape.Item3);
        }
    }

    [SkippableFact]
    public unsafe void EightRowTailMatchesOracle()
    {
        Skip.If(!Avx512F.IsSupported, "AVX512F not available on this machine.");
        var rnd = new Random(Seed);
        foreach (var shape in new[] { (8, 128, 100), (16, 64, 20) })
        {
            var a = FillRect(shape.Item1, shape.Item2, rnd);
            var b = FillRect(shape.Item2, shape.Item3, rnd);
            var p = Tensor<float>.Zeros(shape.Item2, shape.Item3).ToDenseTensor();
            var c = Tensor<float>.Zeros(shape.Item1, shape.Item3).ToDenseTensor();
            RunEight(shape.Item1, shape.Item2, shape.Item3, a, b, p, c);
            AssertNear(Oracle(a, b, shape.Item1, shape.Item2, shape.Item3), c.ToArray());
        }
    }

    [SkippableFact]
    public unsafe void EightRowRejectsRemainder()
    {
        Skip.If(!Avx512F.IsSupported, "AVX512F not available on this machine.");
        var a = FillRect(7, 8, new Random(Seed));
        var b = FillRect(8, 32, new Random(Seed));
        var p = Tensor<float>.Zeros(8, 32).ToDenseTensor();
        var c = Tensor<float>.Zeros(7, 32).ToDenseTensor();
        Assert.Throws<System.ArgumentException>(() =>
        {
            using var pa = a.Buffer.Pin();
            using var pb = b.Buffer.Pin();
            using var pp = p.Buffer.Pin();
            using var pc = c.Buffer.Pin();
            MathOps.mm_unsafe_vectorized_avx512_8x32packed(7, 8, 32, (float*)pa.Pointer, (float*)pp.Pointer, (float*)pc.Pointer);
        });
    }

    [SkippableFact]
    public unsafe void EightRowCompositionMatchesTwoRowBitwise()
    {
        // The dispatched composition (12-row heads, 8-row absorbed
        // remainders, 6-row head, 3/2-row tails) must agree bit-wise with
        // single 2-row calls: rows are independent and every piece shares
        // panel layout and FMA order. Covers the rem==4/rem==2 absorption
        // (M=16/28 and M=14/26) and the small-M 8-row head (M=8).
        Skip.If(!Avx512F.IsSupported, "AVX512F not available on this machine.");
        var rnd = new Random(Seed);
        foreach (var shape in new[] { (8, 64, 64), (14, 64, 64), (16, 128, 96), (21, 64, 64), (26, 64, 64), (28, 64, 96), (33, 64, 64), (40, 64, 64) })
        {
            var a = FillRect(shape.Item1, shape.Item2, rnd);
            var b = FillRect(shape.Item2, shape.Item3, rnd);
            var p = Tensor<float>.Zeros(shape.Item2, shape.Item3).ToDenseTensor();
            var c1 = Tensor<float>.Zeros(shape.Item1, shape.Item3).ToDenseTensor();
            var c2 = Tensor<float>.Zeros(shape.Item1, shape.Item3).ToDenseTensor();
            using var pa = a.Buffer.Pin();
            using var pb = b.Buffer.Pin();
            using var pp = p.Buffer.Pin();
            using var pc1 = c1.Buffer.Pin();
            using var pc2 = c2.Buffer.Pin();
            MathOps.PackPanelsB(shape.Item2, shape.Item3, (float*)pb.Pointer, (float*)pp.Pointer);
            ComposedRows8(shape.Item1, shape.Item2, shape.Item3, (float*)pa.Pointer, (float*)pp.Pointer, (float*)pc1.Pointer);
            if ((shape.Item1 & 1) == 0)
            {
                RunTwo(shape.Item1, shape.Item2, shape.Item3, a, b, p, c2);
                Assert.True(c1.Buffer.Span.SequenceEqual(c2.Buffer.Span),
                    "8-row composition diverges bitwise from 2-row on " + shape.Item1 + "x" + shape.Item2 + "x" + shape.Item3);
            }
            else
            {
                // The 2-row reference requires even rows; odd compositions
                // check against the independent double oracle instead.
                AssertNear(Oracle(a, b, shape.Item1, shape.Item2, shape.Item3), c1.ToArray());
            }
        }
    }

    static unsafe void ComposedRows8(int m, int n, int k, float* x, float* packed, float* dest)
    {
        // Mirrors the dispatched AVX512 decomposition including 8-row
        // remainder absorption.
        int rest = m;
        float* xr = x;
        float* dr = dest;
        int main = (rest / 12) * 12;
        int rem = rest - main;
        if (rem == 1) { main -= 12; rem = 13; }
        if (rem == 4 && main >= 12) { main -= 12; rem = 16; }
        else if (rem == 2 && main >= 12) { main -= 12; rem = 14; }
        if (main > 0)
        {
            MathOps.mm_unsafe_vectorized_avx512_12x32packed(main, n, k, xr, packed, dr);
            xr += main * n;
            dr += main * k;
        }
        rest = rem;
        // Never leave a single row: rem 9 drains to 6+3 through the peel
        // below instead of 8+1, which no kernel covers.
        while (rest >= 8 && rest != 9)
        {
            MathOps.mm_unsafe_vectorized_avx512_8x32packed(8, n, k, xr, packed, dr);
            xr += 8 * n;
            dr += 8 * k;
            rest -= 8;
        }
        if (rest >= 6 && rest != 7)
        {
            MathOps.mm_unsafe_vectorized_avx512_6x32packed(6, n, k, xr, packed, dr);
            xr += 6 * n;
            dr += 6 * k;
            rest -= 6;
        }
        if (rest == 0) return;
        if ((rest % 3) == 0)
        {
            MathOps.mm_unsafe_vectorized_intrinsics_3x4packed(rest, n, k, xr, packed, dr);
            return;
        }
        if ((rest & 1) == 0)
        {
            MathOps.mm_unsafe_vectorized_intrinsics_2x4packed(rest, n, k, xr, packed, dr);
            return;
        }
        MathOps.mm_unsafe_vectorized_intrinsics_3x4packed(3, n, k, xr, packed, dr);
        MathOps.mm_unsafe_vectorized_intrinsics_2x4packed(rest - 3, n, k, xr + 3 * n, packed, dr + 3 * k);
    }
}
