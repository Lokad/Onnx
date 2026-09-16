using System.Collections.Generic;
using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx.Backend.Tests;

// Covers the blocked 3x3 kernel (M5): agreement with production Conv2D at
// the reassociation gate plus the decline contract. Production paths are
// ORT-validated; the blocked reduction order differs, so this pins 1e-4
// scaled, never bit identity. Timing lives in the embedding benchmarks.
public class ConvBlockedTests
{
    static void FillPattern(float[] a, int mul, int mod)
    {
        for (int i = 0; i < a.Length; i++) a[i] = ((i * mul % mod) - mod / 2) / (float)(mod / 2);
    }

    static double WorstScaled(float[] actual, float[] expected)
    {
        double w = 0.0;
        for (int i = 0; i < actual.Length; i++)
        {
            double e = System.Math.Abs((double)actual[i] - expected[i]) / (1.0 + System.Math.Abs((double)expected[i]));
            if (e > w) w = e;
        }
        return w;
    }

    static void Case(int c, int h, int w, int m, bool bias, bool relu)
    {
        var rnd = new Random(3100 + c * 17 + h + w * 3 + m);
        var input = new float[c * h * w];
        var filter = new float[m * c * 9];
        var bv = new float[m];
        FillPattern(input, 7, 127);
        FillPattern(filter, 13, 89);
        for (int i = 0; i < bv.Length; i++) bv[i] = (rnd.NextSingle() - 0.5f) * 0.2f;
        var opts = TensorExecutionOptions.Auto with { MaxDegreeOfParallelism = 1 };
        var x = new DenseTensor<float>(new Memory<float>(input), new[] { 1, c, h, w });
        var f = new DenseTensor<float>(new Memory<float>(filter), new[] { m, c, 3, 3 });
        Tensor<float>? b = bias ? new DenseTensor<float>(new Memory<float>(bv), new[] { m }) : null;
        var expected = ((Tensor<float>)Tensor<float>.Conv2D(x, f, 1, new[] { 1, 1, 1, 1 }, b, new[] { 3, 3 }, new[] { 1, 1 }, new[] { 1, 1 }, opts, relu)).ToArray();
        var pad = new MathOps.PadInfo { top = 1, left = 1, right = 1, bottom = 1, h = 2, w = 2 };
        Assert.True(MathOpsConvBlocked.TryConvBlocked3x3S1P1(x, f, b, 1, c, h, w, m, pad, opts, relu, out var got), "kernel declined c=" + c + " h=" + h + " w=" + w + " m=" + m);
        Assert.NotNull(got);
        var actual = ((Tensor<float>)got).ToArray();
        Assert.Equal(expected.Length, actual.Length);
        double worst = WorstScaled(actual, expected);
        Assert.True(worst <= 1e-4, "c=" + c + " h=" + h + " w=" + w + " m=" + m + " bias=" + bias + " relu=" + relu + " worst=" + worst.ToString("E2"));
    }

    [SkippableFact]
    public void ProbeShape_Agrees()
    {
        Skip.If(!Avx512F.IsSupported, "Blocked kernel needs AVX512F.");
        Case(32, 80, 200, 32, true, false);
    }

    [SkippableFact]
    public void EmbeddingShapes_Agree()
    {
        Skip.If(!Avx512F.IsSupported, "Blocked kernel needs AVX512F.");
        Case(64, 48, 96, 64, true, false);
        Case(128, 24, 40, 128, true, false);
        Case(256, 12, 20, 256, true, false);
    }

    [SkippableFact]
    public void OddWidth_RemainderAgrees()
    {
        Skip.If(!Avx512F.IsSupported, "Blocked kernel needs AVX512F.");
        Case(32, 80, 201, 32, true, false);
        Case(32, 7, 13, 32, true, false);
    }

    [SkippableFact]
    public void EdgeRowsAndRemainders_Agree()
    {
        Skip.If(!Avx512F.IsSupported, "Blocked kernel needs AVX512F.");
        Case(32, 3, 200, 32, true, false);
        Case(32, 80, 9, 32, true, false);
        Case(32, 80, 10, 32, true, false);
        Case(32, 80, 11, 32, true, false);
        Case(64, 4, 100, 64, true, false);
        Case(32, 80, 200, 64, true, true);
        Case(32, 5, 33, 32, false, false);
    }

    [SkippableFact]
    public void BiasAndRelu_Agree()
    {
        Skip.If(!Avx512F.IsSupported, "Blocked kernel needs AVX512F.");
        Case(32, 32, 32, 32, false, false);
        Case(32, 32, 32, 32, true, true);
        Case(32, 32, 32, 32, false, true);
    }

    [Fact]
    public void SmallChannel_Declines()
    {
        var opts = TensorExecutionOptions.Auto with { MaxDegreeOfParallelism = 1 };
        var x = DenseTensor<float>.OfShape(1, 8, 16, 16);
        var f = DenseTensor<float>.OfShape(8, 8, 3, 3);
        var pad = new MathOps.PadInfo { top = 1, left = 1, right = 1, bottom = 1, h = 2, w = 2 };
        Assert.False(MathOpsConvBlocked.TryConvBlocked3x3S1P1(x, f, null, 1, 8, 16, 16, 8, pad, opts, false, out var got));
        Assert.Null(got);
    }

    [Fact]
    public void WrongPads_Declines()
    {
        var opts = TensorExecutionOptions.Auto with { MaxDegreeOfParallelism = 1 };
        var x = DenseTensor<float>.OfShape(1, 16, 16, 16);
        var f = DenseTensor<float>.OfShape(16, 16, 3, 3);
        var pad = new MathOps.PadInfo { top = 0, left = 0, right = 0, bottom = 0, h = 0, w = 0 };
        Assert.False(MathOpsConvBlocked.TryConvBlocked3x3S1P1(x, f, null, 1, 16, 16, 16, 16, pad, opts, false, out var got));
        Assert.Null(got);
    }
    static void BlockedAddCase(int n)
    {
        var rnd = new Random(4100 + n);
        var a = new float[n];
        var b = new float[n];
        for (int i = 0; i < n; i++)
        {
            a[i] = (float)(rnd.NextDouble() * 20 - 10);
            b[i] = (float)(rnd.NextDouble() * 20 - 10);
        }
        if (n > 3)
        {
            a[0] = -0f; b[0] = 0f;
            a[1] = float.NaN; b[1] = 1f;
            a[2] = float.PositiveInfinity; b[2] = float.NegativeInfinity;
        }
        var dSimp = new float[n];
        var opts = TensorExecutionOptions.Intrinsics;
        MathOpsConvBlocked.BlockedAdd(a, b, dSimp, opts);
        var expected = new float[n];
        for (int i = 0; i < n; i++) expected[i] = a[i] + b[i];
        Assert.Equal(expected.Length, dSimp.Length);
        for (int i = 0; i < n; i++)
            Assert.Equal(BitConverter.SingleToInt32Bits(expected[i]), BitConverter.SingleToInt32Bits(dSimp[i]));
        var dScalar = new float[n];
        MathOpsConvBlocked.BlockedAdd(a, b, dScalar, TensorExecutionOptions.Scalar);
        for (int i = 0; i < n; i++)
            Assert.Equal(BitConverter.SingleToInt32Bits(dSimp[i]), BitConverter.SingleToInt32Bits(dScalar[i]));
        MathOpsConvBlocked.BlockedAdd(a, b, a, opts);
        for (int i = 0; i < n; i++)
            Assert.Equal(BitConverter.SingleToInt32Bits(expected[i]), BitConverter.SingleToInt32Bits(a[i]));
    }

    [Fact]
    public void BlockedAdd_Bitwise()
    {
        foreach (int n in new[] { 0, 1, 7, 8, 15, 16, 17, 31, 32, 33, 100, 1000 })
            BlockedAddCase(n);
    }

    static void BlockedReluCase(int n)
    {
        var rnd = new Random(4200 + n);
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)(rnd.NextDouble() * 20 - 10);
        if (n > 4)
        {
            data[0] = -0f;
            data[1] = 0f;
            data[2] = float.NaN;
            data[3] = float.NegativeInfinity;
            data[4] = float.PositiveInfinity;
        }
        var vec = (float[])data.Clone();
        MathOpsConvBlocked.BlockedRelu(vec, TensorExecutionOptions.Intrinsics);
        var prod = Tensor<float>.Relu(new DenseTensor<float>(new Memory<float>((float[])data.Clone()), new[] { n })).ToArray();
        Assert.Equal(prod.Length, vec.Length);
        for (int i = 0; i < n; i++)
            Assert.Equal(BitConverter.SingleToInt32Bits(prod[i]), BitConverter.SingleToInt32Bits(vec[i]));
        var sc = (float[])data.Clone();
        MathOpsConvBlocked.BlockedRelu(sc, TensorExecutionOptions.Scalar);
        for (int i = 0; i < n; i++)
            Assert.Equal(BitConverter.SingleToInt32Bits(vec[i]), BitConverter.SingleToInt32Bits(sc[i]));
    }

    [Fact]
    public void BlockedRelu_Bitwise()
    {
        foreach (int n in new[] { 0, 1, 7, 8, 15, 16, 17, 31, 32, 33, 100, 1000 })
            BlockedReluCase(n);
    }

    [Fact]
    public void BlockedAdd_Mismatched_Throws()
    {
        Assert.Throws<System.ArgumentException>(() =>
            MathOpsConvBlocked.BlockedAdd(new float[4], new float[5], new float[5], TensorExecutionOptions.Intrinsics));
    }

    [SkippableFact]
    public void PairedBlocks_Agree()
    {
        Skip.If(!Avx512F.IsSupported, "Blocked kernel needs AVX512F.");
        // Even block counts run the paired broadcast-sharing tiles on the
        // interior bulk; the odd count exercises pair plus single tail.
        Case(32, 80, 200, 64, true, false);
        Case(64, 48, 96, 128, true, false);
        Case(32, 16, 40, 48, true, true);
        Case(32, 16, 40, 16, false, false);
        Case(64, 5, 33, 64, true, false);
    }
}
