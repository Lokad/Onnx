using System.Collections.Generic;

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

    [Fact]
    public void ProbeShape_Agrees()
    {
        Case(32, 80, 200, 32, true, false);
    }

    [Fact]
    public void EmbeddingShapes_Agree()
    {
        Case(64, 48, 96, 64, true, false);
        Case(128, 24, 40, 128, true, false);
        Case(256, 12, 20, 256, true, false);
    }

    [Fact]
    public void OddWidth_RemainderAgrees()
    {
        Case(32, 80, 201, 32, true, false);
        Case(32, 7, 13, 32, true, false);
    }

    [Fact]
    public void BiasAndRelu_Agree()
    {
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
}

