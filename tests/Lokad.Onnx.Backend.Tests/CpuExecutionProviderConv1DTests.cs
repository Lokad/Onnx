using System;
using System.Linq;
using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

// One-dimensional Conv/MaxPool coverage (PLAN.md Milestone 3). Rank-three
// inputs ride the proven two-dimensional kernels through shape
// normalization; expected values are frozen Python onnxruntime 1.29.0
// oracles (ORT_SEQUENTIAL, intra/inter-op 1, ORT_ENABLE_ALL, opset 17)
// from the ignored .agent/voice-probe/gen_conv1d.py. Double-precision
// agreement follows the ConvDirectTests pattern (ORT has no double Conv
// CPU kernel): on exactly-representable ints both precisions must agree
// bit for bit.
public class CpuExecutionProviderConv1DTests
{
    const double Tol = 1e-5;

    static float[] Range(float start, float step, int n)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = start + step * i;
        return v;
    }

    static DenseTensor<float> FT(float[] values, params int[] dims) =>
        new DenseTensor<float>(values, dims);

    static void AssertNear(float[] actual, float[] expected, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(actual[i] - expected[i]) < Tol, what + "[" + i + "] drifted: " + actual[i] + " vs " + expected[i] + ".");
    }

    [Fact]
    public void ConvPointwise_MatchesOrt()
    {
        var x = FT(Range(0.5f, 0.5f, 20), 1, 4, 5);
        var w = FT(Range(-2.75f, 0.25f, 24), 6, 4, 1);
        var b = FT(new float[] { -1.25f, -0.75f, -0.25f, 0.25f, 0.75f, 1.25f }, 6);
        var r = CPU.Conv(x, w, b, null, new[] { 1 }, 1, new[] { 1 }, new[] { 0, 0 }, new[] { 1 }, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs![0];
        Assert.Equal(new[] { 1, 6, 5 }, y.Dimensions.ToArray());
        AssertNear(y.ToArray(), new float[]
        {
            -38.5f, -43.25f, -48f, -52.75f, -57.5f, -21f, -23.75f, -26.5f, -29.25f, -32f,
            -3.5f, -4.25f, -5f, -5.75f, -6.5f, 14f, 15.25f, 16.5f, 17.75f, 19f,
            31.5f, 34.75f, 38f, 41.25f, 44.5f, 49f, 54.25f, 59.5f, 64.75f, 70f,
        }, "conv-pointwise");
    }

    [Fact]
    public void ConvDepthwiseGroups_MatchesOrt()
    {
        var x = FT(Range(0.25f, 0.25f, 32), 1, 4, 8);
        var w = FT(Range(-1.25f, 0.25f, 12), 4, 1, 3);
        var r = CPU.Conv(x, w, null, null, new[] { 1 }, 4, new[] { 3 }, new[] { 0, 0 }, new[] { 1 }, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs![0];
        Assert.Equal(new[] { 1, 4, 6 }, y.Dimensions.ToArray());
        AssertNear(y.ToArray(), new float[]
        {
            -1.375f, -2.125f, -2.875f, -3.625f, -4.375f, -5.125f,
            -1.75f, -1.9375f, -2.125f, -2.3125f, -2.5f, -2.6875f,
            6.875f, 7.25f, 7.625f, 8f, 8.375f, 8.75f,
            24.5f, 25.4375f, 26.375f, 27.3125f, 28.25f, 29.1875f,
        }, "conv-depthwise");
    }

    [Fact]
    public void ConvPaddedStridedDilated_MatchesOrt()
    {
        var x = FT(Range(0.5f, 0.5f, 18), 1, 2, 9);
        var w = FT(Range(-2f, 0.25f, 18), 3, 2, 3);
        var b = FT(new float[] { 1f, -1f, 0.5f }, 3);
        var r = CPU.Conv(x, w, b, null, new[] { 2 }, 1, new[] { 3 }, new[] { 1, 1 }, new[] { 2 }, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs![0];
        Assert.Equal(new[] { 1, 3, 4 }, y.Dimensions.ToArray());
        AssertNear(y.ToArray(), new float[]
        {
            -14.125f, -28f, -36.25f, -29.875f, 6.375f, 8.25f, 9f, 2.625f,
            30.375f, 48f, 57.75f, 38.625f,
        }, "conv-pad-stride-dil");
    }

    [Fact]
    public void ConvSameAutoPad_MatchesOrt()
    {
        var x = FT(Range(0.5f, 0.5f, 14), 1, 2, 7);
        var w = FT(Range(-1.25f, 0.25f, 12), 2, 2, 3);
        var r = CPU.Conv(x, w, null, "SAME_UPPER", new[] { 1 }, 1, new[] { 3 }, null, new[] { 2 }, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs![0];
        Assert.Equal(new[] { 1, 2, 4 }, y.Dimensions.ToArray());
        AssertNear(y.ToArray(), new float[] { -2.25f, -7.75f, -11.5f, -12.25f, 12.75f, 21.5f, 26.75f, 17.75f }, "conv-same");
    }

    [Fact]
    public void Conv1D_DoubleMatchesFloatOnExactInts()
    {
        var xf = FT(Range(1f, 1f, 8), 1, 2, 4);
        var wf = FT(Range(1f, 1f, 6), 3, 2, 1);
        var rf = CPU.Conv(xf, wf, null, null, new[] { 1 }, 1, new[] { 1 }, new[] { 0, 0 }, new[] { 1 }, null);
        Assert.Equal(OpStatus.Success, rf.Status);
        var xd = new DenseTensor<double>(Range(1f, 1f, 8).Select(v => (double)v).ToArray(), new[] { 1, 2, 4 });
        var wd = new DenseTensor<double>(Range(1f, 1f, 6).Select(v => (double)v).ToArray(), new[] { 3, 2, 1 });
        var rd = CPU.Conv(xd, wd, null, null, new[] { 1 }, 1, new[] { 1 }, new[] { 0, 0 }, new[] { 1 }, null);
        Assert.Equal(OpStatus.Success, rd.Status);
        Assert.Equal(((Tensor<float>)rf.Outputs![0]).ToArray(), ((Tensor<double>)rd.Outputs![0]).ToArray().Select(v => (float)v).ToArray());
    }

    [Fact]
    public void Conv1D_RankMismatch_Fails()
    {
        var x = FT(Range(0.5f, 0.5f, 20), 1, 4, 5);
        var w4 = FT(Range(0.25f, 0.25f, 24), 6, 4, 1, 1);
        var r = CPU.Conv(x, w4, null, null, new[] { 1 }, 1, new[] { 1 }, new[] { 0, 0 }, new[] { 1 }, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }

    [Fact]
    public void Conv1D_BadAttrRank_Fails()
    {
        var x = FT(Range(0.5f, 0.5f, 20), 1, 4, 5);
        var w = FT(Range(-2.75f, 0.25f, 24), 6, 4, 1);
        var r = CPU.Conv(x, w, null, null, new[] { 1 }, 1, new[] { 1 }, new[] { 0, 0, 0, 0 }, new[] { 1 }, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }

    [Fact]
    public void PoolBasic_MatchesOrt()
    {
        var x = FT(Range(0.5f, 0.5f, 18), 1, 2, 9);
        var r = CPU.MaxPool(x, null, 0, new[] { 1 }, new[] { 3 }, new[] { 0, 0 }, null, new[] { 3 }, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs![0];
        Assert.Equal(new[] { 1, 2, 3 }, y.Dimensions.ToArray());
        AssertNear(y.ToArray(), new float[] { 1.5f, 3f, 4.5f, 6f, 7.5f, 9f }, "pool-basic");
    }

    [Fact]
    public void PoolPaddedCeil_MatchesOrt()
    {
        var x = FT(Range(0.5f, 0.5f, 8), 1, 1, 8);
        var r = CPU.MaxPool(x, null, 1, new[] { 1 }, new[] { 3 }, new[] { 1, 1 }, null, new[] { 2 }, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs![0];
        Assert.Equal(new[] { 1, 1, 5 }, y.Dimensions.ToArray());
        AssertNear(y.ToArray(), new float[] { 1f, 2f, 3f, 4f, 4f }, "pool-pad-ceil");
    }

    [Fact]
    public void PoolDilated_MatchesOrt()
    {
        var x = FT(Range(0.5f, 0.5f, 10), 1, 1, 10);
        var r = CPU.MaxPool(x, null, 0, new[] { 2 }, new[] { 2 }, new[] { 0, 0 }, null, new[] { 1 }, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs![0];
        Assert.Equal(new[] { 1, 1, 8 }, y.Dimensions.ToArray());
        AssertNear(y.ToArray(), new float[] { 1.5f, 2f, 2.5f, 3f, 3.5f, 4f, 4.5f, 5f }, "pool-dilated");
    }

    [Fact]
    public void Pool1D_DoubleMatchesFloatOnExactInts()
    {
        var xf = FT(new float[] { 3f, 1f, 4f, 1f, 5f, 9f }, 1, 1, 6);
        var rf = CPU.MaxPool(xf, null, 0, new[] { 1 }, new[] { 2 }, new[] { 0, 0 }, null, new[] { 2 }, null);
        Assert.Equal(OpStatus.Success, rf.Status);
        var xd = new DenseTensor<double>(new double[] { 3.0, 1.0, 4.0, 1.0, 5.0, 9.0 }, new[] { 1, 1, 6 });
        var rd = CPU.MaxPool(xd, null, 0, new[] { 1 }, new[] { 2 }, new[] { 0, 0 }, null, new[] { 2 }, null);
        Assert.Equal(OpStatus.Success, rd.Status);
        Assert.Equal(((Tensor<float>)rf.Outputs![0]).ToArray(), ((Tensor<double>)rd.Outputs![0]).ToArray().Select(v => (float)v).ToArray());
    }

    // Independent naive oracle: direct gather-multiply loops over the
    // rank-three layout, sharing no indexing formula with the
    // unsqueeze-to-2D adapter under test.
    static float[] NaiveConv1D(float[] x, int n, int c, int l, float[] w, int m, int cg, int k, float[]? b, int s, int d, int padL, int padR, bool relu, out int outL)
    {
        int effK = (k - 1) * d + 1;
        outL = (l + padL + padR - effK) / s + 1;
        int mPerGroup = m / NumGroups(m, c, cg);
        var y = new float[n * m * outL];
        for (int nn = 0; nn < n; nn++)
            for (int mm = 0; mm < m; mm++)
                for (int o = 0; o < outL; o++)
                {
                    float acc = b is null ? 0f : b[mm];
                    int g = mm / mPerGroup;
                    for (int cc = 0; cc < cg; cc++)
                        for (int kk = 0; kk < k; kk++)
                        {
                            int pos = o * s - padL + kk * d;
                            if (pos < 0 || pos >= l) continue;
                            acc += x[(nn * c + g * cg + cc) * l + pos] * w[(mm * cg + cc) * k + kk];
                        }
                    if (relu && acc < 0f) acc = 0f;
                    y[(nn * m + mm) * outL + o] = acc;
                }
        return y;
    }

    static int NumGroups(int m, int c, int cg) => c / cg;

    static void AgreeNaive(string what, int n, int c, int l, int m, int cg, int k, int s, int d, int padL, int padR, bool bias, bool relu, int group)
    {
        var rnd = new Random(1234 + n * 100000 + m * 1000 + l);
        var xv = new float[n * c * l];
        var wv = new float[m * cg * k];
        for (int i = 0; i < xv.Length; i++) xv[i] = (float)(rnd.NextDouble() * 2 - 1);
        for (int i = 0; i < wv.Length; i++) wv[i] = (float)(rnd.NextDouble() * 2 - 1);
        float[]? bv = bias ? new float[m] : null;
        if (bv is not null) for (int i = 0; i < m; i++) bv[i] = (float)(rnd.NextDouble() * 2 - 1);
        var x = FT(xv, n, c, l);
        var w = FT(wv, m, cg, k);
        Tensor<float>? b = bv is null ? null : FT(bv, m);
        var r = CPU.Conv(x, w, b, null, new[] { d }, group, new[] { k }, new[] { padL, padR }, new[] { s }, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs![0];
        var expected = NaiveConv1D(xv, n, c, l, wv, m, cg, k, bv, s, d, padL, padR, false, out int outL);
        Assert.Equal(new[] { n, m, outL }, y.Dimensions.ToArray());
        AssertNear(y.ToArray(), expected, what);
        if (relu)
        {
            var rr = CPU.Conv(x, w, b, null, new[] { d }, group, new[] { k }, new[] { padL, padR }, new[] { s }, null, true);
            Assert.Equal(OpStatus.Success, rr.Status);
            var ry = (Tensor<float>)rr.Outputs![0];
            var rexp = NaiveConv1D(xv, n, c, l, wv, m, cg, k, bv, s, d, padL, padR, true, out _);
            AssertNear(ry.ToArray(), rexp, what + "-relu");
        }
    }

    [Fact]
    public void ConvNaive_PointwiseMatches() => AgreeNaive("naive-pointwise", 1, 4, 5, 6, 4, 1, 1, 1, 0, 0, true, false, 1);

    [Fact]
    public void ConvNaive_SegLikeMatches() => AgreeNaive("naive-seglike", 1, 4, 32, 3, 4, 5, 1, 1, 0, 0, true, false, 1);

    [Fact]
    public void ConvNaive_StridedMatches() => AgreeNaive("naive-strided", 1, 2, 16, 3, 2, 3, 2, 1, 0, 0, true, true, 1);

    [Fact]
    public void ConvNaive_AsymmetricPadsMatches() => AgreeNaive("naive-asympad", 1, 2, 9, 2, 2, 3, 1, 1, 2, 0, false, false, 1);

    [Fact]
    public void ConvNaive_DilatedMatches() => AgreeNaive("naive-dilated", 1, 2, 12, 2, 2, 3, 1, 2, 1, 1, true, false, 1);

    [Fact]
    public void ConvNaive_GroupedMatches() => AgreeNaive("naive-grouped", 1, 4, 8, 4, 2, 3, 1, 1, 0, 0, true, false, 2);

    [Fact]
    public void ConvNaive_BatchedMatches() => AgreeNaive("naive-batched", 2, 3, 10, 4, 3, 3, 1, 1, 1, 1, true, true, 1);

    [Fact]
    public void ConvNaive_AutoPadModesMatchSpec()
    {
        // Spec-literal pads: SAME_UPPER splits extra to the end,
        // SAME_LOWER to the begin, VALID pads nothing.
        var x = FT(Range(0.5f, 0.5f, 14), 1, 2, 7);
        var w = FT(Range(-1.25f, 0.25f, 12), 2, 2, 3);
        foreach (string mode in new[] { "SAME_UPPER", "SAME_LOWER", "VALID" })
        {
            var r = CPU.Conv(x, w, null, mode, new[] { 1 }, 1, new[] { 3 }, null, new[] { 2 }, null);
            Assert.Equal(OpStatus.Success, r.Status);
            var y = (Tensor<float>)r.Outputs![0];
            int effK = 3, s = 2, l = 7;
            int outL, padL, padR;
            if (mode == "VALID") { padL = 0; padR = 0; outL = (l - effK) / s + 1; }
            else
            {
                outL = (l + s - 1) / s;
                int need = System.Math.Max(0, (outL - 1) * s + effK - l);
                if (mode == "SAME_UPPER") { padL = need / 2; padR = need - padL; }
                else { padR = need / 2; padL = need - padR; }
            }
            var expected = NaiveConv1D(x.ToArray(), 1, 2, 7, w.ToArray(), 2, 2, 3, null, s, 1, padL, padR, false, out int nl);
            Assert.Equal(outL, nl);
            Assert.Equal(new[] { 1, 2, outL }, y.Dimensions.ToArray());
            AssertNear(y.ToArray(), expected, "conv-autopad-" + mode);
        }
    }

    [Fact]
    public void ConvNaive_DoubleBitMatchesFloatOnInts()
    {
        var xf = FT(new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }, 1, 2, 4);
        var wf = FT(new float[] { 1f, 0f, -1f, 2f }, 2, 2, 1);
        var rf = CPU.Conv(xf, wf, null, null, new[] { 1 }, 1, new[] { 1 }, new[] { 1, 0 }, new[] { 1 }, null);
        Assert.Equal(OpStatus.Success, rf.Status);
        var xd = new DenseTensor<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8 }, new[] { 1, 2, 4 });
        var wd = new DenseTensor<double>(new double[] { 1, 0, -1, 2 }, new[] { 2, 2, 1 });
        var rd = CPU.Conv(xd, wd, null, null, new[] { 1 }, 1, new[] { 1 }, new[] { 1, 0 }, new[] { 1 }, null);
        Assert.Equal(OpStatus.Success, rd.Status);
        Assert.Equal(((Tensor<float>)rf.Outputs![0]).ToArray(), ((Tensor<double>)rd.Outputs![0]).ToArray().Select(v => (float)v).ToArray());
    }

    [Fact]
    public void Conv1D_BadStrideRank_Fails()
    {
        var x = FT(Range(0.5f, 0.5f, 20), 1, 4, 5);
        var w = FT(Range(-2.75f, 0.25f, 24), 6, 4, 1);
        var r = CPU.Conv(x, w, null, null, new[] { 1 }, 1, new[] { 1 }, new[] { 0, 0 }, new[] { 1, 1 }, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }

    [Fact]
    public void Conv1D_BadDilationRank_Fails()
    {
        var x = FT(Range(0.5f, 0.5f, 20), 1, 4, 5);
        var w = FT(Range(-2.75f, 0.25f, 24), 6, 4, 1);
        var r = CPU.Conv(x, w, null, null, new[] { 1, 1 }, 1, new[] { 1 }, new[] { 0, 0 }, new[] { 1 }, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }

    [Fact]
    public void Pool1D_MissingKernel_Fails()
    {
        var x = FT(Range(0.5f, 0.5f, 18), 1, 2, 9);
        var r = CPU.MaxPool(x, null, 0, new[] { 1 }, null, new[] { 0, 0 }, null, new[] { 3 }, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }
}
