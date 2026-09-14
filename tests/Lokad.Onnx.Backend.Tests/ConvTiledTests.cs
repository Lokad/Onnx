namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Covers bounded column tiling for large float convolutions: tiled shapes
/// agree with an independent naive oracle, stay deterministic, keep scratch
/// bounded, and run the batch-parallel branch. All shapes below exceed the
/// 256 KiB single-pass patch budget, so they exercise the tiled runner.
/// </summary>
public class ConvTiledTests
{
    static float Pattern(int i) => ((i * 37 + 11) % 97 - 48) * 0.03f;

    static DenseTensor<float> FilledTensor(int[] dims)
    {
        var t = DenseTensor<float>.OfShape(dims);
        var span = t.Buffer.Span;
        for (int i = 0; i < span.Length; i++) span[i] = Pattern(i);
        return t;
    }

    static double[] NaiveConv(float[] x, int n, int c, int h, int w, float[] wt, int m, int kh, int kw, float[]? bias, int pad, int sH, int sW, int group)
    {
        int outH = (h + 2 * pad - kh) / sH + 1;
        int outW = (w + 2 * pad - kw) / sW + 1;
        int cg = c / group;
        int mg = m / group;
        var y = new double[n * m * outH * outW];
        for (int nn = 0; nn < n; nn++)
            for (int mm = 0; mm < m; mm++)
            {
                int g = mm / mg;
                for (int oy = 0; oy < outH; oy++)
                    for (int ox = 0; ox < outW; ox++)
                    {
                        double acc = bias is null ? 0.0 : bias[mm];
                        for (int cc = 0; cc < cg; cc++)
                            for (int ky = 0; ky < kh; ky++)
                            {
                                int iy = oy * sH - pad + ky;
                                if ((uint)iy >= (uint)h) continue;
                                for (int kx = 0; kx < kw; kx++)
                                {
                                    int ix = ox * sW - pad + kx;
                                    if ((uint)ix >= (uint)w) continue;
                                    double xv = x[((nn * c + g * cg + cc) * h + iy) * w + ix];
                                    double wv = wt[((mm * cg + cc) * kh + ky) * kw + kx];
                                    acc += xv * wv;
                                }
                            }
                        y[((nn * m + mm) * outH + oy) * outW + ox] = acc;
                    }
            }
        return y;
    }

    static void AssertNear(double[] expected, float[] actual, bool relu)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            double e = relu && expected[i] < 0.0 ? 0.0 : expected[i];
            double tol = 1e-4 * (1.0 + System.Math.Abs(e));
            Assert.True(System.Math.Abs(actual[i] - e) <= tol, "index " + i + ": actual " + actual[i] + " expected " + e);
        }
    }

    [Fact]
    public void Tiled3x3_WithBiasAndRelu_MatchesNaive()
    {
        var x = FilledTensor(new[] { 1, 8, 70, 70 });
        var w = FilledTensor(new[] { 8, 8, 3, 3 });
        var b = FilledTensor(new[] { 8 });
        var pads = new int[] { 1, 1, 1, 1 };
        var y = Tensor<float>.Conv2D(x, w, 1, pads, b, null, new int[] { 1, 1 }, null, TensorExecutionOptions.Auto, true);
        Assert.Equal(new int[] { 1, 8, 70, 70 }, y.Dimensions.ToArray());
        var expected = NaiveConv(x.Buffer.ToArray(), 1, 8, 70, 70, w.Buffer.ToArray(), 8, 3, 3, b.Buffer.ToArray(), 1, 1, 1, 1);
        AssertNear(expected, y.ToArray(), true);
    }

    [Fact]
    public void TiledStride2_GroupedNoBias_MatchesNaive()
    {
        var x = FilledTensor(new[] { 1, 8, 65, 65 });
        var w = FilledTensor(new[] { 8, 4, 3, 3 });
        var pads = new int[] { 1, 1, 1, 1 };
        var y = Tensor<float>.Conv2D(x, w, 2, pads, null, null, new int[] { 2, 2 }, null, TensorExecutionOptions.Auto, false);
        Assert.Equal(new int[] { 1, 8, 33, 33 }, y.Dimensions.ToArray());
        var expected = NaiveConv(x.Buffer.ToArray(), 1, 8, 65, 65, w.Buffer.ToArray(), 8, 3, 3, null, 1, 2, 2, 2);
        AssertNear(expected, y.ToArray(), false);
    }
    [Fact]
    public void TiledResult_IsDeterministic()
    {
        var x = FilledTensor(new[] { 1, 8, 70, 70 });
        var w = FilledTensor(new[] { 8, 8, 3, 3 });
        var b = FilledTensor(new[] { 8 });
        var pads = new int[] { 1, 1, 1, 1 };
        var first = Tensor<float>.Conv2D(x, w, 1, pads, b, null, new int[] { 1, 1 }, null, TensorExecutionOptions.Auto, true);
        var second = Tensor<float>.Conv2D(x, w, 1, pads, b, null, new int[] { 1, 1 }, null, TensorExecutionOptions.Auto, true);
        Assert.Equal(first.ToArray(), second.ToArray());
    }

    [Fact]
    public void TiledScratch_StaysBounded()
    {
        // The 8x8 3x3 at 70x70 shape needs a 1.4 MiB full patch; tiling must
        // hold one column block plus its output tile near the 256 KiB budget.
        var x = FilledTensor(new[] { 1, 8, 70, 70 });
        var w = FilledTensor(new[] { 8, 8, 3, 3 });
        var b = FilledTensor(new[] { 8 });
        var acc = new ScratchAccountant();
        var opts = TensorExecutionOptions.Auto with { ScratchReporter = acc };
        var y = Tensor<float>.Conv2D(x, w, 1, new int[] { 1, 1, 1, 1 }, b, null, new int[] { 1, 1 }, null, opts, true);
        Assert.Equal(1 * 8 * 70 * 70, y.ToArray().Length);
        Assert.True(acc.TotalScratchBytes <= 512 * 1024, "scratch " + acc.TotalScratchBytes);
    }

    [Fact]
    public void TiledBatched_ParallelBranch_MatchesNaive()
    {
        var x = FilledTensor(new[] { 2, 8, 70, 70 });
        var w = FilledTensor(new[] { 8, 8, 3, 3 });
        var b = FilledTensor(new[] { 8 });
        var pads = new int[] { 1, 1, 1, 1 };
        var y = Tensor<float>.Conv2D(x, w, 1, pads, b, null, new int[] { 1, 1 }, null, TensorExecutionOptions.Parallel(2), true);
        Assert.Equal(new int[] { 2, 8, 70, 70 }, y.Dimensions.ToArray());
        var expected = NaiveConv(x.Buffer.ToArray(), 2, 8, 70, 70, w.Buffer.ToArray(), 8, 3, 3, b.Buffer.ToArray(), 1, 1, 1, 1);
        AssertNear(expected, y.ToArray(), true);
    }
    [Fact]
    public void TiledWidth_AlignsToKernelPanels()
    {
        // C=32, K=288 fits 204 columns in budget; the runner must choose 192
        // (6 kernel panels) so full tiles avoid remainders entirely.
        var x = FilledTensor(new[] { 1, 32, 16, 16 });
        var w = FilledTensor(new[] { 32, 32, 3, 3 });
        var b = FilledTensor(new[] { 32 });
        var acc = new ScratchAccountant();
        var opts = TensorExecutionOptions.Auto with { ScratchReporter = acc };
        var y = Tensor<float>.Conv2D(x, w, 1, new int[] { 1, 1, 1, 1 }, b, null, new int[] { 1, 1 }, null, opts, true);
        Assert.Equal(new int[] { 1, 32, 16, 16 }, y.Dimensions.ToArray());
        Assert.Equal((288 + 32) * 192 * 4L, acc.TotalScratchBytes);
        var expected = NaiveConv(x.Buffer.ToArray(), 1, 32, 16, 16, w.Buffer.ToArray(), 32, 3, 3, b.Buffer.ToArray(), 1, 1, 1, 1);
        AssertNear(expected, y.ToArray(), true);
    }

    [Fact]
    public void TiledWidth_NarrowFitTakesOnePanel()
    {
        // C=256, K=2304 fits 25 columns; the runner takes one 32-column
        // panel (reported exactly) rather than clamping to a wider minimum.
        var x = FilledTensor(new[] { 1, 256, 10, 10 });
        var w = FilledTensor(new[] { 4, 256, 3, 3 });
        var b = FilledTensor(new[] { 4 });
        var acc = new ScratchAccountant();
        var opts = TensorExecutionOptions.Auto with { ScratchReporter = acc };
        var pads = new int[] { 1, 1, 1, 1 };
        var y = Tensor<float>.Conv2D(x, w, 1, pads, b, null, new int[] { 1, 1 }, null, opts, true);
        Assert.Equal(new int[] { 1, 4, 10, 10 }, y.Dimensions.ToArray());
        Assert.Equal((2304 + 4) * 32 * 4L, acc.TotalScratchBytes);
        var expected = NaiveConv(x.Buffer.ToArray(), 1, 256, 10, 10, w.Buffer.ToArray(), 4, 3, 3, b.Buffer.ToArray(), 1, 1, 1, 1);
        AssertNear(expected, y.ToArray(), true);
    }
}
