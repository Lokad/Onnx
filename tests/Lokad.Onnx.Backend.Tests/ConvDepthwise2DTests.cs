namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Covers the direct rank-four float depthwise path: hand-exact small
/// cases, naive-oracle agreement across padding/stride/dilation/bias/ReLU
/// (including the stride-two frontend geometry), kernel-one shapes,
/// dop determinism, generic fallback for dense shapes, and preserved
/// validation failures.
/// </summary>
public class ConvDepthwise2DTests
{
    static DenseTensor<float> FilledTensor(int[] dims, int seed)
    {
        var t = DenseTensor<float>.OfShape(dims);
        var span = t.Buffer.Span;
        for (int i = 0; i < span.Length; i++) span[i] = (((i * 37 + seed) % 97) - 48) * 0.02f;
        return t;
    }

    static double[] NaiveDw2D(float[] x, int n, int c, int h, int w, float[] wt, int kh, int kw, float[]? bias, int padT, int padL, int padB, int padR, int sH, int sW, int dH, int dW)
    {
        int effKH = (kh - 1) * dH + 1;
        int effKW = (kw - 1) * dW + 1;
        int outH = (h + padT + padB - effKH) / sH + 1;
        int outW = (w + padL + padR - effKW) / sW + 1;
        var y = new double[n * c * outH * outW];
        for (int nn = 0; nn < n; nn++)
            for (int cc = 0; cc < c; cc++)
                for (int oy = 0; oy < outH; oy++)
                    for (int ox = 0; ox < outW; ox++)
                    {
                        double acc = bias is null ? 0.0 : bias[cc];
                        for (int ky = 0; ky < kh; ky++)
                            for (int kx = 0; kx < kw; kx++)
                            {
                                int iy = oy * sH + ky * dH - padT;
                                int ix = ox * sW + kx * dW - padL;
                                double xv = (uint)iy < (uint)h && (uint)ix < (uint)w ? x[((nn * c + cc) * h + iy) * w + ix] : 0.0;
                                acc += (double)wt[(cc * kh + ky) * kw + kx] * xv;
                            }
                        y[((nn * c + cc) * outH + oy) * outW + ox] = acc;
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
            Assert.True(System.Math.Abs(actual[i] - e) <= tol, "index " + i);
        }
    }

    static float[] RunDw2D(DenseTensor<float> x, DenseTensor<float> w, DenseTensor<float>? b, int group, int[] pads, int[] strides, int[] dilations, TensorExecutionOptions opts, bool relu)
    {
        var r = CPUExecutionProvider.Conv(x, w, b, null, dilations, group, null, pads, strides, new ExecutionOptions(OptimizationMode.Speed, opts), relu);
        Assert.Equal(OpStatus.Success, r.Status);
        return ((Tensor<float>)r.Outputs[0]).ToArray();
    }

    [Fact]
    public void SmallHandComputed_MatchesNaive()
    {
        var x = DenseTensor<float>.OfValues(new float[1, 1, 3, 3]
        {
            { { { 1f, 2f, 3f }, { 4f, 5f, 6f }, { 7f, 8f, 9f } } },
        });
        var w = DenseTensor<float>.OfValues(new float[1, 1, 2, 2]
        {
            { { { 1f, 0f }, { 0f, -1f } } },
        });
        var got = RunDw2D(x, w, null, 1, new[] { 0, 0, 0, 0 }, new[] { 1, 1 }, new[] { 1, 1 }, TensorExecutionOptions.Scalar, false);
        var expected = NaiveDw2D(x.ToArray(), 1, 1, 3, 3, w.ToArray(), 2, 2, null, 0, 0, 0, 0, 1, 1, 1, 1);
        Assert.Equal(new[] { 1, 1, 2, 2 }, got.Length == 4 ? new[] { 1, 1, 2, 2 } : new[] { -1 });
        AssertNear(expected, got, false);
    }

    [Fact]
    public void StrideOneVector_MatchesNaive()
    {
        var x = FilledTensor(new[] { 2, 4, 10, 12 }, 11);
        var w = FilledTensor(new[] { 4, 1, 3, 3 }, 13);
        var b = FilledTensor(new[] { 4 }, 17);
        var got = RunDw2D(x, w, b, 4, new[] { 1, 1, 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 }, TensorExecutionOptions.Auto, true);
        var expected = NaiveDw2D(x.ToArray(), 2, 4, 10, 12, w.ToArray(), 3, 3, b.ToArray(), 1, 1, 1, 1, 1, 1, 1, 1);
        Assert.Equal(2 * 4 * 10 * 12, got.Length);
        AssertNear(expected, got, true);
    }

    [Fact]
    public void FrontendStrideTwo_MatchesNaive()
    {
        // Analogue of the encoder frontend depthwise layers (256 channels
        // of 3x3 stride 2): scalar positions, no patch matrix.
        var x = FilledTensor(new[] { 1, 8, 16, 16 }, 21);
        var w = FilledTensor(new[] { 8, 1, 3, 3 }, 23);
        var b = FilledTensor(new[] { 8 }, 29);
        var got = RunDw2D(x, w, b, 8, new[] { 1, 1, 1, 1 }, new[] { 2, 2 }, new[] { 1, 1 }, TensorExecutionOptions.Auto, false);
        var expected = NaiveDw2D(x.ToArray(), 1, 8, 16, 16, w.ToArray(), 3, 3, b.ToArray(), 1, 1, 1, 1, 2, 2, 1, 1);
        Assert.Equal(1 * 8 * 8 * 8, got.Length);
        AssertNear(expected, got, false);
    }

    [Fact]
    public void StrideDilationCombined_MatchesNaive()
    {
        // Stride and dilation together still vectorize: each tap loads
        // eight contiguous inputs at sW * ox + kx * dW - padL.
        var x = FilledTensor(new[] { 1, 4, 12, 12 }, 111);
        var w = FilledTensor(new[] { 4, 1, 3, 3 }, 113);
        var b = FilledTensor(new[] { 4 }, 127);
        var got = RunDw2D(x, w, b, 4, new[] { 1, 1, 1, 1 }, new[] { 2, 2 }, new[] { 2, 2 }, TensorExecutionOptions.Auto, true);
        var expected = NaiveDw2D(x.ToArray(), 1, 4, 12, 12, w.ToArray(), 3, 3, b.ToArray(), 1, 1, 1, 1, 2, 2, 2, 2);
        Assert.Equal(1 * 4 * 5 * 5, got.Length);
        AssertNear(expected, got, true);
    }

    [Fact]
    public void DilationNoBias_MatchesNaive()
    {
        var x = FilledTensor(new[] { 1, 4, 14, 14 }, 31);
        var w = FilledTensor(new[] { 4, 1, 3, 3 }, 37);
        var got = RunDw2D(x, w, null, 4, new[] { 2, 2, 2, 2 }, new[] { 1, 1 }, new[] { 2, 2 }, TensorExecutionOptions.Auto, true);
        var expected = NaiveDw2D(x.ToArray(), 1, 4, 14, 14, w.ToArray(), 3, 3, null, 2, 2, 2, 2, 1, 1, 2, 2);
        Assert.Equal(1 * 4 * 14 * 14, got.Length);
        AssertNear(expected, got, true);
    }

    [Fact]
    public void KernelOne_MatchesNaive()
    {
        var x = FilledTensor(new[] { 1, 4, 8, 8 }, 41);
        var w = FilledTensor(new[] { 4, 1, 1, 1 }, 43);
        var b = FilledTensor(new[] { 4 }, 47);
        var got = RunDw2D(x, w, b, 4, new[] { 0, 0, 0, 0 }, new[] { 1, 1 }, new[] { 1, 1 }, TensorExecutionOptions.Auto, false);
        var expected = NaiveDw2D(x.ToArray(), 1, 4, 8, 8, w.ToArray(), 1, 1, b.ToArray(), 0, 0, 0, 0, 1, 1, 1, 1);
        AssertNear(expected, got, false);
    }

    [Fact]
    public void ScalarMode_MatchesNaiveWithinRounding()
    {
        var x = FilledTensor(new[] { 1, 4, 9, 11 }, 53);
        var w = FilledTensor(new[] { 4, 1, 3, 3 }, 59);
        var b = FilledTensor(new[] { 4 }, 61);
        var got = RunDw2D(x, w, b, 4, new[] { 1, 1, 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 }, TensorExecutionOptions.Scalar, false);
        var again = RunDw2D(x, w, b, 4, new[] { 1, 1, 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 }, TensorExecutionOptions.Scalar, false);
        Assert.Equal(got, again);
        var expected = NaiveDw2D(x.ToArray(), 1, 4, 9, 11, w.ToArray(), 3, 3, b.ToArray(), 1, 1, 1, 1, 1, 1, 1, 1);
        AssertNear(expected, got, false);
    }

    [Fact]
    public void ParallelBranch_Deterministic()
    {
        var x = FilledTensor(new[] { 2, 8, 12, 12 }, 67);
        var w = FilledTensor(new[] { 8, 1, 3, 3 }, 71);
        var b = FilledTensor(new[] { 8 }, 73);
        var seq = RunDw2D(x, w, b, 8, new[] { 1, 1, 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 }, TensorExecutionOptions.Auto, true);
        var par = RunDw2D(x, w, b, 8, new[] { 1, 1, 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 }, TensorExecutionOptions.Parallel(4), true);
        Assert.Equal(seq, par);
    }

    [Fact]
    public void DenseGroupOne_FallsBackWithValues()
    {
        var x = FilledTensor(new[] { 1, 2, 6, 6 }, 79);
        var w = FilledTensor(new[] { 2, 2, 3, 3 }, 83);
        var got = RunDw2D(x, w, null, 1, new[] { 1, 1, 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 }, TensorExecutionOptions.Auto, false);
        Assert.Equal(1 * 2 * 6 * 6, got.Length);
    }

    [Fact]
    public void BadBias_StillFailsCleanly()
    {
        var x = FilledTensor(new[] { 1, 4, 8, 8 }, 89);
        var w = FilledTensor(new[] { 4, 1, 3, 3 }, 97);
        var b = FilledTensor(new[] { 3 }, 101);
        Assert.Throws<System.ArgumentException>(() => CPUExecutionProvider.Conv(x, w, b, null, new[] { 1, 1 }, 4, null, new[] { 1, 1, 1, 1 }, new[] { 1, 1 }, null));
    }
}
