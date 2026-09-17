namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Covers the direct rank-four float single-channel path (B1 lane):
/// hand-exact small cases, naive-oracle agreement across the stem
/// geometry, strides, asymmetric pads, dilation, kernel-one, bias/ReLU
/// (both vector and scalar modes), dop determinism, generic fallback
/// past the M cap, and preserved validation failures.
/// </summary>
public class ConvSingleChannelTests
{
    static DenseTensor<float> FilledTensor(int[] dims, int seed)
    {
        var t = DenseTensor<float>.OfShape(dims);
        var span = t.Buffer.Span;
        for (int i = 0; i < span.Length; i++) span[i] = (((i * 37 + seed) % 97) - 48) * 0.02f;
        return t;
    }

    static double[] NaiveSc2D(float[] x, int n, int h, int w, float[] wt, int m, int kh, int kw, float[]? bias, int padT, int padL, int padB, int padR, int sH, int sW, int dH, int dW)
    {
        int effKH = (kh - 1) * dH + 1;
        int effKW = (kw - 1) * dW + 1;
        int outH = (h + padT + padB - effKH) / sH + 1;
        int outW = (w + padL + padR - effKW) / sW + 1;
        var y = new double[n * m * outH * outW];
        for (int nn = 0; nn < n; nn++)
            for (int mm = 0; mm < m; mm++)
                for (int oy = 0; oy < outH; oy++)
                    for (int ox = 0; ox < outW; ox++)
                    {
                        double acc = bias is null ? 0.0 : bias[mm];
                        for (int ky = 0; ky < kh; ky++)
                            for (int kx = 0; kx < kw; kx++)
                            {
                                int iy = oy * sH + ky * dH - padT;
                                int ix = ox * sW + kx * dW - padL;
                                double xv = (uint)iy < (uint)h && (uint)ix < (uint)w ? x[(nn * h + iy) * w + ix] : 0.0;
                                acc += (double)wt[(mm * kh + ky) * kw + kx] * xv;
                            }
                        y[((nn * m + mm) * outH + oy) * outW + ox] = acc;
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
            Assert.True(System.Math.Abs(actual[i] - e) <= tol, i.ToString());
        }
    }

    static float[] RunSc2D(DenseTensor<float> x, DenseTensor<float> w, DenseTensor<float>? b, int[] pads, int[] strides, int[] dilations, TensorExecutionOptions opts, bool relu)
    {
        var r = CPUExecutionProvider.Conv(x, w, b, null, dilations, 1, null, pads, strides, new ExecutionOptions(OptimizationMode.Speed, opts), relu);
        Assert.Equal(OpStatus.Success, r.Status);
        return ((Tensor<float>)r.Outputs[0]).ToArray();
    }
    [Fact]
    public void StemGeometry_MatchesNaive()
    {
        // Analogue of the embedding stem (32x1x3x3 over 80x200, pads 1):
        // the B1 pricing case, no patch matrix.
        var x = FilledTensor(new[] { 1, 1, 80, 200 }, 311);
        var w = FilledTensor(new[] { 32, 1, 3, 3 }, 313);
        var b = FilledTensor(new[] { 32 }, 317);
        var got = RunSc2D(x, w, b, new[] { 1, 1, 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 }, TensorExecutionOptions.Auto, true);
        var expected = NaiveSc2D(x.ToArray(), 1, 80, 200, w.ToArray(), 32, 3, 3, b.ToArray(), 1, 1, 1, 1, 1, 1, 1, 1);
        Assert.Equal(1 * 32 * 80 * 200, got.Length);
        AssertNear(expected, got, true);
    }

    [Fact]
    public void SmallHandComputed_MatchesNaive()
    {
        var x = DenseTensor<float>.OfValues(new float[1, 1, 3, 3]
        {
            { { { 1f, 2f, 3f }, { 4f, 5f, 6f }, { 7f, 8f, 9f } } },
        });
        var w = DenseTensor<float>.OfValues(new float[2, 1, 2, 2]
        {
            { { { 1f, 0f }, { 0f, -1f } } },
            { { { 0f, 1f }, { 1f, 0f } } },
        });
        var got = RunSc2D(x, w, null, new[] { 0, 0, 0, 0 }, new[] { 1, 1 }, new[] { 1, 1 }, TensorExecutionOptions.Scalar, false);
        var expected = NaiveSc2D(x.ToArray(), 1, 3, 3, w.ToArray(), 2, 2, 2, null, 0, 0, 0, 0, 1, 1, 1, 1);
        Assert.Equal(2 * 2 * 2, got.Length);
        AssertNear(expected, got, false);
    }

    [Fact]
    public void StrideTwo_MatchesNaive()
    {
        var x = FilledTensor(new[] { 1, 1, 16, 16 }, 321);
        var w = FilledTensor(new[] { 8, 1, 3, 3 }, 323);
        var b = FilledTensor(new[] { 8 }, 329);
        var got = RunSc2D(x, w, b, new[] { 1, 1, 1, 1 }, new[] { 2, 2 }, new[] { 1, 1 }, TensorExecutionOptions.Auto, false);
        var expected = NaiveSc2D(x.ToArray(), 1, 16, 16, w.ToArray(), 8, 3, 3, b.ToArray(), 1, 1, 1, 1, 2, 2, 1, 1);
        Assert.Equal(1 * 8 * 8 * 8, got.Length);
        AssertNear(expected, got, false);
    }

    [Fact]
    public void AsymmetricPadsDilation_MatchesNaive()
    {
        var x = FilledTensor(new[] { 1, 1, 14, 12 }, 331);
        var w = FilledTensor(new[] { 4, 1, 3, 3 }, 337);
        var got = RunSc2D(x, w, null, new[] { 0, 2, 1, 0 }, new[] { 1, 1 }, new[] { 2, 2 }, TensorExecutionOptions.Auto, true);
        var expected = NaiveSc2D(x.ToArray(), 1, 14, 12, w.ToArray(), 4, 3, 3, null, 0, 2, 1, 0, 1, 1, 2, 2);
        Assert.Equal(1 * 4 * 11 * 10, got.Length);
        AssertNear(expected, got, true);
    }
    [Fact]
    public void KernelOneSingleOutput_MatchesNaive()
    {
        var x = FilledTensor(new[] { 2, 1, 8, 9 }, 341);
        var w = FilledTensor(new[] { 1, 1, 1, 1 }, 343);
        var b = FilledTensor(new[] { 1 }, 347);
        var got = RunSc2D(x, w, b, new[] { 0, 0, 0, 0 }, new[] { 1, 1 }, new[] { 1, 1 }, TensorExecutionOptions.Auto, false);
        var expected = NaiveSc2D(x.ToArray(), 2, 8, 9, w.ToArray(), 1, 1, 1, b.ToArray(), 0, 0, 0, 0, 1, 1, 1, 1);
        Assert.Equal(2 * 1 * 8 * 9, got.Length);
        AssertNear(expected, got, false);
    }

    [Fact]
    public void ScalarMode_MatchesNaive()
    {
        var x = FilledTensor(new[] { 1, 1, 9, 11 }, 349);
        var w = FilledTensor(new[] { 4, 1, 3, 3 }, 353);
        var b = FilledTensor(new[] { 4 }, 359);
        var got = RunSc2D(x, w, b, new[] { 1, 1, 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 }, TensorExecutionOptions.Scalar, true);
        var expected = NaiveSc2D(x.ToArray(), 1, 9, 11, w.ToArray(), 4, 3, 3, b.ToArray(), 1, 1, 1, 1, 1, 1, 1, 1);
        AssertNear(expected, got, true);
    }

    [Fact]
    public void ParallelBranch_Deterministic()
    {
        var x = FilledTensor(new[] { 2, 1, 12, 12 }, 367);
        var w = FilledTensor(new[] { 8, 1, 3, 3 }, 371);
        var b = FilledTensor(new[] { 8 }, 373);
        var seq = RunSc2D(x, w, b, new[] { 1, 1, 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 }, TensorExecutionOptions.Auto, true);
        var par = RunSc2D(x, w, b, new[] { 1, 1, 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 }, TensorExecutionOptions.Parallel(4), true);
        Assert.Equal(seq, par);
    }

    [Fact]
    public void WideM_FallsBackWithValues()
    {
        // Past the M cap the generic patch path serves the shape; values
        // must still agree with the naive oracle.
        var x = FilledTensor(new[] { 1, 1, 6, 6 }, 379);
        var w = FilledTensor(new[] { 65, 1, 3, 3 }, 383);
        var got = RunSc2D(x, w, null, new[] { 1, 1, 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 }, TensorExecutionOptions.Auto, false);
        var expected = NaiveSc2D(x.ToArray(), 1, 6, 6, w.ToArray(), 65, 3, 3, null, 1, 1, 1, 1, 1, 1, 1, 1);
        Assert.Equal(1 * 65 * 6 * 6, got.Length);
        AssertNear(expected, got, false);
    }

    [Fact]
    public void BadBias_StillFailsCleanly()
    {
        var x = FilledTensor(new[] { 1, 1, 8, 8 }, 389);
        var w = FilledTensor(new[] { 4, 1, 3, 3 }, 397);
        var b = FilledTensor(new[] { 3 }, 401);
        Assert.Throws<System.ArgumentException>(() => CPUExecutionProvider.Conv(x, w, b, null, new[] { 1, 1 }, 1, null, new[] { 1, 1, 1, 1 }, new[] { 1, 1 }, null));
    }
    [Fact]
    public void DirectLane_RentsNoScratch()
    {
        // The direct lane allocates no patch matrix, so a C=1 shape
        // inside the cap reports zero scratch where the patch path
        // reported 36 floats for the same geometry.
        var x = FilledTensor(new[] { 1, 1, 4, 4 }, 409);
        var w = FilledTensor(new[] { 1, 1, 2, 2 }, 419);
        var acc = new ScratchAccountant();
        var opts = ExecutionOptions.Default with { Tensor = TensorExecutionOptions.Scalar with { ScratchReporter = acc } };
        var r = CPUExecutionProvider.Conv(x, w, null, null, null, 1, null, null, null, opts, false);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(0, acc.TotalScratchBytes);
        Assert.Equal(9, ((Tensor<float>)r.Outputs[0]).ToArray().Length);
    }
}
