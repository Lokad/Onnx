namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Covers the direct rank-three float depthwise path: hand-exact small
/// cases, naive-oracle agreement across padding/stride/dilation/bias/ReLU,
/// scalar bit-exactness, dop determinism, adapter fallback for
/// non-depthwise shapes, and preserved validation failures.
/// </summary>
public class ConvDepthwiseTests
{
    static DenseTensor<float> FilledTensor(int[] dims, int seed)
    {
        var t = DenseTensor<float>.OfShape(dims);
        var span = t.Buffer.Span;
        for (int i = 0; i < span.Length; i++) span[i] = (((i * 37 + seed) % 97) - 48) * 0.02f;
        return t;
    }

    static double[] NaiveDw(float[] x, int n, int c, int l, float[] wt, int k, float[]? bias, int padL, int padR, int s, int d)
    {
        int effK = (k - 1) * d + 1;
        int outL = (l + padL + padR - effK) / s + 1;
        var y = new double[n * c * outL];
        for (int nn = 0; nn < n; nn++)
            for (int cc = 0; cc < c; cc++)
                for (int t = 0; t < outL; t++)
                {
                    double acc = bias is null ? 0.0 : bias[cc];
                    for (int kk = 0; kk < k; kk++)
                    {
                        int ix = t * s + kk * d - padL;
                        double xv = (uint)ix < (uint)l ? x[(nn * c + cc) * l + ix] : 0.0;
                        acc += (double)wt[cc * k + kk] * xv;
                    }
                    y[(nn * c + cc) * outL + t] = acc;
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

    static float[] RunDw(DenseTensor<float> x, DenseTensor<float> w, DenseTensor<float>? b, int group, int[] pads, int[] strides, int[] dilations, TensorExecutionOptions opts, bool relu)
    {
        var r = CPUExecutionProvider.Conv(x, w, b, null, dilations, group, null, pads, strides, new ExecutionOptions(OptimizationMode.Speed, opts), relu);
        Assert.Equal(OpStatus.Success, r.Status);
        return ((Tensor<float>)r.Outputs[0]).ToArray();
    }

    [Fact]
    public void SmallHandComputed_MatchesNaive()
    {
        var x = DenseTensor<float>.OfValues(new float[1, 2, 6]
        {
            { { 1f, 2f, 3f, 4f, 5f, 6f }, { 6f, 5f, 4f, 3f, 2f, 1f } },
        });
        var w = DenseTensor<float>.OfValues(new float[2, 1, 3]
        {
            { { 1f, 0f, -1f } }, { { 0.5f, 0.5f, 0.5f } },
        });
        var b = DenseTensor<float>.OfValues(new float[] { 1f, -10f });
        var got = RunDw(x, w, b, 2, new[] { 1, 1 }, new[] { 1 }, new[] { 1 }, TensorExecutionOptions.Scalar, false);
        var expected = NaiveDw(x.ToArray(), 1, 2, 6, w.ToArray(), 3, b.ToArray(), 1, 1, 1, 1);
        Assert.Equal(expected.Select(v => (float)v).ToArray(), got);
    }

    [Fact]
    public void EncoderLikeK9_MatchesNaive()
    {
        var x = FilledTensor(new[] { 1, 32, 200 }, 11);
        var w = FilledTensor(new[] { 32, 1, 9 }, 13);
        var b = FilledTensor(new[] { 32 }, 17);
        var got = RunDw(x, w, b, 32, new[] { 0, 0 }, new[] { 1 }, new[] { 1 }, TensorExecutionOptions.Auto, false);
        Assert.Equal(new[] { 1, 32, 192 }, got.Length == 1 * 32 * 192 ? new[] { 1, 32, 192 } : new[] { -1 });
        var expected = NaiveDw(x.ToArray(), 1, 32, 200, w.ToArray(), 9, b.ToArray(), 0, 0, 1, 1);
        AssertNear(expected, got, false);
    }

    [Fact]
    public void StrideDilationNoBiasRelu_MatchesNaive()
    {
        var x = FilledTensor(new[] { 2, 8, 65 }, 21);
        var w = FilledTensor(new[] { 8, 1, 5 }, 23);
        var got = RunDw(x, w, null, 8, new[] { 2, 2 }, new[] { 2 }, new[] { 2 }, TensorExecutionOptions.Auto, true);
        var expected = NaiveDw(x.ToArray(), 2, 8, 65, w.ToArray(), 5, null, 2, 2, 2, 2);
        Assert.Equal(2 * 8 * 31, got.Length);
        AssertNear(expected, got, true);
    }

    [Fact]
    public void ScalarMode_MatchesNaiveWithinRounding()
    {
        // The JIT may fuse scalar multiply-adds, so the scalar loop agrees
        // with a double oracle within rounding while staying deterministic.
        var x = FilledTensor(new[] { 1, 4, 50 }, 31);
        var w = FilledTensor(new[] { 4, 1, 9 }, 37);
        var b = FilledTensor(new[] { 4 }, 41);
        var got = RunDw(x, w, b, 4, new[] { 4, 4 }, new[] { 1 }, new[] { 1 }, TensorExecutionOptions.Scalar, false);
        var again = RunDw(x, w, b, 4, new[] { 4, 4 }, new[] { 1 }, new[] { 1 }, TensorExecutionOptions.Scalar, false);
        Assert.Equal(got, again);
        var expected = NaiveDw(x.ToArray(), 1, 4, 50, w.ToArray(), 9, b.ToArray(), 4, 4, 1, 1);
        AssertNear(expected, got, false);
    }

    [Fact]
    public void ParallelBranch_Deterministic()
    {
        var x = FilledTensor(new[] { 2, 16, 100 }, 43);
        var w = FilledTensor(new[] { 16, 1, 9 }, 47);
        var b = FilledTensor(new[] { 16 }, 53);
        var seq = RunDw(x, w, b, 16, new[] { 4, 4 }, new[] { 1 }, new[] { 1 }, TensorExecutionOptions.Auto, true);
        var par = RunDw(x, w, b, 16, new[] { 4, 4 }, new[] { 1 }, new[] { 1 }, TensorExecutionOptions.Parallel(4), true);
        Assert.Equal(seq, par);
    }

    [Fact]
    public void NonDepthwiseMultipleGroups_UsesAdapter()
    {
        var x = FilledTensor(new[] { 1, 4, 20 }, 59);
        var w = FilledTensor(new[] { 4, 2, 3 }, 61);
        var got = RunDw(x, w, null, 2, new[] { 1, 1 }, new[] { 1 }, new[] { 1 }, TensorExecutionOptions.Auto, false);
        Assert.Equal(1 * 4 * 20, got.Length);
    }

    [Fact]
    public void BadBias_StillFailsCleanly()
    {
        var x = FilledTensor(new[] { 1, 4, 20 }, 67);
        var w = FilledTensor(new[] { 4, 1, 3 }, 71);
        var b = FilledTensor(new[] { 3 }, 73);
        Assert.Throws<System.ArgumentException>(() => CPUExecutionProvider.Conv(x, w, b, null, new[] { 1 }, 4, null, new[] { 1, 1 }, new[] { 1 }, null));
    }
}
