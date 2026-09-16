namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Covers the stride-two width gather in MathOps.Im2colRange (vectorized
/// deinterleave of contiguous inputs): stride-2 3x3 convolutions agree with
/// an independent naive oracle across vector/scalar widths, even/odd pad
/// parities, asymmetric pads, grouped and biased variants, and stay
/// bit-deterministic across runs (pure copies, no reassociation).
/// </summary>
public class ConvStride2GatherTests
{
    static float Pattern(int i) => ((i * 37 + 11) % 97 - 48) * 0.03f;

    static DenseTensor<float> FilledTensor(int[] dims)
    {
        var t = DenseTensor<float>.OfShape(dims);
        var span = t.Buffer.Span;
        for (int i = 0; i < span.Length; i++) span[i] = Pattern(i);
        return t;
    }

    static double[] NaiveConvPad(float[] x, int n, int c, int h, int w, float[] wt, int m, int kh, int kw, float[]? bias, int padTop, int padLeft, int padBottom, int padRight, int sH, int sW, int group)
    {
        int outH = (h + padTop + padBottom - kh) / sH + 1;
        int outW = (w + padLeft + padRight - kw) / sW + 1;
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
                                int iy = oy * sH - padTop + ky;
                                if ((uint)iy >= (uint)h) continue;
                                for (int kx = 0; kx < kw; kx++)
                                {
                                    int ix = ox * sW - padLeft + kx;
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

    static void AssertNear(double[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            double e = expected[i];
            double tol = 1e-4 * (1.0 + System.Math.Abs(e));
            Assert.True(System.Math.Abs(actual[i] - e) <= tol, "index " + i + ": actual " + actual[i] + " expected " + e);
        }
    }

    static void CheckStride2(int c, int h, int w, int m, int group, int padTop, int padLeft, int padBottom, int padRight, bool bias)
    {
        var x = FilledTensor(new[] { 1, c, h, w });
        var wgt = FilledTensor(new[] { m, c / group, 3, 3 });
        var b = bias ? FilledTensor(new[] { m }) : null;
        var pads = new int[] { padTop, padLeft, padBottom, padRight };
        var y = Tensor<float>.Conv2D(x, wgt, group, pads, b, null, new int[] { 2, 2 }, null, TensorExecutionOptions.Auto, false);
        int outH = (h + padTop + padBottom - 3) / 2 + 1;
        int outW = (w + padLeft + padRight - 3) / 2 + 1;
        Assert.Equal(new int[] { 1, m, outH, outW }, y.Dimensions.ToArray());
        var expected = NaiveConvPad(x.Buffer.ToArray(), 1, c, h, w, wgt.Buffer.ToArray(), m, 3, 3, b is null ? null : b.Buffer.ToArray(), padTop, padLeft, padBottom, padRight, 2, 2, group);
        AssertNear(expected, y.ToArray());
    }

    [Theory]
    [InlineData(9)]
    [InlineData(15)]
    [InlineData(16)]
    [InlineData(17)]
    [InlineData(24)]
    [InlineData(31)]
    [InlineData(32)]
    [InlineData(33)]
    [InlineData(40)]
    [InlineData(65)]
    public void Stride2Pad1_WidthSweep_MatchesNaive(int w)
    {
        // Narrow widths stay on the scalar gather; widths past 16 inputs fire
        // the vectorized deinterleave interior with scalar head/tail.
        CheckStride2(8, 11, w, 8, 1, 1, 1, 1, 1, true);
    }

    [Fact]
    public void Stride2Pad0_ParitySweep_MatchesNaive()
    {
        // Zero padding makes col0 = kx, covering even and odd gather bases.
        CheckStride2(8, 11, 33, 8, 1, 0, 0, 0, 0, false);
    }

    [Fact]
    public void Stride2AsymmetricPads_MatchesNaive()
    {
        CheckStride2(8, 11, 33, 8, 1, 1, 0, 1, 2, true);
        CheckStride2(8, 11, 33, 8, 1, 0, 2, 2, 0, false);
    }

    [Fact]
    public void Stride2Grouped_MatchesNaive()
    {
        CheckStride2(8, 11, 33, 8, 2, 1, 1, 1, 1, false);
    }

    [Fact]
    public void Stride2TransitionLike_MatchesNaive()
    {
        CheckStride2(32, 20, 50, 64, 1, 1, 1, 1, 1, true);
    }

    [Fact]
    public void Stride2Result_IsBitDeterministic()
    {
        var x = FilledTensor(new[] { 1, 8, 11, 33 });
        var w = FilledTensor(new[] { 8, 8, 3, 3 });
        var b = FilledTensor(new[] { 8 });
        var pads = new int[] { 1, 1, 1, 1 };
        var first = Tensor<float>.Conv2D(x, w, 1, pads, b, null, new int[] { 2, 2 }, null, TensorExecutionOptions.Auto, false);
        var second = Tensor<float>.Conv2D(x, w, 1, pads, b, null, new int[] { 2, 2 }, null, TensorExecutionOptions.Auto, false);
        Assert.Equal(first.ToArray(), second.ToArray());
    }
}