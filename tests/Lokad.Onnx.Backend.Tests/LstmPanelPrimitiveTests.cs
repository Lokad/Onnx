using System;

namespace Lokad.Onnx.Backend.Tests;

// Covers the shared output-lane GEMV primitive over prepared [K,O] panels:
// independent double-oracle agreement on LSTM-like shapes, bitwise
// vector/scalar agreement, tails, overwrite/accumulate modes, and degenerate
// extents. Timing lives in the segmentation/decoder benchmarks, not here.
public class LstmPanelPrimitiveTests
{
    static void Fill(Random rnd, Span<float> s, float scale)
    {
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rnd.NextDouble() * 2 * scale - scale);
    }

    static double[] Oracle(ReadOnlySpan<float> x, ReadOnlySpan<float> panel, double[] y0, int o, int k, bool accumulate)
    {
        var y = new double[o];
        for (int t = 0; t < o; t++)
        {
            double s = accumulate ? y0[t] : 0.0;
            for (int j = 0; j < k; j++) s += (double)x[j] * panel[j * o + t];
            y[t] = s;
        }
        return y;
    }

    static void AssertScaledNear(double[] expected, float[] actual, double tol, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        double worst = 0;
        int at = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            double err = Math.Abs(actual[i] - expected[i]) / (1.0 + Math.Abs(expected[i]));
            if (err > worst) { worst = err; at = i; }
        }
        Assert.True(worst <= tol, what + " worst=" + worst.ToString("E2") + " at " + at);
    }

    static void Case(int k, int o, bool accumulate, int seed, string what) =>
        CaseTol(k, o, accumulate, seed, what, 1e-6);

    static void CaseTol(int k, int o, bool accumulate, int seed, string what, double tol)
    {
        var rnd = new Random(seed);
        var xs = new float[k];
        var ps = new float[k * o];
        var ys = new float[o];
        var y0 = new float[o];
        Fill(rnd, xs, 1.0f);
        Fill(rnd, ps, 0.5f);
        Fill(rnd, ys, 2.0f);
        Array.Copy(ys, y0, o);
        MathOps.MatVecPanel(xs, ps, ys, o, k, accumulate, TensorExecutionOptions.Intrinsics);
        AssertScaledNear(Oracle(xs, ps, Array.ConvertAll(y0, v => (double)v), o, k, accumulate), ys, tol, what);
    }

    [Fact]
    public void MatchesDoubleOracle_SegRecurrentShape()
    {
        Case(128, 512, false, 11, "overwrite-128x512");
        Case(128, 512, true, 12, "accumulate-128x512");
    }

    [Fact]
    public void MatchesDoubleOracle_DecoderRecurrentShape()
    {
        // K=640 sums accumulate float rounding to a few ulps (random-walk
        // scale), so this shape pins 1e-5 -- still ten times inside the
        // bench gate -- while K<=128 shapes hold 1e-6.
        CaseTol(640, 2560, false, 13, "overwrite-640x2560", 1e-5);
        CaseTol(640, 2560, true, 14, "accumulate-640x2560", 1e-5);
    }

    [Fact]
    public void MatchesDoubleOracle_Tails()
    {
        Case(128, 1, false, 21, "tail-1");
        Case(128, 7, true, 22, "tail-7");
        Case(64, 100, false, 23, "tail-100");
        Case(127, 129, true, 24, "tail-127x129");
        Case(48, 63, false, 25, "tail-48x63");
    }

    [Fact]
    public void VectorScalarAgreeBitwise()
    {
        var rnd = new Random(31);
        foreach (int o in new[] { 1, 7, 63, 64, 65, 100, 127, 128, 129, 200, 512 })
        {
            int k = 128;
            var xs = new float[k];
            var ps = new float[k * o];
            Fill(rnd, xs, 1.0f);
            Fill(rnd, ps, 0.5f);
            foreach (bool acc in new[] { false, true })
            {
                var yv = new float[o];
                var ys = new float[o];
                Fill(rnd, yv, 2.0f);
                Array.Copy(yv, ys, o);
                MathOps.MatVecPanel(xs, ps, yv, o, k, acc, TensorExecutionOptions.Intrinsics);
                MathOps.MatVecPanel(xs, ps, ys, o, k, acc, TensorExecutionOptions.Scalar);
                Assert.Equal(yv, ys);
            }
        }
    }

    [Fact]
    public void DegenerateExtents()
    {
        var xs = new float[] { 1f, 2f };
        var ps = new float[] { 1f, 0f, 0f, 1f };
        var y = new float[] { 9f, 9f };
        MathOps.MatVecPanel(xs, ps, y, 0, 2, false, TensorExecutionOptions.Intrinsics);
        Assert.Equal(new float[] { 9f, 9f }, y);
        MathOps.MatVecPanel(xs, ps, y, 2, 0, false, TensorExecutionOptions.Intrinsics);
        Assert.Equal(new float[] { 0f, 0f }, y);
        y[0] = 9f; y[1] = 9f;
        MathOps.MatVecPanel(xs, ps, y, 2, 0, true, TensorExecutionOptions.Intrinsics);
        Assert.Equal(new float[] { 9f, 9f }, y);
        MathOps.MatVecPanel(xs, ps, y, 2, 2, false, TensorExecutionOptions.Intrinsics);
        Assert.Equal(new float[] { 1f, 2f }, y);
    }

    [Fact]
    public void EvolvingTrajectoryMatchesOracle()
    {
        var rnd = new Random(41);
        int k = 128, o = 512;
        var ps = new float[k * o];
        Fill(rnd, ps, 0.5f);
        var xs = new float[k];
        Fill(rnd, xs, 1.0f);
        var ys = new float[o];
        for (int step = 0; step < 5; step++)
        {
            for (int i = 0; i < k; i++) xs[i] = 0.99f * xs[i] + 0.01f * (float)rnd.NextDouble();
            Array.Clear(ys, 0, o);
            MathOps.MatVecPanel(xs, ps, ys, o, k, false, TensorExecutionOptions.Intrinsics);
            AssertScaledNear(Oracle(xs, ps, new double[o], o, k, false), ys, 1e-6, "trajectory-" + step);
        }
    }
}
