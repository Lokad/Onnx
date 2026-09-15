namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Pins the vector exp-based sigmoid/tanh spans against scalar libm
/// oracles: dense coverage over the operating range plus infinities, NaN,
/// signed zeros, saturation knees, and tail widths, all within 1e-6 scaled.
/// </summary>
public class ExpSpanAccuracyTests
{
    static float[] Grid()
    {
        var list = new System.Collections.Generic.List<float>();
        for (int i = -10000; i <= 10000; i++) list.Add(i * 0.01f);
        list.AddRange(new float[]
        {
            float.PositiveInfinity, float.NegativeInfinity, float.NaN, -float.NaN,
            0f, -0f, 1f, -1f, 10f, -10f, 88f, -88f, 88.5f, -88.5f,
            1e-38f, -1e-38f, 1e-30f, 3.14f, -2.718f,
        });
        return list.ToArray();
    }

    static void AssertNear(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        double worst = 0;
        int worstIdx = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            double e = expected[i];
            double a = actual[i];
            double err;
            if (double.IsNaN(e) && double.IsNaN(a)) continue;
            if (double.IsInfinity(e) || double.IsInfinity(a))
            {
                Assert.True(e == a, what + " infinite mismatch at " + i + ": " + a + " vs " + e);
                continue;
            }
            err = System.Math.Abs(a - e) / (1.0 + System.Math.Abs(e));
            if (err > worst) { worst = err; worstIdx = i; }
        }
        Assert.True(worst <= 1e-6, what + " worst=" + worst.ToString("E2") + " at " + worstIdx);
    }

    [Fact]
    public void SigmoidSpan_MatchesScalar()
    {
        var x = Grid();
        var expected = new float[x.Length];
        for (int i = 0; i < x.Length; i++) expected[i] = 1f / (1f + MathF.Exp(-x[i]));
        var actual = new float[x.Length];
        MathOps.SigmoidSpan(x, actual);
        AssertNear(expected, actual, "sigmoid");
    }

    [Fact]
    public void TanhSpan_MatchesScalar()
    {
        var x = Grid();
        var expected = new float[x.Length];
        for (int i = 0; i < x.Length; i++) expected[i] = MathF.Tanh(x[i]);
        var actual = new float[x.Length];
        MathOps.TanhSpan(x, actual);
        AssertNear(expected, actual, "tanh");
    }

    [Fact]
    public void Spans_HandleTailWidths()
    {
        var rnd = new Random(97);
        foreach (int n in new[] { 1, 7, 8, 9, 15, 16, 100 })
        {
            var x = new float[n];
            for (int i = 0; i < n; i++) x[i] = (float)(rnd.NextDouble() * 20 - 10);
            var ss = new float[n];
            var ts = new float[n];
            MathOps.SigmoidSpan(x, ss);
            MathOps.TanhSpan(x, ts);
            for (int i = 0; i < n; i++)
            {
                double se = 1.0 / (1.0 + Math.Exp(-x[i]));
                double te = Math.Tanh(x[i]);
                Assert.True(System.Math.Abs(ss[i] - se) <= 1e-6 * (1 + System.Math.Abs(se)), "sigmoid n=" + n);
                Assert.True(System.Math.Abs(ts[i] - te) <= 1e-6 * (1 + System.Math.Abs(te)), "tanh n=" + n);
            }
        }
    }
}

