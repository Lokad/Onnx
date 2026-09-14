namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Pins the shared row-dot primitive behind the short LSTM projections:
/// exact on tiny sizes, within float rounding of a double oracle at width,
/// deterministic, and consistent across SIMD gating modes.
/// </summary>
public class RowDotTests
{
    static float[] Pattern(int n, int seed)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (((i * 37 + seed) % 97) - 48) * 0.02f;
        return a;
    }

    static double Oracle(float[] x, float[] y)
    {
        double acc = 0.0;
        for (int i = 0; i < x.Length; i++) acc += (double)x[i] * y[i];
        return acc;
    }

    [Fact]
    public void Widths_MatchOracle()
    {
        foreach (int n in new[] { 1, 7, 8, 9, 15, 16, 63, 64, 128, 640 })
        {
            var x = Pattern(n, 11);
            var y = Pattern(n, 77);
            double e = Oracle(x, y);
            foreach (var opts in new[] { TensorExecutionOptions.Auto, TensorExecutionOptions.Scalar, TensorExecutionOptions.Simd })
            {
                float a = MathOps.RowDot(x, y, opts);
                double tol = 2e-6 * (1.0 + System.Math.Abs(e));
                Assert.True(System.Math.Abs(a - e) <= tol, "n=" + n + " opts=" + opts + ": " + a + " vs " + e);
            }
        }
    }

    [Fact]
    public void Empty_ReturnsZero()
    {
        Assert.Equal(0f, MathOps.RowDot(System.Array.Empty<float>(), System.Array.Empty<float>(), TensorExecutionOptions.Auto));
    }

    [Fact]
    public void QuadMatchesFourDotsBitwise()
    {
        // The fused gate projection must agree bit-wise with four RowDot
        // calls in the same gating mode: any FMA-order drift fails loudly.
        foreach (int n in new[] { 1, 7, 8, 9, 15, 16, 63, 64, 128, 640 })
        {
            var x = Pattern(n, 11);
            var y0 = Pattern(n, 77);
            var y1 = Pattern(n, 78);
            var y2 = Pattern(n, 79);
            var y3 = Pattern(n, 80);
            foreach (var opts in new[] { TensorExecutionOptions.Auto, TensorExecutionOptions.Scalar, TensorExecutionOptions.Simd })
            {
                float e0 = MathOps.RowDot(x, y0, opts);
                float e1 = MathOps.RowDot(x, y1, opts);
                float e2 = MathOps.RowDot(x, y2, opts);
                float e3 = MathOps.RowDot(x, y3, opts);
                MathOps.RowDot4(x, y0, y1, y2, y3, out float a0, out float a1, out float a2, out float a3, opts);
                Assert.True(a0 == e0 && a1 == e1 && a2 == e2 && a3 == e3,
                    "n=" + n + " opts=" + opts);
            }
        }
    }

    [Fact]
    public void QuadEmpty_ReturnsZero()
    {
        MathOps.RowDot4(System.Array.Empty<float>(), System.Array.Empty<float>(), System.Array.Empty<float>(), System.Array.Empty<float>(), System.Array.Empty<float>(), out float a0, out float a1, out float a2, out float a3, TensorExecutionOptions.Auto);
        Assert.True(a0 == 0f && a1 == 0f && a2 == 0f && a3 == 0f);
    }

    [Fact]
    public void SimdAgreesWithScalar()
    {
        var x = Pattern(640, 5);
        var y = Pattern(640, 9);
        float s = MathOps.RowDot(x, y, TensorExecutionOptions.Scalar);
        float v = MathOps.RowDot(x, y, TensorExecutionOptions.Auto);
        double tol = 1e-6 * (1.0 + System.Math.Abs((double)s));
        Assert.True(System.Math.Abs(v - s) <= tol, v + " vs " + s);
    }
}
