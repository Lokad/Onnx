namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// E66: the Estrin exp keeps the documented ExpVector contract (order 1e-7
/// against MathF.Exp, NaN in NaN out, overflow/underflow edges) while
/// restructuring only the polynomial core with identical constants.
/// </summary>
public class ExpEstrinTests
{
    static float[] RunNew(float[] x)
    {
        int w = System.Numerics.Vector<float>.Count;
        var y = new float[x.Length];
        for (int i = 0; i < x.Length; i += w)
        {
            int n = System.Math.Min(w, x.Length - i);
            var v = new System.Numerics.Vector<float>(x, i);
            var r = MathOps.ExpVectorEstrin(v);
            r.CopyTo(y, i);
            _ = n;
        }
        return y;
    }

    static float[] RunOld(float[] x)
    {
        int w = System.Numerics.Vector<float>.Count;
        var y = new float[x.Length];
        for (int i = 0; i < x.Length; i += w)
        {
            var v = new System.Numerics.Vector<float>(x, i);
            MathOps.ExpVector(v).CopyTo(y, i);
        }
        return y;
    }

    [Fact]
    public void EstrinMatchesHornerWithinEnvelope()
    {
        var rnd = new Random(4242);
        var x = new float[4096];
        for (int i = 0; i < 2048; i++) x[i] = (float)(rnd.NextDouble() * 200 - 100);
        for (int i = 2048; i < x.Length; i++) x[i] = (float)(rnd.NextDouble() * 2 - 1);
        var a = RunNew(x);
        var b = RunOld(x);
        double worst = 0;
        for (int i = 0; i < x.Length; i++)
        {
            if (float.IsNaN(a[i]) && float.IsNaN(b[i])) continue;
            double d = System.Math.Abs(a[i] - b[i]) / (1.0 + System.Math.Abs(b[i]));
            if (d > worst) worst = d;
        }
        Assert.True(worst < 1e-6, $"estrin diverges {worst:E2} from horner.");
    }

    [Fact]
    public void EstrinMeetsScalarEnvelope()
    {
        var rnd = new Random(777);
        double worst = 0;
        for (int t = 0; t < 2048; t++)
        {
            float v = (float)(rnd.NextDouble() * 176 - 88);
            int w = System.Numerics.Vector<float>.Count;
            var probe = new float[w];
            for (int i = 0; i < w; i++) probe[i] = (i == 0) ? v : (0.25f * i - 1f);
            var a = RunNew(probe);
            double e = System.Math.Exp(v);
            double d = System.Math.Abs(a[0] - e) / (1.0 + System.Math.Abs(e));
            if (d > worst) worst = d;
        }
        Assert.True(worst < 1e-6, $"estrin scalar envelope {worst:E2} breached.");
    }

    [Fact]
    public void EstrinExceptionalParity()
    {
        float[] xs = { float.NaN, float.PositiveInfinity, float.NegativeInfinity, 89f, 100f, -89f, -100f, 0f, -0f };
        int w = System.Numerics.Vector<float>.Count;
        var x = new float[w * 2];
        for (int i = 0; i < x.Length; i++) x[i] = xs[i % xs.Length];
        var a = RunNew(x);
        var b = RunOld(x);
        for (int i = 0; i < x.Length; i++)
        {
            bool an = float.IsNaN(a[i]), bn = float.IsNaN(b[i]);
            Assert.True(an == bn, $"NaN parity differs at {i} for {x[i]:R}.");
            if (!an) Assert.True(a[i] == b[i], $"exceptional value differs at {i}: {a[i]:R} vs {b[i]:R}.");
        }
    }
}
