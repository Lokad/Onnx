using Xunit;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// E81: the 512-bit exponential probe holds the documented contract
/// (order 1e-7 against MathF.Exp) and matches the 256-bit Estrin core
/// within a tight envelope, proving the port changed width, not
/// mathematics. Skips where 512-bit vectors or FMA are unavailable.
/// </summary>
public class Exp512Tests
{
    static bool ProbeAvailable()
    {
        return System.Runtime.Intrinsics.Vector512.IsHardwareAccelerated
            && System.Runtime.Intrinsics.X86.Fma.IsSupported;
    }

    static void Sweep(float lo, float hi, float step, double contract)
    {
        int n = (int)((hi - lo) / step) + 1;
        var x = new float[n];
        for (int i = 0; i < n; i++) x[i] = lo + i * step;
        var y = new float[n];
        MathOps.ExpSpan512(x, y);
        double worst = 0;
        for (int i = 0; i < n; i++)
        {
            float want = MathF.Exp(x[i]);
            if (float.IsInfinity(want)) { Assert.True(float.IsPositiveInfinity(y[i]), $"expected +Inf at {x[i]}."); continue; }
            if (want == 0f) { Assert.True(y[i] == 0f, $"expected +0 at {x[i]}, got {y[i]:R}."); continue; }
            double rel = Math.Abs((double)y[i] - want) / want;
            if (rel > worst) worst = rel;
        }
        Assert.True(worst <= contract, $"sweep [{lo},{hi}] worst rel err {worst:E2} exceeds {contract:E2}.");
    }

    [SkippableFact]
    public void ProbeHoldsContractOnWideSweep()
    {
        Skip.IfNot(ProbeAvailable(), "Requires AVX-512 plus FMA.");
        Sweep(-88f, 88f, 0.011f, 1e-7);
    }

    [SkippableFact]
    public void ProbeHoldsContractOnReducedRange()
    {
        Skip.IfNot(ProbeAvailable(), "Requires AVX-512 plus FMA.");
        Sweep(-1f, 1f, 0.00021f, 1e-7);
    }

    [SkippableFact]
    public void ProbeMatchesEstrinCore()
    {
        Skip.IfNot(ProbeAvailable(), "Requires AVX-512 plus FMA.");
        var rnd = new Random(81081);
        int n = 4096;
        var x = new float[n];
        for (int i = 0; i < n; i++) x[i] = (float)(rnd.NextDouble() * 176.0 - 88.0);
        var y512 = new float[n];
        MathOps.ExpSpan512(x, y512);
        double worst = 0;
        int w = System.Numerics.Vector<float>.Count;
        for (int i = 0; i < n; i += w)
        {
            var v = new System.Numerics.Vector<float>(x, i);
            var e = MathOps.ExpVectorEstrin(v);
            for (int l = 0; l < w && i + l < n; l++)
            {
                float a = y512[i + l];
                float b = e[l];
                if (float.IsNaN(a) || float.IsNaN(b)) continue;
                double denom = Math.Max(Math.Abs((double)b), 1e-30);
                double rel = Math.Abs((double)a - b) / denom;
                if (rel > worst) worst = rel;
            }
        }
        Assert.True(worst <= 2e-7, $"512 vs Estrin worst rel {worst:E2} exceeds 2e-7.");
    }

    [SkippableFact]
    public void ProbeHandlesExceptionals()
    {
        Skip.IfNot(ProbeAvailable(), "Requires AVX-512 plus FMA.");
        float[] x = new float[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity, -0f, 0f, 1e30f, -1e30f, 88.722839f, -88.722839f, 89f, -89f };
        var y = new float[x.Length];
        MathOps.ExpSpan512(x, y);
        Assert.True(float.IsNaN(y[0]), "NaN in must give NaN out.");
        Assert.True(float.IsPositiveInfinity(y[1]), "+Inf in must overflow.");
        Assert.True(y[2] == 0f, "-Inf in must underflow to zero.");
        Assert.True(y[3] == 1f && y[4] == 1f, "signed zero must give exactly 1.");
        Assert.True(float.IsPositiveInfinity(y[5]), "1e30 must overflow.");
        Assert.True(y[6] == 0f, "-1e30 must underflow.");
    }

    [SkippableTheory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(15)]
    [InlineData(16)]
    [InlineData(17)]
    [InlineData(33)]
    public void ProbeTailsMatchScalar(int n)
    {
        Skip.IfNot(ProbeAvailable(), "Requires AVX-512 plus FMA.");
        var rnd = new Random(n * 31 + 5);
        var x = new float[n];
        for (int i = 0; i < n; i++) x[i] = (float)(rnd.NextDouble() * 12.0 - 6.0);
        var y = new float[n];
        MathOps.ExpSpan512(x, y);
        for (int i = 0; i < n; i++)
        {
            float want = MathF.Exp(x[i]);
            double rel = Math.Abs((double)y[i] - want) / Math.Max(Math.Abs((double)want), 1e-30);
            Assert.True(rel <= 1e-7, $"tail differs at {i}: {y[i]:R} vs {want:R}.");
        }
    }
}