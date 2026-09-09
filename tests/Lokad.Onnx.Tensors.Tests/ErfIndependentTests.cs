using System;

namespace Lokad.Onnx.Tensors.Tests;

/// <summary>
/// Independent oracle for scalar Erf against the Abramowitz and Stegun
/// Handbook formula 7.1.26 reference values (public-domain mathematics).
/// High-precision expected values are mathematical facts, not copied code;
/// the 1e-6 tolerance covers the documented A and S approximation error.
/// Passes on the pre-rewrite and rewritten bodies, proving spec parity.
/// </summary>
public class ErfIndependentTests
{
    static readonly double[] Points = new double[] { 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0 };

    static readonly double[] Expected = new double[]
    {
        0.0,
        0.2763263901682369,
        0.5204998778130465,
        0.7111556336535151,
        0.8427007929497149,
        0.9661051464753107,
        0.9953222650189527,
        0.9999779095030014,
    };

    [Fact]
    public void FloatErf_MatchesStegunReference()
    {
        for (int i = 0; i < Points.Length; i++)
        {
            float actual = MathOps.Erf((float)Points[i]);
            double drift = Math.Abs(actual - Expected[i]);
            Assert.True(drift <= 1e-6, "float erf(" + Points[i] + ") drift " + drift);
        }
    }

    [Fact]
    public void DoubleErf_MatchesStegunReference()
    {
        for (int i = 0; i < Points.Length; i++)
        {
            double actual = MathOps.Erf(Points[i]);
            double drift = Math.Abs(actual - Expected[i]);
            Assert.True(drift <= 1e-6, "double erf(" + Points[i] + ") drift " + drift);
        }
    }

    [Fact]
    public void Erf_EdgeCases()
    {
        Assert.True(float.IsNaN(MathOps.Erf(float.NaN)));
        Assert.True(double.IsNaN(MathOps.Erf(double.NaN)));
        Assert.Equal(1f, MathOps.Erf(float.PositiveInfinity));
        Assert.Equal(-1f, MathOps.Erf(float.NegativeInfinity));
        Assert.Equal(1.0, MathOps.Erf(double.PositiveInfinity));
        Assert.Equal(-1.0, MathOps.Erf(double.NegativeInfinity));
        Assert.True(Math.Abs(MathOps.Erf(0f)) <= 1e-7);
        Assert.True(Math.Abs(MathOps.Erf(0.0)) <= 1e-9);
    }

    [Fact]
    public void Erf_IsOddFunction()
    {
        float[] floats = new float[] { 0.25f, 0.5f, 1f, 1.5f, 2f };
        foreach (float x in floats)
        {
            Assert.True(MathOps.Erf(-x) == -MathOps.Erf(x));
        }
        double[] doubles = new double[] { 0.25, 0.5, 1.0, 1.5, 2.0 };
        foreach (double x in doubles)
        {
            Assert.True(MathOps.Erf(-x) == -MathOps.Erf(x));
        }
    }
}
