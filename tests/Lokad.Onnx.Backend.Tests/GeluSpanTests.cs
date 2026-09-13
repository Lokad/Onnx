namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Pins the single-pass exact GELU span kernel: bitwise identity with the
/// pre-change composed path (same ErfVector call and combine order, verified
/// against the git-history formula quoted below), in-place/out-of-place
/// parity, exceptional values, and double-reference bounds.
/// Pre-change reference (TensorOps.Elementwise.cs at 6e76d85):
/// vector: new Vector(0.5f) * v * (One + ErfVector(new Vector(0.7071067811865476f) * v))
/// scalar: 0.5f * v * (1f + MathOps.Erf(v * 0.7071067811865476f))
/// </summary>
public class GeluSpanTests
{
    static float[] ReferenceComposed(float[] data)
    {
        var v = new System.Numerics.Vector<float>[1];
        var out_ = new float[data.Length];
        int w = System.Numerics.Vector<float>.Count;
        var half = new System.Numerics.Vector<float>(0.5f);
        var one = System.Numerics.Vector<float>.One;
        var scale = new System.Numerics.Vector<float>(0.7071067811865476f);
        int i = 0;
        for (; i + w <= data.Length; i += w)
        {
            var x = new System.Numerics.Vector<float>(data, i);
            (half * x * (one + MathOps.ErfVector(scale * x))).CopyTo(out_, i);
        }
        for (; i < data.Length; i++)
        {
            float x = data[i];
            out_[i] = 0.5f * x * (1f + MathOps.Erf(x * 0.7071067811865476f));
        }
        return out_;
    }

    static void AssertBitsEqual(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(System.BitConverter.SingleToInt32Bits(expected[i]) == System.BitConverter.SingleToInt32Bits(actual[i]),
                what + " differs at " + i + ": " + expected[i] + " vs " + actual[i]);
    }

    static float[] RandomData(int n, int seed)
    {
        var rnd = new System.Random(seed);
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)rnd.NextDouble() * 8f - 4f;
        return data;
    }

    [Theory]
    [InlineData(257 * 384, 11)]
    [InlineData(30 * 1536, 12)]
    [InlineData(201 * 1536, 13)]
    [InlineData(100, 14)]
    [InlineData(7, 15)]
    [InlineData(1, 16)]
    public void SpanKernel_BitwiseMatchesComposed(int n, int seed)
    {
        var data = RandomData(n, seed);
        var input = DenseTensor<float>.OfValues((float[])data.Clone());
        var got = Tensor<float>.Gelu(input, TensorExecutionOptions.Auto).ToArray();
        AssertBitsEqual(ReferenceComposed(data), got, "n=" + n);
    }

    [Fact]
    public void SpanKernel_ExceptionalMatchesComposed()
    {
        var data = new float[] { float.PositiveInfinity, float.NegativeInfinity, float.NaN, -0f, 0f, 1e-30f, -1e-30f, 88f, -88f, 3.925f, 0.921875f };
        var input = DenseTensor<float>.OfValues((float[])data.Clone());
        var got = Tensor<float>.Gelu(input, TensorExecutionOptions.Auto).ToArray();
        AssertBitsEqual(ReferenceComposed(data), got, "exceptional");
        Assert.True(float.IsNaN(got[2]));
        Assert.Equal(int.MinValue, System.BitConverter.SingleToInt32Bits(got[3]));
    }

    [Fact]
    public void SpanKernel_InPlaceMatchesOutOfPlace()
    {
        var data = RandomData(1000, 21);
        var a = DenseTensor<float>.OfValues((float[])data.Clone());
        var b = DenseTensor<float>.OfValues((float[])data.Clone());
        Tensor<float>.GeluSpanFloat(a.Buffer.Span, a.Buffer.Span);
        var ref_ = Tensor<float>.Gelu(b, TensorExecutionOptions.Auto).ToArray();
        AssertBitsEqual(ref_, a.ToArray(), "in-place");
    }

    [Fact]
    public void SpanKernel_DoubleReferenceBounds()
    {
        var data = RandomData(2048, 31);
        var input = DenseTensor<float>.OfValues((float[])data.Clone());
        var got = Tensor<float>.Gelu(input, TensorExecutionOptions.Auto).ToArray();
        double worst = 0;
        for (int i = 0; i < data.Length; i++)
        {
            double x = data[i];
            double ax = x * 0.7071067811865476;
            double erf = ax == 0 ? ax : (ax < 0 ? -ErfMag(-ax) : ErfMag(ax));
            double expected = 0.5 * x * (1.0 + erf);
            double scaled = System.Math.Abs(got[i] - expected) / (1.0 + System.Math.Abs(expected));
            if (scaled > worst) worst = scaled;
        }
        Assert.True(worst <= 1e-6, "scaled error " + worst);
    }

    static double ErfMag(double ax)
    {
        double t = 1.0 / (1.0 + 0.3275911 * ax);
        double[] table = new double[] { 1.061405429, -1.453152027, 1.421413741, -0.284496736, 0.254829592 };
        double poly = table[0];
        for (int i = 1; i < table.Length; i++) poly = poly * t + table[i];
        return 1.0 - poly * t * System.Math.Exp(-ax * ax);
    }
}
