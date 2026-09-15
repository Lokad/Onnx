namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Pins the single-pass scalar tanh-GELU kernel (the contract-gated
/// vectorization baseline): out-of-place/destination parity, exceptional
/// propagation, and double-reference bounds. Any future vectorized twin
/// must reproduce these exact bits and bounds; the fused/unfused chain
/// agreement itself lives in GraphFusionGeluTanhTests.
/// </summary>
public class GeluTanhKernelTests
{
    static float[] RandomData(int n, int seed)
    {
        var rnd = new System.Random(seed);
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)rnd.NextDouble() * 8f - 4f;
        return data;
    }

    static void AssertBitsEqual(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(System.BitConverter.SingleToInt32Bits(expected[i]) == System.BitConverter.SingleToInt32Bits(actual[i]),
                what + " differs at " + i + ": " + expected[i] + " vs " + actual[i]);
    }

    static float[] RunOutOfPlace(float[] data)
    {
        var input = DenseTensor<float>.OfValues((float[])data.Clone());
        return Tensor<float>.GeluTanh(input, TensorExecutionOptions.Auto).ToArray();
    }

    static float[] RunDestination(float[] data)
    {
        var input = DenseTensor<float>.OfValues((float[])data.Clone());
        var dest = DenseTensor<float>.OfShape(new[] { data.Length });
        Tensor<float>.GeluTanh(input, dest, TensorExecutionOptions.Auto);
        return dest.ToArray();
    }

    [Theory]
    [InlineData(1, 41)]
    [InlineData(7, 42)]
    [InlineData(100, 43)]
    [InlineData(768, 44)]
    public void Kernel_DestinationMatchesOutOfPlace(int n, int seed)
    {
        var data = RandomData(n, seed);
        AssertBitsEqual(RunOutOfPlace(data), RunDestination(data), "n=" + n);
    }

    [Fact]
    public void Kernel_ExceptionalPropagation()
    {
        var data = new float[] { float.PositiveInfinity, float.NegativeInfinity, float.NaN, -0f, 0f, 1e-30f, -1e-30f, 88f, -88f };
        var got = RunOutOfPlace(data);
        AssertBitsEqual(RunDestination(data), got, "exceptional parity");
        Assert.Equal(float.PositiveInfinity, got[0]);
        // -Inf: t1 * (tanh(-Inf) + 1) is -Inf * 0, so the kernel yields NaN.
        // Pinned deliberately: a vectorized twin must decide this case, not drift into it.
        Assert.True(float.IsNaN(got[1]));
        Assert.True(float.IsNaN(got[2]));
        Assert.Equal(int.MinValue, System.BitConverter.SingleToInt32Bits(got[3]));
        Assert.Equal(0, System.BitConverter.SingleToInt32Bits(got[4]));
        Assert.True(float.IsFinite(got[5]) && got[5] > 0f);
        Assert.True(float.IsFinite(got[6]) && got[6] < 0f);
        Assert.Equal(88f, got[7]);
        // -88f: tanh saturates to exactly -1f, so -44f * 0f is negative zero.
        Assert.Equal(int.MinValue, System.BitConverter.SingleToInt32Bits(got[8]));
    }

    [Fact]
    public void Kernel_DoubleReferenceBounds()
    {
        var data = RandomData(2048, 45);
        var got = RunOutOfPlace(data);
        double worst = 0;
        for (int i = 0; i < data.Length; i++)
        {
            double v = data[i];
            double reference = 0.5 * v * (1.0 + System.Math.Tanh(0.79788456 * (v + 0.044715 * v * v * v)));
            double scaled = System.Math.Abs(got[i] - reference) / (1.0 + System.Math.Abs(reference));
            if (scaled > worst) worst = scaled;
        }
        Assert.True(worst <= 1e-6, "worst scaled error " + worst);
    }

    static float[] RunVectorTwin(float[] data)
    {
        var xs = new float[data.Length];
        var ys = new float[data.Length];
        System.Array.Copy(data, xs, data.Length);
        Tensor<float>.GeluTanhSpanFloat(xs, ys);
        return ys;
    }

    [Theory]
    [InlineData(1, 51)]
    [InlineData(7, 52)]
    [InlineData(9, 53)]
    [InlineData(100, 54)]
    [InlineData(768, 55)]
    [InlineData(3072, 56)]
    public void VectorTwin_MatchesScalarWithin1e6(int n, int seed)
    {
        var data = RandomData(n, seed);
        var scalar = RunDestination(data);
        var twin = RunVectorTwin(data);
        double worst = 0;
        for (int i = 0; i < data.Length; i++)
        {
            double scaled = System.Math.Abs(twin[i] - scalar[i]) / (1.0 + System.Math.Abs(scalar[i]));
            if (scaled > worst) worst = scaled;
        }
        Assert.True(worst <= 1e-6, "vector twin worst scaled error " + worst);
    }

    [Fact]
    public void VectorTwin_ExceptionalBitParity()
    {
        var data = new float[] { float.PositiveInfinity, float.NegativeInfinity, float.NaN, -0f, 0f, 1e-30f, -1e-30f, 88f, -88f, 1e30f, -1e30f, 10f, -10f };
        AssertBitsEqual(RunDestination(data), RunVectorTwin(data), "exceptional twin parity");
    }

    [Fact]
    public void VectorTwin_DoubleReferenceBounds()
    {
        var data = RandomData(2048, 57);
        var got = RunVectorTwin(data);
        double worst = 0;
        for (int i = 0; i < data.Length; i++)
        {
            double v = data[i];
            double reference = 0.5 * v * (1.0 + System.Math.Tanh(0.79788456 * (v + 0.044715 * v * v * v)));
            double scaled = System.Math.Abs(got[i] - reference) / (1.0 + System.Math.Abs(reference));
            if (scaled > worst) worst = scaled;
        }
        Assert.True(worst <= 1e-6, "vector twin worst scaled error " + worst);
    }
}
