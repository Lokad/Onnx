namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// E72: the two-way and four-way unrolled BiasGelu pointer twins reproduce
/// BiasGeluSpanFloatPtr bit-identically. Unrolling only interleaves
/// independent vectors, so per-lane operation order is unchanged; the tests
/// prove it with integer bit patterns (NaN payloads and signed zero included).
/// </summary>
public class BiasGeluUnrollTests
{
    static void BitsEqual(float[] actual, float[] expected, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            int ea = BitConverter.SingleToInt32Bits(actual[i]);
            int ee = BitConverter.SingleToInt32Bits(expected[i]);
            Assert.True(ea == ee, $"{what} differs at {i}: {actual[i]:R} vs {expected[i]:R}.");
        }
    }

    static void TwinsMatchRef(float[] x, float[] bias)
    {
        var yRef = new float[x.Length];
        var y2 = new float[x.Length];
        var y4 = new float[x.Length];
        for (int i = 0; i < yRef.Length; i++) { yRef[i] = float.NaN; y2[i] = float.NaN; y4[i] = float.NaN; }
        Tensor<float>.BiasGeluSpanFloatPtr(x, bias, yRef);
        Tensor<float>.BiasGeluSpanFloatPtr2x(x, bias, y2);
        Tensor<float>.BiasGeluSpanFloatPtr4x(x, bias, y4);
        BitsEqual(y2, yRef, "2x twin");
        BitsEqual(y4, yRef, "4x twin");
    }

    static float[] Rand(int n, Random rnd, float s)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rnd.NextDouble() * 2.0 - 1.0) * s;
        return a;
    }

    [Theory]
    [InlineData(384, 8)]
    [InlineData(384, 30)]
    [InlineData(384, 128)]
    [InlineData(1536, 8)]
    [InlineData(1536, 30)]
    [InlineData(1536, 128)]
    [InlineData(1536, 512)]
    public void TwinsMatchRefOnE5Shapes(int m, int rows)
    {
        var rnd = new Random(m * 7919 + rows);
        TwinsMatchRef(Rand(rows * m, rnd, 6f), Rand(m, rnd, 2f));
    }

    [Theory]
    [InlineData(16, 3)]
    [InlineData(16, 5)]
    [InlineData(24, 7)]
    [InlineData(8, 1)]
    [InlineData(8, 3)]
    [InlineData(32, 9)]
    public void TwinsMatchRefOnRemaindersAndWraps(int m, int rows)
    {
        var rnd = new Random(m * 104729 + rows);
        TwinsMatchRef(Rand(rows * m, rnd, 6f), Rand(m, rnd, 2f));
    }

    [Fact]
    public void TwinsMatchRefOnTails()
    {
        var rnd = new Random(7207);
        foreach (int extra in new[] { 1, 3, 7, 9 })
        {
            int m = 384;
            TwinsMatchRef(Rand(30 * m + extra, rnd, 6f), Rand(m, rnd, 2f));
        }
    }

    [Fact]
    public void TwinsMatchRefOnExceptionalPayloads()
    {
        float[] specials = new float[]
        {
            float.NegativeInfinity, float.PositiveInfinity, float.NaN,
            -0f, 0f, 1e30f, -1e30f, 88.7f, -88.7f, 1e-30f, -1e-30f, 10f, -10f, 3.925f, 0.921875f,
        };
        var rnd = new Random(72072);
        int m = 64;
        int rows = 5;
        var x = Rand(rows * m, rnd, 6f);
        var bias = Rand(m, rnd, 2f);
        for (int i = 0; i < specials.Length; i++)
        {
            x[(i * 37) % x.Length] = specials[i];
            bias[(i * 11) % bias.Length] = specials[(i * 5) % specials.Length];
        }
        TwinsMatchRef(x, bias);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(7)]
    public void TwinsMatchRefOnScalarFallback(int m)
    {
        var rnd = new Random(m * 31 + 7);
        TwinsMatchRef(Rand(48, rnd, 6f), Rand(m, rnd, 2f));
    }
}