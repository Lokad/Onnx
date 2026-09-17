namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// E73: the row-pair softmax twins reproduce their references bit-identically.
/// Rows are independent by the definition of softmax, so interleaving two rows
/// changes no per-element operation order; the tests prove it with integer bit
/// patterns (NaN payloads and signed zero included).
/// </summary>
public class SoftmaxRowUnrollTests
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

    static float[] Rand(int n, Random rnd, float s)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rnd.NextDouble() * 2.0 - 1.0) * s;
        return a;
    }

    static void MaskedTwinsMatchRef(float[] x, float[] mask, int outer, int block, bool useSimd)
    {
        var yRef = new float[x.Length];
        var y2 = new float[x.Length];
        Tensor<float>.SoftmaxMaskedFloatSpan(x, mask, yRef, outer, block, useSimd);
        Tensor<float>.SoftmaxMaskedFloatSpan2x(x, mask, y2, outer, block, useSimd);
        BitsEqual(y2, yRef, "masked 2x twin");
    }

    static void PlainTwinsMatchRef(float[] x, int outer, int block, bool useSimd)
    {
        var yRef = new float[x.Length];
        var y2 = new float[x.Length];
        Tensor<float>.SoftmaxContiguousFloatSpan(x, yRef, outer, block, useSimd);
        Tensor<float>.SoftmaxContiguousFloatSpan2x(x, y2, outer, block, useSimd);
        BitsEqual(y2, yRef, "plain 2x twin");
    }

    static void MaskedQuadMatchesRef(float[] x, float[] mask, int outer, int block, bool useSimd)
    {
        var yRef = new float[x.Length];
        var y4 = new float[x.Length];
        Tensor<float>.SoftmaxMaskedFloatSpan(x, mask, yRef, outer, block, useSimd);
        Tensor<float>.SoftmaxMaskedFloatSpan4x(x, mask, y4, outer, block, useSimd);
        BitsEqual(y4, yRef, "masked 4x twin");
    }

    static void PlainQuadMatchesRef(float[] x, int outer, int block, bool useSimd)
    {
        var yRef = new float[x.Length];
        var y4 = new float[x.Length];
        Tensor<float>.SoftmaxContiguousFloatSpan(x, yRef, outer, block, useSimd);
        Tensor<float>.SoftmaxContiguousFloatSpan4x(x, y4, outer, block, useSimd);
        BitsEqual(y4, yRef, "plain 4x twin");
    }

    static float[] CausalMask(int block, Random rnd)
    {
        var m = new float[block];
        for (int i = 0; i < block; i++) m[i] = (float)(rnd.NextDouble() * 2.0 - 1.0);
        return m;
    }

    [Theory]
    [InlineData(8, 128)]
    [InlineData(30, 128)]
    [InlineData(128, 128)]
    [InlineData(5, 512)]
    [InlineData(7, 512)]
    [InlineData(12, 512)]
    public void TwinsMatchRefOnE5Shapes(int outer, int block)
    {
        var rnd = new Random(outer * 7919 + block);
        foreach (bool simd in new[] { true, false })
        {
            MaskedTwinsMatchRef(Rand(outer * block, rnd, 6f), CausalMask(block, rnd), outer, block, simd);
            PlainTwinsMatchRef(Rand(outer * block, rnd, 6f), outer, block, simd);
        }
    }

    [Theory]
    [InlineData(1, 128)]
    [InlineData(3, 128)]
    [InlineData(3, 7)]
    [InlineData(5, 9)]
    [InlineData(2, 30)]
    public void TwinsMatchRefOnRemaindersAndTails(int outer, int block)
    {
        var rnd = new Random(outer * 104729 + block);
        foreach (bool simd in new[] { true, false })
        {
            MaskedTwinsMatchRef(Rand(outer * block, rnd, 6f), CausalMask(block, rnd), outer, block, simd);
            PlainTwinsMatchRef(Rand(outer * block, rnd, 6f), outer, block, simd);
        }
    }

    [Theory]
    [InlineData(4, 128)]
    [InlineData(5, 128)]
    [InlineData(6, 128)]
    [InlineData(7, 128)]
    [InlineData(9, 128)]
    [InlineData(30, 128)]
    [InlineData(128, 128)]
    [InlineData(9, 512)]
    [InlineData(5, 30)]
    [InlineData(3, 7)]
    public void QuadMatchesRefOnRemainderClasses(int outer, int block)
    {
        var rnd = new Random(outer * 230909 + block);
        foreach (bool simd in new[] { true, false })
        {
            MaskedQuadMatchesRef(Rand(outer * block, rnd, 6f), CausalMask(block, rnd), outer, block, simd);
            PlainQuadMatchesRef(Rand(outer * block, rnd, 6f), outer, block, simd);
        }
    }

    [Fact]
    public void TwinsMatchRefOnExceptionalPayloads()
    {
        float[] specials = new float[]
        {
            float.NegativeInfinity, float.PositiveInfinity, float.NaN,
            -0f, 0f, 1e30f, -1e30f, 88.7f, -88.7f, 1e-30f, -1e-30f, 10f, -10f,
        };
        var rnd = new Random(73073);
        int outer = 4, block = 64;
        var x = Rand(outer * block, rnd, 6f);
        var mask = CausalMask(block, rnd);
        for (int i = 0; i < specials.Length; i++)
        {
            x[(i * 37) % x.Length] = specials[i];
            mask[(i * 11) % mask.Length] = specials[(i * 5) % specials.Length];
        }
        foreach (bool simd in new[] { true, false })
        {
            MaskedTwinsMatchRef(x, mask, outer, block, simd);
            PlainTwinsMatchRef(x, outer, block, simd);
            MaskedQuadMatchesRef(x, mask, outer, block, simd);
            PlainQuadMatchesRef(x, outer, block, simd);
        }
    }

    [Fact]
    public void MaskedTwinMatchesRefOnFullyMaskedRow()
    {
        int outer = 3, block = 32;
        var rnd = new Random(73074);
        var x = Rand(outer * block, rnd, 6f);
        var mask = new float[block];
        for (int i = 0; i < block; i++) mask[i] = float.NegativeInfinity;
        foreach (bool simd in new[] { true, false })
        {
            MaskedTwinsMatchRef(x, mask, outer, block, simd);
            MaskedQuadMatchesRef(x, mask, outer, block, simd);
        }
    }
}