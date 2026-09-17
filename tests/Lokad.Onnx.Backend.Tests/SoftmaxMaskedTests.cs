namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// E69: the fused masked softmax is bit-identical to Add-then-span-softmax
/// (same additions in the same order, only the materialized intermediate is
/// skipped), including all-masked NaN rows, infinities and SIMD on/off.
/// </summary>
public class SoftmaxMaskedTests
{
    static void MaskedEqual(int rows, int block, float[] mask, int seed, bool simd)
    {
        var rnd = new Random(seed);
        var scores = new float[rows * block];
        for (int i = 0; i < scores.Length; i++) scores[i] = (float)(rnd.NextDouble() * 8 - 4);
        var added = new float[scores.Length];
        for (int r = 0; r < rows; r++)
            for (int j = 0; j < block; j++)
                added[r * block + j] = scores[r * block + j] + mask[j];
        // Direct span-kernel reference (never the public Softmax: the E59 ablation
        // flags are process-wide and a concurrent legacy routing would break bitwise
        // comparison through summation grouping; the fused kernel replaces the span path).
        var expected = new float[added.Length];
        Tensor<float>.SoftmaxContiguousFloatSpan(added, expected, rows, block, simd);
        var actual = new float[scores.Length];
        Tensor<float>.SoftmaxMaskedFloatSpan(scores, mask, actual, rows, block, simd);
        Assert.True(expected.AsSpan().SequenceEqual(actual),
            $"masked diverges bitwise on {rows}x{block} simd={simd}.");
    }

    static float[] KeepMask(int block) { return new float[block]; }

    static float[] BanMask(int block)
    {
        var m = new float[block];
        for (int i = 0; i < m.Length; i++) m[i] = float.NegativeInfinity;
        return m;
    }

    static float[] MixedMask(int block, int seed)
    {
        var rnd = new Random(seed);
        var m = new float[block];
        for (int i = 0; i < m.Length; i++) m[i] = (rnd.NextDouble() < 0.3) ? float.NegativeInfinity : 0f;
        m[0] = 0f;
        return m;
    }

    [Fact]
    public void MaskedMatchesUnfusedBitwise()
    {
        foreach (int block in new[] { 8, 30, 128 })
        {
            MaskedEqual(3, block, KeepMask(block), 5000 + block, true);
            MaskedEqual(3, block, BanMask(block), 5100 + block, true);
            MaskedEqual(3, block, MixedMask(block, 5200 + block), 5200 + block, true);
        }
    }

    [Fact]
    public void MaskedMatchesUnfusedScalarMode()
    {
        foreach (int block in new[] { 7, 8, 9, 30, 129 })
            MaskedEqual(2, block, MixedMask(block, 5300 + block), 5300 + block, false);
    }

    [Fact]
    public void MaskedMatchesUnfusedWidthsAndZeroMix()
    {
        var mask = new float[64];
        for (int i = 0; i < mask.Length; i++) mask[i] = (i % 3 == 0) ? float.NegativeInfinity : ((i % 2 == 0) ? 0f : float.NegativeZero);
        mask[0] = 0f;
        MaskedEqual(1, 64, mask, 5400, true);
        MaskedEqual(5, 8, MixedMask(8, 5401), 5401, true);
    }
}
