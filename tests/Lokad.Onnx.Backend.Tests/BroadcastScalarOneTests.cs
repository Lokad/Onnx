namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Single-element broadcast operands take the scalar tier with bit-wise
/// identical values to an independent naive loop: broadcasting only selects
/// which elements combine, and each combination is computed once identically.
/// Covers the GPT-2 Mul_1 pattern (large tensor times rank-1 length-1),
/// mirror shapes, scalar fallback modes and exceptional payloads.
/// </summary>
public class BroadcastScalarOneTests
{
    static DenseTensor<float> Fill(int[] dims, int seed)
    {
        var rnd = new Random(seed);
        int total = 1;
        foreach (var q in dims) total *= q;
        var data = new float[total];
        for (int i = 0; i < total; i++)
        {
            float v = (float)(rnd.NextDouble() * 4 - 2);
            if (seed % 3 == 0 && i % 29 == 0) v = float.NaN;
            data[i] = v;
        }
        return new DenseTensor<float>(data, (int[])dims.Clone());
    }

    static void AgreeMul(int[] bigDims, int[] oneDims, int seed)
    {
        var big = Fill(bigDims, seed);
        var one = Fill(oneDims, seed + 1);
        var got = big.BroadcastApply<MultiplyBroadcast<float>>(one, TensorExecutionOptions.Auto).ToDenseTensor();
        var naive = new float[big.Length];
        var bs = big.Buffer.Span;
        float s = one.Buffer.Span[0];
        for (int i = 0; i < naive.Length; i++) naive[i] = bs[i] * s;
        Assert.True(got.Buffer.Span.SequenceEqual(naive), $"mul diverged on [{string.Join(",", bigDims)}]x[{string.Join(",", oneDims)}].");
    }

    static void AgreeAdd(int[] bigDims, int[] oneDims, int seed)
    {
        var big = Fill(bigDims, seed);
        var one = Fill(oneDims, seed + 1);
        var got = big.BroadcastApply<AddBroadcast<float>>(one, TensorExecutionOptions.Auto).ToDenseTensor();
        var naive = new float[big.Length];
        var bs = big.Buffer.Span;
        float s = one.Buffer.Span[0];
        for (int i = 0; i < naive.Length; i++) naive[i] = bs[i] + s;
        Assert.True(got.Buffer.Span.SequenceEqual(naive), $"add diverged on [{string.Join(",", bigDims)}]x[{string.Join(",", oneDims)}].");
    }

    [Fact]
    public void ScalarOneRightOperandMatchesNaive()
    {
        AgreeMul(new[] { 1, 12, 64, 513 }, new[] { 1 }, 91);
        AgreeMul(new[] { 1, 30, 384 }, new[] { 1 }, 92);
        AgreeAdd(new[] { 1, 12, 64, 513 }, new[] { 1 }, 93);
    }

    [Fact]
    public void ScalarOneLeftOperandMatchesNaive()
    {
        var one = Fill(new[] { 1 }, 94);
        var big = Fill(new[] { 1, 12, 64, 513 }, 95);
        var got = one.BroadcastApply<MultiplyBroadcast<float>>(big, TensorExecutionOptions.Auto).ToDenseTensor();
        var naive = new float[big.Length];
        var bs = big.Buffer.Span;
        float s = one.Buffer.Span[0];
        for (int i = 0; i < naive.Length; i++) naive[i] = s * bs[i];
        Assert.True(got.Buffer.Span.SequenceEqual(naive), "left-operand mirror diverged.");
    }

    [Fact]
    public void ScalarOneHigherRankSingletonMatchesNaive()
    {
        AgreeMul(new[] { 2, 30, 32 }, new[] { 1, 1, 1 }, 96);
        AgreeAdd(new[] { 8, 1536 }, new[] { 1 }, 97);
    }

    [Fact]
    public void ScalarOneScalarModeMatchesNaive()
    {
        var big = Fill(new[] { 1, 12, 64, 513 }, 98);
        var one = Fill(new[] { 1 }, 99);
        var got = big.BroadcastApply<MultiplyBroadcast<float>>(one, TensorExecutionOptions.Scalar).ToDenseTensor();
        var naive = new float[big.Length];
        var bs = big.Buffer.Span;
        float s = one.Buffer.Span[0];
        for (int i = 0; i < naive.Length; i++) naive[i] = bs[i] * s;
        Assert.True(got.Buffer.Span.SequenceEqual(naive), "scalar mode diverged.");
    }
}
