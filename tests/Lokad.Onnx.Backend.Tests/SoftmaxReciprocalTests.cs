using System.Numerics;

namespace Lokad.Onnx.Backend.Tests;

public class SoftmaxReciprocalTests
{
    static void Run(bool candidate, bool masked, float[] x, float[] mask, Span<float> output, int rows, int block, bool simd)
    {
        if (candidate)
        {
            if (masked) Tensor<float>.SoftmaxMaskedFloatSpanPtrReciprocal(x, mask, output, rows, block, simd);
            else Tensor<float>.SoftmaxContiguousFloatSpanPtrReciprocal(x, output, rows, block, simd);
        }
        else
        {
            if (masked) Tensor<float>.SoftmaxMaskedFloatSpanPtr(x, mask, output, rows, block, simd);
            else Tensor<float>.SoftmaxContiguousFloatSpanPtr(x, output, rows, block, simd);
        }
    }

    static void Check(float[] x, float[] mask, int rows, int block, bool masked, bool simd)
    {
        var beforeX = x.Select(BitConverter.SingleToInt32Bits).ToArray();
        var beforeMask = mask.Select(BitConverter.SingleToInt32Bits).ToArray();
        var reference = new float[x.Length];
        var guarded = Enumerable.Repeat(1234567f, x.Length + 10).ToArray();
        var repeated = new float[x.Length];
        Run(false, masked, x, mask, reference, rows, block, simd);
        Run(true, masked, x, mask, guarded.AsSpan(5, x.Length), rows, block, simd);
        Run(true, masked, x, mask, repeated, rows, block, simd);
        Assert.All(guarded.Take(5).Concat(guarded.Skip(5 + x.Length)), v => Assert.Equal(1234567f, v));
        Assert.Equal(beforeX, x.Select(BitConverter.SingleToInt32Bits));
        Assert.Equal(beforeMask, mask.Select(BitConverter.SingleToInt32Bits));
        for (int i = 0; i < x.Length; i++)
        {
            float got = guarded[5 + i], want = reference[i];
            Assert.Equal(BitConverter.SingleToInt32Bits(got), BitConverter.SingleToInt32Bits(repeated[i]));
            if (!simd || !Vector.IsHardwareAccelerated || !float.IsFinite(want))
                Assert.Equal(BitConverter.SingleToInt32Bits(want), BitConverter.SingleToInt32Bits(got));
            else
            {
                Assert.True(float.IsFinite(got) && got >= 0f);
                Assert.InRange(Math.Abs((double)got - want), 0, 2e-7);
                Assert.InRange(Math.Abs((long)BitConverter.SingleToInt32Bits(got) - BitConverter.SingleToInt32Bits(want)), 0, 2);
            }
        }
        for (int row = 0; row < rows && block > 0; row++)
        {
            var values = Enumerable.Range(0, block).Select(i => (double)(x[row * block + i] + (masked ? mask[i] : 0f))).ToArray();
            if (values.Any(v => double.IsNaN(v) || double.IsPositiveInfinity(v)) || values.All(double.IsNegativeInfinity)) continue;
            double max = values.Max();
            var exp = values.Select(v => Math.Exp(v - max)).ToArray();
            double sum = exp.Sum();
            double actualSum = 0;
            for (int i = 0; i < block; i++)
            {
                double actual = guarded[5 + row * block + i];
                Assert.InRange(Math.Abs(actual - exp[i] / sum), 0, 1e-6);
                actualSum += actual;
            }
            double referenceSum = reference.Skip(row * block).Take(block).Sum(v => (double)v);
            // The existing float reduction already drifts on very long rows.
            // Bound the new normalization's total drift independently of that
            // unchanged error; per-value double agreement is checked above.
            Assert.InRange(Math.Abs(actualSum - referenceSum), 0, 2e-7);
        }
    }

    [Theory]
    [InlineData(0, 8)]
    [InlineData(3, 0)]
    [InlineData(1, 1)]
    [InlineData(3, 7)]
    [InlineData(4, 8)]
    [InlineData(5, 9)]
    [InlineData(7, 17)]
    [InlineData(12, 30)]
    [InlineData(13, 128)]
    [InlineData(5, 201)]
    [InlineData(7, 512)]
    [InlineData(3, 1500)]
    [InlineData(1, 16385)]
    public void FiniteRowsMeetRoundingAndIndependentDoubleBounds(int rows, int block)
    {
        var random = new Random(rows * 313 + block);
        foreach (float scale in new[] { 0f, 0.01f, 1f, 10f, 100f })
        {
            var x = Enumerable.Range(0, rows * block).Select(_ => (float)(random.NextDouble() * 2 - 1) * scale).ToArray();
            var mask = Enumerable.Range(0, block).Select(i => i > 0 && i % 3 != 0 ? -float.MaxValue : 0f).ToArray();
            foreach (bool simd in new[] { false, true })
                foreach (bool masked in new[] { false, true }) Check(x, mask, rows, block, masked, simd);
        }
    }

    [Theory]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(9)]
    [InlineData(30)]
    [InlineData(128)]
    public void ExceptionalRowsPreserveReferenceBits(int block)
    {
        foreach (float special in new[] { float.NaN, BitConverter.Int32BitsToSingle(0x7FA12345), float.PositiveInfinity, float.NegativeInfinity,
            float.MaxValue, -float.MaxValue, 0f, -0f, float.Epsilon, -float.Epsilon })
        {
            var x = new float[3 * block];
            x[0] = special; x[2 * block - 1] = special;
            Array.Fill(x, special, 2 * block, block);
            foreach (float maskValue in new[] { 0f, -float.MaxValue, float.NegativeInfinity, float.NaN })
            {
                var mask = Enumerable.Repeat(maskValue, block).ToArray();
                foreach (bool simd in new[] { false, true })
                    foreach (bool masked in new[] { false, true }) Check(x, mask, 3, block, masked, simd);
            }
        }
    }

    [Theory]
    [InlineData(8)]
    [InlineData(30)]
    [InlineData(128)]
    [InlineData(512)]
    public void SharedMaskAndMaterializedAddAgreeExactly(int block)
    {
        var random = new Random(block);
        var x = Enumerable.Range(0, 5 * block).Select(_ => (float)(random.NextDouble() * 12 - 6)).ToArray();
        var mask = Enumerable.Range(0, block).Select(i => i % 4 == 0 ? 0f : -10000f).ToArray();
        var added = x.Select((v, i) => v + mask[i % block]).ToArray();
        var plain = new float[x.Length]; var masked = new float[x.Length];
        Tensor<float>.SoftmaxContiguousFloatSpanPtrReciprocal(added, plain, 5, block, true);
        Tensor<float>.SoftmaxMaskedFloatSpanPtrReciprocal(x, mask, masked, 5, block, true);
        Assert.Equal(plain.Select(BitConverter.SingleToInt32Bits), masked.Select(BitConverter.SingleToInt32Bits));
    }
}
