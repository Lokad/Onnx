using System.Runtime.InteropServices;
using System.Security.Cryptography;

namespace Lokad.Onnx.Backend.Tests;

// Same helpers and fact bodies as the validated LSTM layout contracts.
// Captured-model and exact-VM identity checks stay in their original campaign.
[Collection("NonParallelAllocation")]
public class PreparedLstmProjectionTests
{
    const int H = 640, Outputs = 4 * H;
    static string Hash(ReadOnlySpan<float> values) => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(values)));
    static void Equal(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        Assert.Equal(a.Length, b.Length);
        for (int i = 0; i < a.Length; i++)
        {
            // NaN payload/sign selection is not a floating-point arithmetic
            // contract: the JIT can commute operands of multiplication/addition.
            // Every other result, including signed zero and infinity, stays exact.
            if (float.IsNaN(a[i])) Assert.True(float.IsNaN(b[i]), $"Expected NaN at {i}");
            else Assert.Equal(BitConverter.SingleToInt32Bits(a[i]), BitConverter.SingleToInt32Bits(b[i]));
        }
    }
    static float[] Values(int count) => Enumerable.Range(0, count).Select(i => (i % 113 - 56) * .015625f).ToArray();
    static float[] Pack(ReadOnlySpan<float> flat, int n, int columns)
    {
        var result = new float[flat.Length]; int block = PreparedLstmProjection.ColumnsPerBlock;
        for (int o = 0; o < columns; o++)
        for (int k = 0; k < n; k++)
        {
            int group = o / block * block, width = Math.Min(block, columns - group);
            result[group * n + k * width + o - group] = flat[k * columns + o];
        }
        return result;
    }
    static void Raw(float[] input, float[] flat, int columns)
    {
        var packed = Pack(flat, input.Length, columns);
        string inputHash = Hash(input), packedHash = Hash(packed);
        const float Guard = 192837f;
        var expected = Enumerable.Repeat(Guard, columns + 2).ToArray();
        var actual = (float[])expected.Clone();
        CPUExecutionProvider.LstmProjectOrdered(input, flat, expected.AsSpan(1, columns));
        PreparedLstmProjection.Multiply(input, packed, actual.AsSpan(1, columns));
        Equal(expected, actual);
        long start = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 8; i++) PreparedLstmProjection.Multiply(input, packed, actual.AsSpan(1, columns));
        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - start);
        Equal(expected, actual); Assert.Equal(inputHash, Hash(input)); Assert.Equal(packedHash, Hash(packed));
    }

    [Fact]
    public void BoundariesRetainBitsGuardsAndZeroAllocation()
    {
        foreach (int n in new[] { 0, 1, 3, 17 })
        foreach (int columns in new[] { 0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 2560 })
            Raw(Values(n), Values(n * columns), columns);
        Raw(Values(H), Values(H * Outputs), Outputs);
    }

    [Fact]
    public void ExceptionalValuesRetainClassificationAndFiniteBits()
    {
        float[] values = [float.NegativeInfinity, float.PositiveInfinity, float.MaxValue, -float.MaxValue,
            float.Epsilon, -float.Epsilon, -0f, 0f, BitConverter.Int32BitsToSingle(0x7fc12345),
            BitConverter.Int32BitsToSingle(unchecked((int)0xffc54321)), 1f, -1f];
        for (int shift = 0; shift < values.Length; shift++)
            Raw(Enumerable.Range(0, 3).Select(i => values[(i + shift) % values.Length]).ToArray(),
                Enumerable.Range(0, 3 * 71).Select(i => values[(i + shift + 4) % values.Length]).ToArray(), 71);
    }

    [Fact]
    public void ComparisonChecksZeroInfinityAndNaNClassification()
    {
        Equal([BitConverter.Int32BitsToSingle(0x7fc12345)], [BitConverter.Int32BitsToSingle(unchecked((int)0xffc54321))]);
        Assert.ThrowsAny<Xunit.Sdk.XunitException>(() => Equal([float.NaN], [0f]));
        Assert.ThrowsAny<Xunit.Sdk.XunitException>(() => Equal([0f], [float.NaN]));
        Assert.ThrowsAny<Xunit.Sdk.XunitException>(() => Equal([0f], [-0f]));
        Assert.ThrowsAny<Xunit.Sdk.XunitException>(() => Equal([float.PositiveInfinity], [float.NegativeInfinity]));
        Assert.ThrowsAny<Xunit.Sdk.XunitException>(() => Equal([1f], [MathF.BitIncrement(1f)]));
    }

    [Theory]
    [InlineData(0, 65)]
    [InlineData(37, 65)]
    [InlineData(640, 2560)]
    public void WideProjectionPreservesAscendingSeparateArithmetic(int inputs, int columns)
    {
        int block = PreparedLstmProjection.ColumnsPerBlock;
        var random = new Random(42);
        var x = Enumerable.Range(0, inputs).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
        var weights = Enumerable.Range(0, inputs * columns).Select(_ => (float)(random.NextDouble() * .2 - .1)).ToArray();
        var panel = new float[weights.Length]; var expected = new float[columns];
        for (int col = 0; col < columns; col++)
        {
            int group = col / block * block, count = Math.Min(block, columns - group);
            float sum = 0;
            for (int k = 0; k < inputs; k++)
            {
                float weight = weights[k * columns + col];
                panel[group * inputs + k * count + col - group] = weight;
                float product = (float)((double)x[k] * weight);
                sum = (float)((double)sum + product);
            }
            expected[col] = sum;
        }
        var savedX = x.ToArray(); var savedPanel = panel.ToArray();
        foreach (bool wide in new[] { true, false })
        {
            var actual = new float[columns];
            PreparedLstmProjection.Multiply(x, panel, actual, wide);
            for (int i = 0; i < columns; i++)
                Assert.Equal(BitConverter.SingleToInt32Bits(expected[i]), BitConverter.SingleToInt32Bits(actual[i]));
        }
        Assert.Equal(savedX, x); Assert.Equal(savedPanel, panel);
    }
}
