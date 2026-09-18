using System.Numerics;

namespace Lokad.Onnx.Backend.Tests;

public class BiasGeluInlineTests
{
    static void SameBits(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(actual[i]))
                Assert.Fail($"Lane {i}: expected {BitConverter.SingleToInt32Bits(expected[i]):X8}, actual {BitConverter.SingleToInt32Bits(actual[i]):X8}");
    }

    [Fact]
    public void InlinedErfPreservesRandomBitsAndSplitBoundaries()
    {
        var random = new Random(20260918);
        var x = new float[Vector<float>.Count];
        var expected = new float[x.Length];
        var actual = new float[x.Length];
        float[] edges = { -0f, 0f, .921875f, MathF.BitIncrement(.921875f), MathF.BitDecrement(.921875f),
            3.925f, MathF.BitIncrement(3.925f), MathF.BitDecrement(3.925f), float.Epsilon, float.MaxValue,
            float.PositiveInfinity, float.NegativeInfinity, float.NaN, BitConverter.Int32BitsToSingle(0x7FA12345) };
        for (int group = 0; group < 262144; group++)
        {
            for (int i = 0; i < x.Length; i++) x[i] = group < edges.Length * 2
                ? edges[(group / 2 + i) % edges.Length] * (group % 2 == 0 ? 1 : -1)
                : group % 2 == 0 ? BitConverter.Int32BitsToSingle((int)random.NextInt64(int.MinValue, (long)int.MaxValue + 1))
                : (float)(random.NextDouble() * 10 - 5);
            var vector = new Vector<float>(x);
            MathOps.ErfVector(vector).CopyTo(expected);
            MathOps.ErfVectorInline(vector).CopyTo(actual);
            SameBits(expected, actual);
        }
    }

    [Theory]
    [InlineData(384, 8, 0)]
    [InlineData(1536, 30, 0)]
    [InlineData(1536, 128, 0)]
    [InlineData(1536, 512, 0)]
    [InlineData(16, 3, 1)]
    [InlineData(24, 5, 7)]
    [InlineData(8, 3, 3)]
    [InlineData(1, 2, 1)]
    [InlineData(3, 8, 0)]
    [InlineData(7, 9, 5)]
    [InlineData(384, 0, 0)]
    public void KernelPreservesReferenceAcrossShapesOffsetsWrapsTailsAndInPlace(int width, int rows, int tail)
    {
        var random = new Random(width * 11 + rows + tail);
        int length = width * rows + tail;
        var data = Enumerable.Range(0, length + 6).Select(_ => (float)(random.NextDouble() * 12 - 6)).ToArray();
        var bias = Enumerable.Range(0, width + 4).Select(_ => (float)(random.NextDouble() * 4 - 2)).ToArray();
        var before = data.ToArray();
        var biasBefore = bias.ToArray();
        var reference = new float[length];
        var output = Enumerable.Repeat(731f, length + 8).ToArray();
        Tensor<float>.BiasGeluSpanFloatPtr4x(data.AsSpan(3, length), bias.AsSpan(2, width), reference);
        Tensor<float>.BiasGeluSpanFloatInline(data.AsSpan(3, length), bias.AsSpan(2, width), output.AsSpan(4, length));
        SameBits(reference, output.AsSpan(4, length));
        SameBits(before, data); SameBits(biasBefore, bias);
        Assert.All(output.Take(4).Concat(output.Skip(length + 4)), value => Assert.Equal(731f, value));
        Tensor<float>.BiasGeluSpanFloatInline(data.AsSpan(3, length), bias.AsSpan(2, width), data.AsSpan(3, length));
        SameBits(reference, data.AsSpan(3, length));
        SameBits(before.AsSpan(0, 3), data.AsSpan(0, 3));
        SameBits(before.AsSpan(length + 3), data.AsSpan(length + 3));
    }

    [Fact]
    public void KernelPreservesExceptionalInputAndBiasPayloads()
    {
        var special = new[] { float.PositiveInfinity, float.NegativeInfinity, float.NaN, -0f, 0f,
            BitConverter.Int32BitsToSingle(0x7FA12345), BitConverter.Int32BitsToSingle(unchecked((int)0xFFC12345)),
            float.Epsilon, -float.Epsilon, float.MaxValue, float.MinValue, .921875f, 3.925f };
        int width = Vector<float>.Count * 3;
        var data = Enumerable.Range(0, width * 5 + 3).Select(i => special[i % special.Length]).ToArray();
        var bias = Enumerable.Range(0, width).Select(i => special[(i * 7) % special.Length]).ToArray();
        var expected = new float[data.Length];
        var actual = new float[data.Length];
        Tensor<float>.BiasGeluSpanFloatPtr4x(data, bias, expected);
        Tensor<float>.BiasGeluSpanFloatInline(data, bias, actual);
        SameBits(expected, actual);
    }
}
