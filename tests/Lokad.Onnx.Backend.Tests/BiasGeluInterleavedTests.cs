using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx.Backend.Tests;

public class BiasGeluInterleavedTests
{
    static void SameBits(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual) =>
        Assert.True(MemoryMarshal.AsBytes(expected).SequenceEqual(MemoryMarshal.AsBytes(actual)), "Float bits differ");

    [Theory]
    [InlineData(384, 8, 0)]
    [InlineData(1536, 30, 0)]
    [InlineData(1536, 128, 0)]
    [InlineData(1536, 512, 0)]
    [InlineData(32, 3, 0)]
    [InlineData(64, 5, 0)]
    [InlineData(16, 3, 0)]
    [InlineData(32, 3, 1)]
    [InlineData(64, 3, 7)]
    [InlineData(8, 3, 3)]
    [InlineData(1, 2, 1)]
    [InlineData(7, 9, 5)]
    [InlineData(384, 0, 0)]
    public void PreservesBitsOffsetsGuardsWrapsTailsAndInPlace(int width, int rows, int tail)
    {
        int length = width * rows + tail;
        var random = new Random(width * 11 + rows + tail);
        var input = Enumerable.Range(0, length + 8).Select(_ => random.NextSingle() * 18 - 9).ToArray();
        var bias = Enumerable.Range(0, width + 4).Select(_ => random.NextSingle() * 4 - 2).ToArray();
        var heldInput = input.ToArray();
        var heldBias = bias.ToArray();
        var expected = new float[length];
        var actual = Enumerable.Repeat(731f, length + 10).ToArray();
        Tensor<float>.BiasGeluSpanFloatInline(input.AsSpan(3, length), bias.AsSpan(2, width), expected);
        Tensor<float>.BiasGeluSpanFloatInterleaved(input.AsSpan(3, length), bias.AsSpan(2, width), actual.AsSpan(5, length));
        SameBits(expected, actual.AsSpan(5, length));
        SameBits(heldInput, input);
        SameBits(heldBias, bias);
        Assert.All(actual.Take(5).Concat(actual.Skip(length + 5)), value => Assert.Equal(731f, value));
        Tensor<float>.BiasGeluSpanFloatInterleaved(input.AsSpan(3, length), bias.AsSpan(2, width), input.AsSpan(3, length));
        SameBits(expected, input.AsSpan(3, length));
        SameBits(heldInput.AsSpan(0, 3), input.AsSpan(0, 3));
        SameBits(heldInput.AsSpan(length + 3), input.AsSpan(length + 3));
    }

    [Fact]
    public void PreservesRandomFloatBitsAndExceptionalBiasPayloads()
    {
        float[] edges = { 0f, -0f, float.Epsilon, -float.Epsilon, float.PositiveInfinity, float.NegativeInfinity,
            float.NaN, BitConverter.Int32BitsToSingle(0x7FA12345), BitConverter.Int32BitsToSingle(unchecked((int)0xFFA54321)),
            float.MaxValue, -float.MaxValue, .921875f, 3.925f };
        var input = new float[2048];
        var bias = new float[32];
        var expected = new float[input.Length];
        var actual = new float[input.Length];
        uint state = 314159265;
        for (int pass = 0; pass < 1024; pass++)
        {
            for (int i = 0; i < input.Length; i++)
            {
                state ^= state << 13; state ^= state >> 17; state ^= state << 5;
                input[i] = pass < edges.Length ? edges[(pass + i) % edges.Length] : BitConverter.Int32BitsToSingle(unchecked((int)state));
            }
            for (int i = 0; i < bias.Length; i++) bias[i] = pass < edges.Length ? edges[(pass * 7 + i) % edges.Length] : 0f;
            Tensor<float>.BiasGeluSpanFloatInline(input, bias, expected);
            Tensor<float>.BiasGeluSpanFloatInterleaved(input, bias, actual);
            SameBits(expected, actual);
        }
    }

    [Fact]
    public void PreservesErfSplitAndClampNeighborhoods()
    {
        foreach (float center in new[] { 0f, .921875f, -.921875f, 3.925f, -3.925f })
        for (int shift = -32; shift <= 32; shift++)
        {
            float value = center;
            for (int step = 0; step < Math.Abs(shift); step++) value = shift < 0 ? float.BitDecrement(value) : float.BitIncrement(value);
            var input = Enumerable.Repeat(value / .7071067811865476f, 64).ToArray();
            var expected = new float[input.Length];
            var actual = new float[input.Length];
            Tensor<float>.BiasGeluSpanFloatInline(input, new float[32], expected);
            Tensor<float>.BiasGeluSpanFloatInterleaved(input, new float[32], actual);
            SameBits(expected, actual);
        }
    }

    [Fact]
    public void TargetRouteRequiresTheQualifiedHardware()
    {
        var input = Enumerable.Range(0, 96).Select(i => i * .01f - .5f).ToArray();
        var bias = new float[32];
        var actual = Enumerable.Repeat(731f, input.Length).ToArray();
        bool used = Tensor<float>.TryBiasGeluSpanFloatInterleaved(input, bias, actual);
        Assert.Equal(Avx512F.IsSupported && Fma.IsSupported && Vector<float>.Count == 8, used);
        if (used)
        {
            var expected = new float[input.Length];
            Tensor<float>.BiasGeluSpanFloatInline(input, bias, expected);
            SameBits(expected, actual);
        }
        else Assert.All(actual, value => Assert.Equal(731f, value));
    }

    [Theory]
    [InlineData(0, 64, 64)]
    [InlineData(1, 64, 64)]
    [InlineData(16, 64, 64)]
    [InlineData(33, 66, 66)]
    [InlineData(32, 65, 65)]
    [InlineData(32, 64, 63)]
    public void TargetRouteDeclinesWithoutWriting(int width, int length, int outputLength)
    {
        var output = Enumerable.Repeat(731f, outputLength).ToArray();
        Assert.False(Tensor<float>.TryBiasGeluSpanFloatInterleaved(new float[length], new float[width], output));
        Assert.All(output, value => Assert.Equal(731f, value));
    }
}
