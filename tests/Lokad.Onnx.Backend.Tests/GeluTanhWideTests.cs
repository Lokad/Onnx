namespace Lokad.Onnx.Backend.Tests;

using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

public class GeluTanhWideTests
{
    static float[] Inputs(int length)
    {
        var random = new Random(3971);
        var values = new float[length];
        for (int i = 0; i < length; i++) values[i] = (float)(random.NextDouble() * 200 - 100);
        float[] edges = { 0f, -0f, float.Epsilon, -float.Epsilon, float.MaxValue, float.MinValue,
            float.PositiveInfinity, float.NegativeInfinity, float.NaN,
            BitConverter.Int32BitsToSingle(0x7fa12345), BitConverter.Int32BitsToSingle(unchecked((int)0xffc54321)),
            -88.722839f, MathF.BitIncrement(-88.722839f), MathF.BitDecrement(-88.722839f),
            88.722839f, MathF.BitIncrement(88.722839f), MathF.BitDecrement(88.722839f),
            1e-30f, -1e-30f, 10f, -10f };
        // Every lane sees the exceptional cases across the longer arrays.
        for (int i = 0; i < length; i += 31) values[i] = edges[i / 31 % edges.Length];
        return values;
    }

    static void EqualBits(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                $"Lane {i}: {BitConverter.SingleToInt32Bits(expected[i]):X8} vs {BitConverter.SingleToInt32Bits(actual[i]):X8}");
    }

    [Fact]
    public void HornerExponentialPreservesExistingVectorBits()
    {
        var inputs = Inputs(16384);
        var expected = new float[inputs.Length];
        var actual = new float[inputs.Length];
        var source = MemoryMarshal.Cast<float, Vector<float>>(inputs);
        var narrow = MemoryMarshal.Cast<float, Vector<float>>(expected.AsSpan());
        for (int i = 0; i < source.Length; i++) narrow[i] = MathOps.ExpVector(source[i]);
        var wideSource = MemoryMarshal.Cast<float, Vector512<float>>(inputs);
        var wide = MemoryMarshal.Cast<float, Vector512<float>>(actual.AsSpan());
        for (int i = 0; i < wideSource.Length; i++) wide[i] = MathOps.ExpVectorHorner512(wideSource[i]);
        EqualBits(expected, actual);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(15)]
    [InlineData(16)]
    [InlineData(17)]
    [InlineData(23)]
    [InlineData(24)]
    [InlineData(31)]
    [InlineData(32)]
    [InlineData(33)]
    [InlineData(767)]
    [InlineData(768)]
    [InlineData(769)]
    [InlineData(3072)]
    public void WideKernelPreservesBitsTailsAndDestinationBounds(int length)
    {
        var inputs = Inputs(length);
        var expected = new float[length];
        Tensor<float>.GeluTanhSpanFloatReference(inputs, expected);
        var guarded = Enumerable.Repeat(123.25f, length + 6).ToArray();
        Tensor<float>.GeluTanhSpanFloatWide(inputs, guarded.AsSpan(3, length));
        EqualBits(expected, guarded.AsSpan(3, length));
        Assert.All(guarded[..3], x => Assert.Equal(123.25f, x));
        Assert.All(guarded[(length + 3)..], x => Assert.Equal(123.25f, x));
        Tensor<float>.GeluTanhSpanFloatWide(inputs, inputs);
        EqualBits(expected, inputs);
    }

    [Fact]
    public void BoundedHornerPreservesGeneralExponentialWithinItsContract()
    {
        var input = new float[16384];
        for (int i = 0; i < input.Length; i++) input[i] = (i % 4097 - 2048) * (20f / 2048f);
        input[0] = -20f;
        input[1] = MathF.BitIncrement(-20f);
        input[2] = -0f;
        input[3] = MathF.BitDecrement(20f);
        input[4] = 20f;
        var expected = new float[input.Length];
        var actual = new float[input.Length];
        var values = MemoryMarshal.Cast<float, Vector512<float>>(input);
        var oldOutput = MemoryMarshal.Cast<float, Vector512<float>>(expected.AsSpan());
        var newOutput = MemoryMarshal.Cast<float, Vector512<float>>(actual.AsSpan());
        for (int i = 0; i < values.Length; i++)
        {
            oldOutput[i] = MathOps.ExpVectorHorner512(values[i]);
            newOutput[i] = MathOps.ExpVectorHorner512Gelu(values[i]);
        }
        EqualBits(expected, actual);
    }

    [Fact]
    public void SaturationBoundaryPreservesReferenceBits()
    {
        static float Argument(float value)
        {
            float cubic = .044715f * value * value * value;
            float t = .7978846f * (value + cubic);
            return t + t;
        }
        int low = BitConverter.SingleToInt32Bits(4f), high = BitConverter.SingleToInt32Bits(8f);
        while (low < high)
        {
            int middle = low + (high - low) / 2;
            if (Argument(BitConverter.Int32BitsToSingle(middle)) < 20f) low = middle + 1;
            else high = middle;
        }
        var values = new float[16384];
        for (int i = 0; i < 8192; i++)
        {
            values[i] = BitConverter.Int32BitsToSingle(low + i - 4096);
            values[8192 + i] = -values[i];
        }
        var expected = new float[values.Length];
        var actual = new float[values.Length];
        Tensor<float>.GeluTanhSpanFloatReference(values, expected);
        Tensor<float>.GeluTanhSpanFloatWide(values, actual);
        EqualBits(expected, actual);
    }

    [Fact]
    public void EveryFiniteExponentBandPreservesReferenceBits()
    {
        var values = new float[255 * 64 * 2];
        int index = 0;
        for (uint exponent = 0; exponent < 255; exponent++)
        for (uint mantissa = 0; mantissa < 64; mantissa++)
        foreach (uint sign in new[] { 0u, 0x80000000u })
            values[index++] = BitConverter.Int32BitsToSingle(unchecked((int)(sign | exponent << 23 | mantissa << 17)));
        var expected = new float[values.Length];
        var actual = new float[values.Length];
        Tensor<float>.GeluTanhSpanFloatReference(values, expected);
        Tensor<float>.GeluTanhSpanFloatWide(values, actual);
        EqualBits(expected, actual);
    }

    [Fact]
    public void DispatchedKernelMatchesReferenceAndOwnsOutputs()
    {
        var values = Inputs(4097);
        var expected = new float[values.Length];
        Tensor<float>.GeluTanhSpanFloatReference(values, expected);
        var input = DenseTensor<float>.OfValues(values);
        var output = Tensor<float>.GeluTanh(input, TensorExecutionOptions.Auto);
        EqualBits(expected, output.ToArray());
        output[0] = 42;
        EqualBits(values, input.ToArray());
    }
}
