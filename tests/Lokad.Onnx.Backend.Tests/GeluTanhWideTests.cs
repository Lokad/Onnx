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
