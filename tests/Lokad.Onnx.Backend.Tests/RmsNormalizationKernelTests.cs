namespace Lokad.Onnx.Backend.Tests;

public class RmsNormalizationKernelTests
{
    static Tensor<float> Value(OpResult result)
    {
        Assert.Equal(OpStatus.Success, result.Status);
        return (Tensor<float>)Assert.Single(result.Outputs!);
    }

    static Tensor<float> Scalar(float value) => new DenseTensor<float>(new Memory<float>(new[] { value }), Array.Empty<int>());

    static float[] IndependentPrimitives(float[] values, float[] scale, int rows, int width, float epsilon, bool simd)
    {
        var options = new ExecutionOptions(OptimizationMode.Memory, new TensorExecutionOptions(simd, false, 1));
        var input = new DenseTensor<float>(new Memory<float>(values), new[] { rows, width });
        var gamma = DenseTensor<float>.OfValues(scale);
        var squared = Value(CPUExecutionProvider.Mul(input, input, options, null));
        var mean = Value(CPUExecutionProvider.ReduceMean(squared, DenseTensor<long>.OfValues(new[] { -1L }), 1, 0, options));
        var added = Value(CPUExecutionProvider.Add(mean, Scalar(epsilon), options, null));
        var inverse = Value(CPUExecutionProvider.Pow(added, Scalar(-.5f), options));
        var normalized = Value(CPUExecutionProvider.Mul(input, inverse, options, null));
        return Value(CPUExecutionProvider.Mul(normalized, gamma, options, null)).ToArray();
    }

    static int[] Bits(IEnumerable<float> values) => values.Select(BitConverter.SingleToInt32Bits).ToArray();

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void EveryRowAndColumnTailMatchesSixIndependentPrimitives(int kind)
    {
        var random = new Random(6413);
        float[] magnitudes = { 1e10f, 1f, -1e10f, .125f, -.25f, 1e-20f, -0f, float.Epsilon, MathF.BitIncrement(1f) };
        float[] extremes = { float.MaxValue, float.MinValue, float.Epsilon, -float.Epsilon, 0f, -0f };
        foreach (int rows in new[] { 1, 3, 4, 5, 8, 9, 16, 17, 19, 20 })
        foreach (int width in new[] { 1, 7, 8, 15, 16, 31, 32, 33, 34, 35, 128, 256, 512, 513 })
        foreach (float epsilon in new[] { 1e-6f, float.Epsilon, 1f })
        foreach (bool simd in new[] { false, true })
        {
            var input = new float[rows * width];
            for (int i = 0; i < input.Length; i++)
                input[i] = kind == 0 ? (float)(random.NextDouble() * 10 - 5)
                    : kind == 1 ? magnitudes[i % magnitudes.Length]
                    : kind == 2 ? extremes[i % extremes.Length] : (i % 3 - 1) * 1e-20f;
            var scale = Enumerable.Range(0, width).Select(i => i % 7 == 0 ? -0f : (i % 11 - 5) * .125f).ToArray();
            var before = Bits(input);
            var expected = IndependentPrimitives(input, scale, rows, width, epsilon, simd);
            var guarded = Enumerable.Repeat(123.25f, input.Length + 6).ToArray();
            Assert.True(RmsNormalizationKernel.TryNormalize(input, scale, guarded.AsSpan(3, input.Length), width, epsilon, -.5f, simd));
            Assert.Equal(Bits(expected), Bits(guarded.AsSpan(3, input.Length).ToArray()));
            Assert.Equal(before, Bits(input));
            Assert.All(guarded[..3], x => Assert.Equal(123.25f, x));
            Assert.All(guarded[(input.Length + 3)..], x => Assert.Equal(123.25f, x));
        }
    }

    [Fact]
    public void DeclinedControlsAndNonfiniteValuesDoNotTouchDestination()
    {
        foreach (bool simd in new[] { false, true })
        {
            var input = Enumerable.Repeat(1f, 128).ToArray();
            var scale = Enumerable.Repeat(1f, 32).ToArray();
            var destination = Enumerable.Repeat(77f, 128).ToArray();
            foreach (float epsilon in new[] { 0f, -0f, -1f, float.NaN, float.PositiveInfinity })
                Assert.False(RmsNormalizationKernel.TryNormalize(input, scale, destination, 32, epsilon, -.5f, simd));
            foreach (float exponent in new[] { 0f, .5f, -1f, MathF.BitIncrement(-.5f), float.NaN })
                Assert.False(RmsNormalizationKernel.TryNormalize(input, scale, destination, 32, 1e-6f, exponent, simd));
            foreach (float exceptional in new[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity, BitConverter.Int32BitsToSingle(0x7fa12345) })
            {
                input[127] = exceptional;
                Assert.False(RmsNormalizationKernel.TryNormalize(input, scale, destination, 32, 1e-6f, -.5f, simd));
                input[127] = 1f;
                scale[31] = exceptional;
                Assert.False(RmsNormalizationKernel.TryNormalize(input, scale, destination, 32, 1e-6f, -.5f, simd));
                scale[31] = 1f;
            }
            Assert.All(destination, x => Assert.Equal(77f, x));
            var before = Bits(input);
            Assert.False(RmsNormalizationKernel.TryNormalize(input, scale, input, 32, 1e-6f, -.5f, simd));
            Assert.Equal(before, Bits(input));
        }
    }

    [Fact]
    public void InvalidGeometryAndAliasingAreRefusedBeforeWriting()
    {
        var input = new float[64];
        var scale = new float[32];
        var output = Enumerable.Repeat(19f, 64).ToArray();
        Assert.False(RmsNormalizationKernel.TryNormalize(input, scale, output, 0, 1e-6f, -.5f, true));
        Assert.False(RmsNormalizationKernel.TryNormalize(input, scale, output, 31, 1e-6f, -.5f, true));
        Assert.False(RmsNormalizationKernel.TryNormalize(input, scale.AsSpan(0, 31), output, 32, 1e-6f, -.5f, true));
        Assert.False(RmsNormalizationKernel.TryNormalize(input, scale, output.AsSpan(0, 63), 32, 1e-6f, -.5f, true));
        Assert.False(RmsNormalizationKernel.TryNormalize(Array.Empty<float>(), scale, Array.Empty<float>(), 32, 1e-6f, -.5f, true));
        Assert.False(RmsNormalizationKernel.TryNormalize(input, output.AsSpan(0, 32), output, 32, 1e-6f, -.5f, true));
        Assert.All(output, x => Assert.Equal(19f, x));
    }
}
