using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;

namespace Lokad.Onnx.Backend.Tests;

// Run this class in separate processes with the diagnostic switch off/on.
// Never mutate a process-wide switch while other xUnit classes are executing.
public class LayerNormWideOutputTests
{
    public static IEnumerable<object[]> Widths()
    {
        foreach (int width in new[] { 1, 7, 8, 9, 15, 16, 17, 23, 24, 31, 32, 33, 383, 384, 385, 1280, 1536 })
        foreach (bool bias in new[] { false, true })
            yield return new object[] { width, bias };
    }

    static int[] Bits(float[] values) => values.Select(BitConverter.SingleToInt32Bits).ToArray();

    static float[] Reference(float[] input, float[] scale, float[]? bias, float epsilon)
    {
        int width = scale.Length;
        var result = new float[input.Length];
        for (int start = 0; start < input.Length; start += width)
        {
            double mean = 0;
            for (int j = 0; j < width; j++) mean += input[start + j];
            mean /= width;
            double variance = 0;
            for (int j = 0; j < width; j++) variance += Math.Pow(input[start + j] - mean, 2);
            double denominator = Math.Sqrt(variance / width + epsilon);
            for (int j = 0; j < width; j++)
                result[start + j] = (float)((input[start + j] - mean) / denominator * scale[j] + (bias?[j] ?? 0f));
        }
        return result;
    }

    static void Agrees(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            if (float.IsNaN(expected[i])) Assert.True(float.IsNaN(actual[i]), $"Expected NaN at {i}");
            else if (float.IsInfinity(expected[i])) Assert.Equal(expected[i], actual[i]);
            else Assert.True(float.IsFinite(actual[i]) && Math.Abs((double)actual[i] - expected[i]) <= 1e-6 * Math.Max(1, Math.Abs((double)expected[i])),
                $"Element {i}: expected {expected[i]:R}, actual {actual[i]:R}");
        }
    }

    static (DenseTensor<float> Tensor, float[] Storage) Guarded(float[] values, int[] dimensions)
    {
        var storage = Enumerable.Repeat(-7654321f, values.Length + 6).ToArray();
        values.CopyTo(storage, 3);
        return (new DenseTensor<float>(storage.AsMemory(3, values.Length), dimensions), storage);
    }

    [Theory]
    [MemberData(nameof(Widths))]
    public void PublicEntriesPreserveStorageAndAgreeWithIndependentReference(int width, bool hasBias)
    {
        const float epsilon = 1e-5f;
        var values = Enumerable.Range(0, width * 3).Select(i => (i / width) switch
        {
            0 => (float)Math.Sin(i * .31) * 4f,
            1 => 1f + (i % 2 == 0 ? 1e-7f : -1e-7f),
            _ => 1e10f + (i % width) * 1e4f
        }).ToArray();
        var scales = Enumerable.Range(0, width).Select(i => (i % 7 - 3) * .25f).ToArray();
        var biases = Enumerable.Range(0, width).Select(i => (i % 5 - 2) * .125f).ToArray();
        var (x, xStore) = Guarded(values, new[] { 3, width });
        var (scale, sStore) = Guarded(scales, new[] { width });
        var (bias, bStore) = Guarded(biases, new[] { width });
        var (destination, dStore) = Guarded(new float[values.Length], new[] { 3, width });
        var xBefore = Bits(xStore); var sBefore = Bits(sStore); var bBefore = Bits(bStore);
        var expected = Reference(values, scales, hasBias ? biases : null, epsilon);

        var allocated = Tensor<float>.LayerNormalization(x, scale, hasBias ? bias : null, -1, epsilon);
        var held = Bits(allocated.ToArray());
        Agrees(expected, allocated.ToArray());
        Assert.Same(destination, Tensor<float>.LayerNormalization(x, scale, hasBias ? bias : null, destination, 1, epsilon));
        Assert.Equal(held, Bits(destination.ToArray()));
        Assert.Equal(xBefore, Bits(xStore)); Assert.Equal(sBefore, Bits(sStore)); Assert.Equal(bBefore, Bits(bStore));
        Assert.All(dStore.Take(3).Concat(dStore.TakeLast(3)), v => Assert.Equal(-7654321f, v));

        // Exact input/destination aliasing is supported; a held allocated result
        // must not be backed by either caller buffer or a later output.
        Assert.Same(x, Tensor<float>.LayerNormalization(x, scale, hasBias ? bias : null, x, -1, epsilon));
        Assert.Equal(held, Bits(x.ToArray()));
        Assert.All(xStore.Take(3).Concat(xStore.TakeLast(3)), v => Assert.Equal(-7654321f, v));
        Tensor<float>.LayerNormalization(x, scale, hasBias ? bias : null, destination, -1, epsilon);
        Assert.Equal(held, Bits(allocated.ToArray()));
        Assert.Equal(sBefore, Bits(sStore)); Assert.Equal(bBefore, Bits(bStore));
    }

    [Theory]
    [InlineData(0, 0, float.NaN)]
    [InlineData(0, 15, float.PositiveInfinity)]
    [InlineData(0, 32, float.NegativeInfinity)]
    [InlineData(1, 0, float.NaN)]
    [InlineData(1, 16, float.PositiveInfinity)]
    [InlineData(1, 32, float.NegativeInfinity)]
    [InlineData(2, 15, float.NaN)]
    [InlineData(2, 16, float.PositiveInfinity)]
    [InlineData(2, 32, float.NegativeInfinity)]
    public void ExceptionalValuesCrossWideVectorAndScalarBoundaries(int target, int position, float value)
    {
        var input = Enumerable.Range(0, 66).Select(i => i * .25f).ToArray();
        var scale = Enumerable.Repeat(1f, 33).ToArray();
        var bias = Enumerable.Repeat(.125f, 33).ToArray();
        new[] { input, scale, bias }[target][position] = value;
        var x = new DenseTensor<float>(input, new[] { 2, 33 });
        var s = new DenseTensor<float>(scale, new[] { 33 });
        var b = new DenseTensor<float>(bias, new[] { 33 });
        var expected = Reference(input, scale, bias, 1e-5f);
        Agrees(expected, Tensor<float>.LayerNormalization(x, s, b, -1, 1e-5f).ToArray());
        Tensor<float>.LayerNormalization(x, s, b, x, -1, 1e-5f);
        Agrees(expected, x.ToArray());
    }

    [Fact]
    public void NormalizedDimensionsCanSpanMultipleAxes()
    {
        var values = Enumerable.Range(0, 102).Select(i => (float)Math.Cos(i)).ToArray();
        var scales = Enumerable.Repeat(1.5f, 51).ToArray();
        var x = new DenseTensor<float>(values, new[] { 2, 3, 17 });
        var scale = new DenseTensor<float>(scales, new[] { 3, 17 });
        Agrees(Reference(values, scales, null, 1e-5f), Tensor<float>.LayerNormalization(x, scale, null, -2, 1e-5f).ToArray());
    }

    [Fact]
    public void InvalidParametersLeaveDestinationUntouched()
    {
        var x = new DenseTensor<float>(new[] { 2, 33 });
        var scale = new DenseTensor<float>(new[] { 33 });
        var (destination, storage) = Guarded(Enumerable.Repeat(42f, 66).ToArray(), new[] { 2, 33 });
        var before = Bits(storage);
        Assert.Throws<ArgumentException>(() => Tensor<float>.LayerNormalization(x, new DenseTensor<float>(32), null, destination, -1, 1e-5f));
        Assert.Throws<ArgumentException>(() => Tensor<float>.LayerNormalization(x, scale, new DenseTensor<float>(32), destination, -1, 1e-5f));
        Assert.Throws<ArgumentException>(() => Tensor<float>.LayerNormalization(x, scale, null, destination, 2, 1e-5f));
        Assert.Throws<ArgumentException>(() => Tensor<float>.LayerNormalization(x, scale, null, new DenseTensor<float>(new[] { 3, 22 }), -1, 1e-5f));
        Assert.Equal(before, Bits(storage));
    }
}
