using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx.Backend.Tests;

public class ConvPortableRowsTests
{
    public static IEnumerable<object[]> Shapes()
    {
        foreach (int rows in new[] { 32, 64, 128, 256 })
        foreach (int reduction in new[] { 64, 288 })
        foreach (int columns in new[] { 2, 8, 31, 32, 160, 192 })
            yield return new object[] { rows, reduction, columns };
    }
    [MethodImpl(MethodImplOptions.NoInlining)]
    static float Multiply(float a, float b) => a * b;

    [Theory]
    [MemberData(nameof(Shapes))]
    public void ProductPreservesOrderedReferenceGuardsInputsAndMutableWeights(int rows, int reduction, int columns)
    {
        var weights = Enumerable.Range(0, rows * reduction + 6).Select(i => (i % 31 - 15) / 997f).ToArray();
        var patch = Enumerable.Range(0, reduction * columns + 6).Select(i => (i % 37 - 18) / 991f).ToArray();
        var output = Enumerable.Repeat(12345f, rows * columns + 6).ToArray();
        for (int pass = 0; pass < 2; pass++)
        {
            weights[3] += .125f;
            var originalWeights = weights.ToArray(); var originalPatch = patch.ToArray();
            var accountant = new ScratchAccountant();
            var options = TensorExecutionOptions.Auto with { ScratchReporter = accountant };
            bool handled = Tensor<float>.TryConvPortableRows(weights.AsSpan(3, rows * reduction),
                patch.AsSpan(3, reduction * columns), output.AsSpan(3, rows * columns), rows, reduction, columns, options);
            Assert.Equal(Fma.IsSupported, handled);
            Assert.Equal(originalWeights, weights); Assert.Equal(originalPatch, patch);
            Assert.All(output.Take(3).Concat(output.TakeLast(3)), v => Assert.Equal(12345f, v));
            if (!handled) { Assert.All(output, v => Assert.Equal(12345f, v)); Assert.Equal(0, accountant.TotalScratchBytes); continue; }
            Assert.Equal((long)reduction * columns * 4, accountant.TotalScratchBytes);
            for (int row = 0; row < rows; row++) for (int col = 0; col < columns; col++)
            {
                float expected = 0;
                for (int j = 0; j < reduction; j++)
                    expected = col < columns - columns % 8
                        ? MathF.FusedMultiplyAdd(patch[3 + j * columns + col], weights[3 + row * reduction + j], expected)
                        : expected + Multiply(weights[3 + row * reduction + j], patch[3 + j * columns + col]);
                Assert.Equal(BitConverter.SingleToInt32Bits(expected), BitConverter.SingleToInt32Bits(output[3 + row * columns + col]));
            }
        }
    }

    [Fact]
    public void UnsupportedOptionsShapesAndOversizedPackingDeclineWithoutMutation()
    {
        float[] output = Enumerable.Repeat(12345f, 16).ToArray();
        foreach (var option in new[] { TensorExecutionOptions.Scalar, TensorExecutionOptions.Simd })
            Assert.False(Tensor<float>.TryConvPortableRows(default, default, output, 32, 288, 192, option));
        foreach (var shape in new[] { (0, 64, 32), (30, 64, 32), (33, 64, 32), (96, 64, 32), (32, 9, 192),
            (32, 63, 32), (32, 64, 0), (32, 1024, 65), (64, int.MaxValue, 2), (64, 65536, 1025) })
            Assert.False(Tensor<float>.TryConvPortableRows(default, default, output, shape.Item1, shape.Item2, shape.Item3, TensorExecutionOptions.Auto));
        Assert.All(output, v => Assert.Equal(12345f, v));
    }

    [Fact]
    public void AcceptedShapeValidatesAllSpansBeforeClearingOrRenting()
    {
        var weights = new float[32 * 64]; var patch = new float[64 * 32];
        var output = Enumerable.Repeat(12345f, 32 * 32).ToArray();
        var accountant = new ScratchAccountant(); var options = TensorExecutionOptions.Auto with { ScratchReporter = accountant };
        if (Fma.IsSupported)
        {
            Assert.Throws<ArgumentException>(() => Tensor<float>.TryConvPortableRows(weights.AsSpan(1), patch, output, 32, 64, 32, options));
            Assert.Throws<ArgumentException>(() => Tensor<float>.TryConvPortableRows(weights, patch.AsSpan(1), output, 32, 64, 32, options));
            Assert.Throws<ArgumentException>(() => Tensor<float>.TryConvPortableRows(weights, patch, output.AsSpan(1), 32, 64, 32, options));
        }
        else Assert.False(Tensor<float>.TryConvPortableRows(weights, patch, output, 32, 64, 32, options));
        Assert.All(output, v => Assert.Equal(12345f, v)); Assert.Equal(0, accountant.TotalScratchBytes);
    }
}
