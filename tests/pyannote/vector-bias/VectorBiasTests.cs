using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;

namespace Lokad.Onnx.Backend.Tests;

public class VectorBiasTests
{
    static readonly MethodInfo Tiled = typeof(Tensor<float>).GetMethod(
        "RunTiledBatchFloat", BindingFlags.Static | BindingFlags.NonPublic)!;

    public static IEnumerable<object[]> Cases()
    {
        foreach (int width in new[] { 1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63,
            64, 65, 127, 128, 129, 160, 192, 320, 472, 672, 1440, 1568 })
        foreach (bool bias in new[] { false, true })
        foreach (int policy in new[] { 0, 1, 2 })
            yield return new object[] { width, bias, policy };
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    static float Add(float a, float b) => a + b;

    [MethodImpl(MethodImplOptions.NoInlining)]
    static float Multiply(float a, float b) => a * b;

    static int Bits(float value) => BitConverter.SingleToInt32Bits(value);

    [Theory]
    [MemberData(nameof(Cases))]
    public void TiledBiasPreservesArithmeticOffsetsGuardsAndMutableBias(int block, bool hasBias, int policy)
    {
        var options = policy switch { 0 => TensorExecutionOptions.Scalar,
            1 => TensorExecutionOptions.Simd, _ => TensorExecutionOptions.Auto };
        float[] values = { -0f, 0f, 1f, -1f, .125f, float.Epsilon, -float.Epsilon,
            float.PositiveInfinity, float.NegativeInfinity, float.NaN,
            BitConverter.Int32BitsToSingle(0x7fc12345), 16777216f, -16777216f };
        foreach (int groups in new[] { 1, 2 })
        {
            int columns = 2 * block + 5, rows = 5 * groups;
            int inBatch = groups * columns, outBatch = rows * columns;
            var input = Enumerable.Range(0, 2 * inBatch + 6).Select(i => values[i % values.Length]).ToArray();
            var weights = Enumerable.Repeat(1f, rows + 6).ToArray();
            var biases = Enumerable.Range(0, rows + 6).Select(i => values[(i * 3) % values.Length]).ToArray();
            var originalInput = input.Select(Bits).ToArray();
            var originalWeights = weights.Select(Bits).ToArray();
            var scratch = Enumerable.Repeat(12345f, (groups + rows) * block).ToArray();
            for (int pass = 0; pass < 2; pass++)
            {
                biases[3] = pass == 0 ? -0f : 2f;
                var originalBiases = biases.Select(Bits).ToArray();
                var output = Enumerable.Repeat(12345f, 2 * outBatch + 6).ToArray();
                Tiled.Invoke(null, new object[] {
                    input.AsMemory(3, 2 * inBatch), weights.AsMemory(3, rows),
                    hasBias ? biases.AsMemory(3, rows) : default(Memory<float>), hasBias,
                    output.AsMemory(3, 2 * outBatch), scratch,
                    1, groups, groups, 1, columns, rows, 1, 1, 1, 1, 1, 1,
                    new MathOps.PadInfo(), 1, columns, inBatch, outBatch, columns, block, options });
                Assert.Equal(originalInput, input.Select(Bits));
                Assert.Equal(originalWeights, weights.Select(Bits));
                Assert.Equal(originalBiases, biases.Select(Bits));
                Assert.All(output.Take(3 + outBatch).Concat(output.TakeLast(3)), value => Assert.Equal(12345f, value));
                for (int row = 0; row < rows; row++)
                for (int col = 0; col < columns; col++)
                {
                    float product = Add(0f, Multiply(input[3 + inBatch + (row / 5) * columns + col], 1f));
                    float expected = hasBias ? Add(product, biases[3 + row]) : product;
                    float actual = output[3 + outBatch + row * columns + col];
                    if (float.IsNaN(expected)) Assert.True(float.IsNaN(actual));
                    else Assert.Equal(Bits(expected), Bits(actual));
                }
            }
        }
    }
}
