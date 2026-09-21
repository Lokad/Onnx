using System;
using System.Linq;
using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx.Backend.Tests;

public class ConvPackedRowsTests
{
    static bool Hardware => Avx512F.IsSupported && Fma.IsSupported;

    [Fact]
    public void UnsupportedOptionsShapesAndShortPackLeaveBuffersUntouched()
    {
        float[] output = Enumerable.Repeat(12345f, 32).ToArray();
        float[] packed = Enumerable.Repeat(6789f, 32).ToArray();
        foreach (var options in new[] { TensorExecutionOptions.Scalar, TensorExecutionOptions.Simd })
            Assert.False(Tensor<float>.TryConvPackedTile(default, default, output, packed, 8, 1, 32, options));
        foreach (int columns in new[] { 0, 1, 31, 33 })
            Assert.False(Tensor<float>.TryConvPackedTile(default, default, output, packed, 8, 1, columns, TensorExecutionOptions.Auto));
        Assert.False(Tensor<float>.TryConvPackedTile(default, default, output, packed, 7, 1, 32, TensorExecutionOptions.Auto));
        Assert.False(Tensor<float>.TryConvPackedTile(default, default, output, packed, 8, 0, 32, TensorExecutionOptions.Auto));
        Assert.False(Tensor<float>.TryConvPackedTile(default, default, output, packed, 8, 2, 32, TensorExecutionOptions.Auto));
        if (!Hardware)
            Assert.False(Tensor<float>.TryConvPackedTile(default, default, output, packed, 8, 1, 32, TensorExecutionOptions.Auto));
        Assert.All(output, v => Assert.Equal(12345f, v));
        Assert.All(packed, v => Assert.Equal(6789f, v));
    }

    [Fact]
    public void OptionalPackingDeclinesUnrepresentableCombinedStorage()
    {
        var options = TensorExecutionOptions.Auto;
        Assert.Equal(0, Tensor<float>.ConvPackedScratchLength(32, 288, 192, Array.MaxLength - 55295, options));
        Assert.Equal(0, Tensor<float>.ConvPackedScratchLength(32, int.MaxValue, 192, 1000, options));
        Assert.Equal(Hardware ? 55296 : 0, Tensor<float>.ConvPackedScratchLength(32, 288, 192, 61440, options));
        Assert.Equal(0, Tensor<float>.ConvPackedScratchLength(32, 288, 192, 61440, TensorExecutionOptions.Scalar));
    }

    [SkippableFact]
    public void PackedProductClearsOldOutputAndPreservesGuardsAndInputs()
    {
        Skip.IfNot(Hardware, "AVX-512F and FMA execution required.");
        const int rows = 37, reduction = 9, columns = 64, guard = 3;
        var weights = Enumerable.Range(0, rows * reduction).Select(i => (i % 17 - 8) / 16f).ToArray();
        var patch = Enumerable.Range(0, reduction * columns).Select(i => (i % 13 - 6) / 8f).ToArray();
        var oldWeights = weights.ToArray(); var oldPatch = patch.ToArray();
        var output = Enumerable.Repeat(12345f, rows * columns + 2 * guard).ToArray();
        var packed = Enumerable.Repeat(6789f, reduction * columns + 2 * guard).ToArray();
        Assert.True(Tensor<float>.TryConvPackedTile(weights, patch, output.AsSpan(guard, rows * columns),
            packed.AsSpan(guard, reduction * columns), rows, reduction, columns, TensorExecutionOptions.Auto));
        for (int row = 0; row < rows; row++)
        for (int col = 0; col < columns; col++)
        {
            float expected = 0;
            for (int k = 0; k < reduction; k++) expected = MathF.FusedMultiplyAdd(weights[row * reduction + k], patch[k * columns + col], expected);
            Assert.Equal(BitConverter.SingleToInt32Bits(expected), BitConverter.SingleToInt32Bits(output[guard + row * columns + col]));
        }
        Assert.Equal(oldWeights, weights); Assert.Equal(oldPatch, patch);
        Assert.All(output.Take(guard).Concat(output.TakeLast(guard)), v => Assert.Equal(12345f, v));
        Assert.All(packed.Take(guard).Concat(packed.TakeLast(guard)), v => Assert.Equal(6789f, v));
        Assert.Throws<ArgumentException>(() => Tensor<float>.TryConvPackedTile(weights.AsSpan(1), patch,
            output.AsSpan(guard, rows * columns), packed.AsSpan(guard, reduction * columns), rows, reduction, columns, TensorExecutionOptions.Auto));
    }

    [SkippableFact]
    public void AllAlignedConvTilesUseOneAccountedPackingCarve()
    {
        Skip.IfNot(Hardware, "AVX-512F and FMA execution required.");
        var x = DenseTensor<float>.OfShape(1, 32, 16, 16);
        var w = DenseTensor<float>.OfShape(32, 32, 3, 3);
        x.Buffer.Span.Fill(.125f); w.Buffer.Span.Fill(.0625f);
        var accountant = new ScratchAccountant();
        var result = Tensor<float>.Conv2D(x, w, 1, new[] { 1, 1, 1, 1 }, null, null, null, null,
            TensorExecutionOptions.Auto with { ScratchReporter = accountant });
        // Width256 ->192+64, both aligned. No generic-GEMM packing rental.
        Assert.Equal((288 + 32 + 288) * 192 * 4L, accountant.TotalScratchBytes);
        Assert.Equal(32 * 16 * 16, result.Length);
        Assert.Equal(288 * .125f * .0625f, result.ToArray()[8 * 16 + 8]);
    }

    [Theory]
    [InlineData(1, 16, 33, 1, 1)]
    [InlineData(2, 32, 65, 2, 1)]
    [InlineData(2, 40, 33, 1, 2)]
    public void PublicGroupedTailConvolutionMatchesIndependentCoordinatesAndReusesSafely(
        int group, int filters, int width, int stride, int dilation)
    {
        const int channels = 16, height = 41, kernel = 3, batches = 2;
        int pad = dilation, outH = (height - 1) / stride + 1, outW = (width - 1) / stride + 1;
        var x = DenseTensor<float>.OfShape(batches, channels, height, width);
        var w = DenseTensor<float>.OfShape(filters, channels / group, kernel, kernel);
        var bias = DenseTensor<float>.OfShape(filters);
        for (int i = 0; i < x.Buffer.Length; i++) x.Buffer.Span[i] = (i % 23 - 11) / 32f;
        for (int i = 0; i < w.Buffer.Length; i++) w.Buffer.Span[i] = (i % 17 - 8) / 64f;
        for (int i = 0; i < filters; i++) bias.Buffer.Span[i] = (i % 5 - 2) / 8f;
        var inputs = x.Buffer.ToArray(); var weights = w.Buffer.ToArray();
        Tensor<float>? held = null; float[]? saved = null;
        foreach (var options in new[] { TensorExecutionOptions.Auto with { MaxDegreeOfParallelism = 2 }, TensorExecutionOptions.Scalar })
        {
            // Each call sees current weights; no packed-weight cache is used.
            w.Buffer.Span[0] += .03125f; weights[0] += .03125f;
            var result = Tensor<float>.Conv2D(x, w, group, new[] { pad, pad, pad, pad }, bias, null,
                new[] { stride, stride }, new[] { dilation, dilation }, options);
            var actual = result.ToArray();
            Assert.Equal(new[] { batches, filters, outH, outW }, result.Dimensions.ToArray());
            for (int b = 0; b < batches; b++)
            for (int m = 0; m < filters; m++)
            for (int y = 0; y < outH; y++)
            for (int z = 0; z < outW; z++)
            {
                double expected = bias.Buffer.Span[m];
                for (int c = 0; c < channels / group; c++)
                for (int ky = 0; ky < kernel; ky++)
                for (int kx = 0; kx < kernel; kx++)
                {
                    int iy = y * stride + ky * dilation - pad, ix = z * stride + kx * dilation - pad;
                    if ((uint)iy >= height || (uint)ix >= width) continue;
                    int channel = m / (filters / group) * (channels / group) + c;
                    expected += (double)inputs[((b * channels + channel) * height + iy) * width + ix]
                        * weights[((m * (channels / group) + c) * kernel + ky) * kernel + kx];
                }
                double value = actual[((b * filters + m) * outH + y) * outW + z];
                Assert.True(Math.Abs(value - expected) <= 1e-4 * Math.Max(1, Math.Abs(expected)));
            }
            Assert.Equal(inputs, x.Buffer.ToArray());
            if (held is not null) Assert.Equal(saved, held.ToArray());
            held = result; saved = actual;
        }
    }
}
