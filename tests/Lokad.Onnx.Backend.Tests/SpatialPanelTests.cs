using System;
using System.Linq;
using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx.Backend.Tests;

public class SpatialPanelTests
{
    [Theory]
    [InlineData(0, 0)] [InlineData(0, 1)] [InlineData(0, 2)]
    [InlineData(1, 0)] [InlineData(1, 1)] [InlineData(1, 2)]
    public void TailsPaddingDilationAndMutableWeightsMatchIndependentDouble(int geometry, int mode)
    {
        int channels = geometry == 0 ? 8 : 33, filters = geometry == 0 ? 8 : 5;
        int height = geometry == 0 ? 91 : 20, width = geometry == 0 ? 95 : 27;
        int groups = geometry == 0 ? 2 : 1, kh = 3, kw = geometry == 0 ? 2 : 3;
        int[] pads = geometry == 0 ? new[] { 3, 1, 2, 4 } : new[] { 1, 1, 1, 1 };
        int[] strides = geometry == 0 ? new[] { 2, 1 } : new[] { 1, 1 };
        int[] dilations = geometry == 0 ? new[] { 2, 3 } : new[] { 1, 1 };
        int outH = (height + pads[0] + pads[2] - ((kh - 1) * dilations[0] + 1)) / strides[0] + 1;
        int outW = (width + pads[1] + pads[3] - ((kw - 1) * dilations[1] + 1)) / strides[1] + 1;
        Assert.True((long)channels * kh * kw * outH * outW * 4 > 256 * 1024);
        float[] input = Enumerable.Range(0, channels * height * width).Select(i => ((i * 17 + 3) % 101 - 50) * .015f).ToArray();
        float[] weights = Enumerable.Range(0, filters * channels / groups * kh * kw).Select(i => ((i * 11 + 7) % 79 - 39) * .02f).ToArray();
        float[] bias = Enumerable.Range(0, filters).Select(i => (i - 3) * .05f).ToArray();
        // Offset views protect the values adjacent to the actual source spans.
        var xb = Enumerable.Repeat(12345f, input.Length + 4).ToArray(); input.CopyTo(xb, 2);
        var wb = Enumerable.Repeat(23456f, weights.Length + 4).ToArray(); weights.CopyTo(wb, 2);
        var x = new DenseTensor<float>(xb.AsMemory(2, input.Length), new[] { 1, channels, height, width });
        var w = new DenseTensor<float>(wb.AsMemory(2, weights.Length), new[] { filters, channels / groups, kh, kw });
        var b = new DenseTensor<float>(bias, new[] { filters });
        var options = mode == 0 ? TensorExecutionOptions.Scalar : mode == 1 ? TensorExecutionOptions.Simd : TensorExecutionOptions.Auto;
        var first = Tensor<float>.Conv2D(x, w, groups, pads, b, null, strides, dilations, options);
        float[] held = first.ToArray();
        Check(held, weights);
        Assert.Equal(held, Tensor<float>.Conv2D(x, w, groups, pads, b, null, strides, dilations, options).ToArray());
        weights[13] += .125f; wb[15] = weights[13];
        Check(Tensor<float>.Conv2D(x, w, groups, pads, b, null, strides, dilations, options).ToArray(), weights);
        Assert.Equal(held, first.ToArray());
        Assert.Equal(input, x.ToArray()); Assert.Equal(weights, w.ToArray());
        Assert.All(xb.Take(2).Concat(xb.TakeLast(2)), v => Assert.Equal(12345f, v));
        Assert.All(wb.Take(2).Concat(wb.TakeLast(2)), v => Assert.Equal(23456f, v));

        void Check(float[] actual, float[] kernel)
        {
            Assert.Equal(filters * outH * outW, actual.Length);
            int cg = channels / groups, mg = filters / groups;
            for (int m = 0; m < filters; m++)
            for (int oy = 0; oy < outH; oy++)
            for (int ox = 0; ox < outW; ox++)
            {
                double expected = 0;
                for (int c = 0; c < cg; c++)
                for (int ky = 0; ky < kh; ky++)
                for (int kx = 0; kx < kw; kx++)
                {
                    int sy = oy * strides[0] - pads[0] + ky * dilations[0];
                    int sx = ox * strides[1] - pads[1] + kx * dilations[1];
                    if ((uint)sy >= (uint)height || (uint)sx >= (uint)width) continue;
                    expected += (double)input[((m / mg * cg + c) * height + sy) * width + sx]
                        * kernel[((m * cg + c) * kh + ky) * kw + kx];
                }
                expected += bias[m];
                double value = actual[(m * outH + oy) * outW + ox];
                Assert.True(double.IsFinite(value) && Math.Abs(value - expected) <= 1e-4 * Math.Max(1, Math.Abs(expected)),
                    $"mode={mode}, m={m}, y={oy}, x={ox}: {value} versus {expected}");
            }
        }
    }

    [Fact]
    public void EmptyBatchWithLargeSpatialGeometryRemainsEmpty()
    {
        var input = DenseTensor<float>.OfShape(new[] { 0, 8, 91, 95 });
        var weights = DenseTensor<float>.OfShape(new[] { 8, 8, 3, 3 });
        var output = Tensor<float>.Conv2D(input, weights, 1, new[] { 1, 1, 1, 1 }, null, null, null, null, TensorExecutionOptions.Auto);
        Assert.Equal(new[] { 0, 8, 91, 95 }, output.Dimensions.ToArray());
        Assert.Empty(output.ToArray());
    }
}
