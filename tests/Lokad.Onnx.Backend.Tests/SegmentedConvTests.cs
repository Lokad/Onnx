namespace Lokad.Onnx.Backend.Tests;

using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;

public class SegmentedConvTests
{
    static IEnumerable<TensorExecutionOptions> Modes()
    {
        yield return TensorExecutionOptions.Scalar;
        yield return TensorExecutionOptions.Simd;
        if (Fma.IsSupported) yield return TensorExecutionOptions.Intrinsics;
    }

    static DenseTensor<float> Values(int[] shape, int seed)
    {
        var random = new Random(seed);
        return new DenseTensor<float>(Enumerable.Range(0, shape.Aggregate(1, (a, b) => a * b))
            .Select(_ => (float)(random.NextDouble() - .5)).ToArray(), shape);
    }

    // A separate output-coordinate reference, without im2col or matrix kernels.
    static float[] Reference(Tensor<float> input, Tensor<float> weights, Tensor<float> bias,
        int groups, int[] pads, int[] strides, int[] dilations, bool fused)
    {
        int batches = input.Dimensions[0], channels = input.Dimensions[1];
        int height = input.Dimensions[2], width = input.Dimensions[3];
        int filters = weights.Dimensions[0], kh = weights.Dimensions[2], kw = weights.Dimensions[3];
        int oh = (height + pads[0] + pads[2] - (dilations[0] * (kh - 1) + 1)) / strides[0] + 1;
        int ow = (width + pads[1] + pads[3] - (dilations[1] * (kw - 1) + 1)) / strides[1] + 1;
        var result = new List<float>();
        for (int b = 0; b < batches; b++)
        for (int m = 0; m < filters; m++)
        for (int y = 0; y < oh; y++)
        for (int x = 0; x < ow; x++)
        {
            int group = m / (filters / groups), terms = 0;
            float sum = 0, part = 0;
            double wide = 0;
            for (int c = 0; c < channels / groups; c++)
            for (int i = 0; i < kh; i++)
            for (int j = 0; j < kw; j++)
            {
                int sy = y * strides[0] + i * dilations[0] - pads[0];
                int sx = x * strides[1] + j * dilations[1] - pads[1];
                float value = (uint)sy < (uint)height && (uint)sx < (uint)width
                    ? input[b, group * (channels / groups) + c, sy, sx] : 0;
                float weight = weights[m, c, i, j];
                wide += (double)value * weight;
                part = fused ? MathF.FusedMultiplyAdd(weight, value, part) : AddProduct(weight, value, part);
                terms++;
                if (terms % 128 == 0)
                {
                    sum = terms == 128 ? part : sum + part;
                    part = 0;
                }
            }
            if (terms % 128 != 0) sum = terms < 128 ? part : sum + part;
            float actual = sum + bias[m];
            Assert.InRange(Math.Abs(actual - (wide + bias[m])), 0, 1e-4 * Math.Max(1, Math.Abs(wide + bias[m])));
            result.Add(actual);
        }
        return result.ToArray();
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    static float AddProduct(float a, float b, float c) => a * b + c;

    [Theory]
    [InlineData(1, 1)]
    [InlineData(7, 3)]
    [InlineData(31, 31)]
    [InlineData(127, 32)]
    [InlineData(128, 33)]
    [InlineData(129, 65)]
    [InlineData(257, 5)]
    public void ReductionPositionAndFilterTailsMatchIndependentReference(int positions, int filters)
    {
        var input = Values(new[] { 1, 43, 1, positions + 2 }, 12);
        var weight = Values(new[] { filters, 43, 1, 3 }, 29); // 129 reduction terms.
        var bias = Values(new[] { filters }, 45);
        var before = input.ToArray();
        var weightBefore = weight.ToArray();
        foreach (var mode in Modes())
        {
            var options = mode with { UseSegmentedConvolution = true };
            var output = Tensor<float>.Conv2D(input, weight, 1, new[] { 0, 0, 0, 0 }, bias,
                null, new[] { 1, 1 }, new[] { 1, 1 }, options);
            var expected = Reference(input, weight, bias, 1, new[] { 0, 0, 0, 0 }, new[] { 1, 1 }, new[] { 1, 1 }, mode.UseIntrinsics);
            Assert.Equal(expected.Select(BitConverter.SingleToInt32Bits), output.ToArray().Select(BitConverter.SingleToInt32Bits));
        }
        Assert.Equal(before, input.ToArray());
        Assert.Equal(weightBefore, weight.ToArray());
    }

    [Theory]
    [InlineData(1, 1, 1, 1)]
    [InlineData(2, 3, 1, 2)]
    [InlineData(3, 2, 2, 1)]
    public void GroupedBatchesBordersDilationAndStride(int sh, int sw, int dh, int dw)
    {
        var input = Values(new[] { 3, 86, 5, 71 }, 18);
        var weight = Values(new[] { 6, 43, 2, 3 }, 39); // 258 terms per group.
        var bias = Values(new[] { 6 }, 61);
        int[] pads = { 2, 3, 1, 2 }, strides = { sh, sw }, dilations = { dh, dw };
        foreach (var mode in Modes())
        {
            var options = mode with { UseSegmentedConvolution = true, MaxDegreeOfParallelism = 2 };
            var actual = Tensor<float>.Conv2D(input, weight, 2, pads, bias, null, strides, dilations, options);
            var expected = Reference(input, weight, bias, 2, pads, strides, dilations, mode.UseIntrinsics);
            Assert.Equal(expected.Select(BitConverter.SingleToInt32Bits), actual.ToArray().Select(BitConverter.SingleToInt32Bits));
        }
    }

    [Theory]
    [InlineData(129)]
    [InlineData(10001)]
    public void WorkspaceIsBoundedAndRetainedOutputsAreIndependent(int length)
    {
        var input = Values(new[] { 1, 2, 1, length }, 123);
        var weight = Values(new[] { 3, 2, 1, 5 }, 124);
        var scratch = new ScratchAccountant();
        var options = TensorExecutionOptions.Auto with { UseSegmentedConvolution = true, ScratchReporter = scratch };
        var first = Tensor<float>.Conv2D(input, weight, 1, new int[4], null, null, null, null, options);
        Assert.Equal(98304, scratch.TotalScratchBytes);
        var saved = first.ToArray();
        input.Buffer.Span[0] = 13f;
        var second = Tensor<float>.Conv2D(input, weight, 1, new int[4], null, null, null, null, options);
        Assert.Equal(196608, scratch.TotalScratchBytes);
        Assert.Equal(saved, first.ToArray());
        Assert.NotEqual(saved[0], second.GetValue(0));
    }

    [Fact]
    public void WindowedStorageKeepsGuardsAndUsesCorrectOrigin()
    {
        var x = Values(new[] { 1, 2, 3, 9 }, 67);
        var w = Values(new[] { 3, 2, 2, 3 }, 98);
        var backing = Enumerable.Repeat(12345f, (int)x.Length + 19).ToArray();
        x.ToArray().CopyTo(backing, 7);
        var window = new DenseTensor<float>(backing.AsMemory(7, (int)x.Length), x.Dimensions);
        var bias = Values(new[] { 3 }, 87);
        var expected = Reference(x, w, bias, 1, new int[4], new[] { 1, 1 }, new[] { 1, 1 }, Fma.IsSupported);
        var actual = Tensor<float>.Conv2D(window, w, 1, new int[4], bias, null, null, null,
            TensorExecutionOptions.Auto with { UseSegmentedConvolution = true });
        Assert.Equal(expected, actual.ToArray());
        Assert.All(backing.Take(7).Concat(backing.Skip(7 + (int)x.Length)), value => Assert.Equal(12345f, value));
    }

    [Fact]
    public void ReversedStorageIsDensifiedBeforeCoordinateExpansion()
    {
        var x = new DenseTensor<float>(Values(new[] { 1, 2, 3, 9 }, 31).ToArray(), new[] { 1, 2, 3, 9 }, true);
        var w = new DenseTensor<float>(Values(new[] { 3, 2, 2, 3 }, 41).ToArray(), new[] { 3, 2, 2, 3 }, true);
        var bias = Values(new[] { 3 }, 17);
        var expected = Reference(x, w, bias, 1, new int[4], new[] { 1, 1 }, new[] { 1, 1 }, Fma.IsSupported);
        var actual = Tensor<float>.Conv2D(x, w, 1, new int[4], bias, null, null, null,
            TensorExecutionOptions.Auto with { UseSegmentedConvolution = true });
        Assert.Equal(expected, actual.ToArray());
    }

    [Fact]
    public void EachReductionBlockStartsFromZeroBeforeAddingToPreviousOutput()
    {
        // Continuous accumulation gives 1. The second block rounds -1e8 + 1
        // before adding the first block. Blocking is not an accuracy guarantee.
        var input = new DenseTensor<float>(Enumerable.Repeat(1f, 130).ToArray(), new[] { 1, 1, 1, 130 });
        var values = new float[130]; values[0] = 1e8f; values[128] = -1e8f; values[129] = 1;
        var weight = new DenseTensor<float>(values, new[] { 1, 1, 1, 130 });
        foreach (var mode in Modes())
            Assert.Equal(0, Tensor<float>.Conv2D(input, weight, 1, new int[4], null, null, null, null,
                mode with { UseSegmentedConvolution = true }).GetValue(0));
    }

    [Fact]
    public void EmptyBatchAndEmptyReductionPreserveShapesAndBias()
    {
        var options = TensorExecutionOptions.Auto with { UseSegmentedConvolution = true };
        var empty = Tensor<float>.Conv2D(DenseTensor<float>.OfShape(0, 1, 1, 5),
            DenseTensor<float>.OfShape(2, 1, 1, 3), 1, new int[4], null, null, null, null, options);
        Assert.Equal(new[] { 0, 2, 1, 3 }, empty.Dimensions.ToArray());
        Assert.Empty(empty.ToArray());
        var reduced = Tensor<float>.Conv2D(DenseTensor<float>.OfShape(1, 0, 1, 5),
            DenseTensor<float>.OfShape(2, 0, 1, 3), 1, new int[4], DenseTensor<float>.OfValues(new[] { 2f, -3f }),
            null, null, null, options);
        Assert.Equal(new[] { 2f, 2f, 2f, -3f, -3f, -3f }, reduced.ToArray());
    }

    [Fact]
    public void PaddingTimesInfinityAndOpposingBlocksPropagateNaN()
    {
        foreach (var mode in Modes())
        {
            var options = mode with { UseSegmentedConvolution = true };
            var x = DenseTensor<float>.OfValues(new float[,,,] { { { { 1 } } } });
            var w = DenseTensor<float>.OfValues(new float[,,,] { { { { float.PositiveInfinity, 1 } } } });
            var actual = Tensor<float>.Conv2D(x, w, 1, new[] { 0, 1, 0, 0 }, null, null, null, null, options);
            Assert.True(float.IsNaN(actual.GetValue(0)));
            var input = new DenseTensor<float>(Enumerable.Repeat(1f, 129).ToArray(), new[] { 1, 1, 1, 129 });
            var values = new float[129]; values[0] = float.PositiveInfinity; values[128] = float.NegativeInfinity;
            var weight = new DenseTensor<float>(values, new[] { 1, 1, 1, 129 });
            Assert.True(float.IsNaN(Tensor<float>.Conv2D(input, weight, 1, new int[4], null, null, null, null, options).GetValue(0)));
        }
    }

    [Fact]
    public void PointwiseFastPathStillNeedsNoScratch()
    {
        var accountant = new ScratchAccountant();
        var output = Tensor<float>.Conv2D(Values(new[] { 1, 1, 1, 7 }, 1), Values(new[] { 1, 1, 1, 1 }, 2),
            1, new int[4], null, null, null, null,
            TensorExecutionOptions.Scalar with { UseSegmentedConvolution = true, ScratchReporter = accountant });
        Assert.Equal(7, output.Length);
        Assert.Equal(0, accountant.TotalScratchBytes);
    }
}
