using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx.Backend.Tests;

public class DirectDepthwiseTests
{
    static float[] Values(int count, uint seed)
    {
        var a = new float[count];
        for (int i = 0; i < count; i++)
        {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            a[i] = ((seed & 0xffffff) / 8388608f - 1f) * .73f;
        }
        return a;
    }
    static DenseTensor<float> Data(int[] shape, uint seed) =>
        new DenseTensor<float>(Values(shape.Aggregate(1, (a,b) => a*b), seed), shape);

    static DenseTensor<float> ReferenceTensor(Tensor<float> input)
    {
        var value = DenseTensor<float>.OfShape(input.Dimensions.ToArray());
        input.ToDenseTensor().Buffer.Span.CopyTo(value.Buffer.Span);
        return value;
    }

    static void EqualBits(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            if (float.IsNaN(expected[i])) { Assert.True(float.IsNaN(actual[i]), $"NaN at {i}"); continue; }
            Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                $"index {i}: original={expected[i]:R}, candidate={actual[i]:R}");
        }
    }

    static DenseTensor<float> Check(Tensor<float> x, Tensor<float> w, Tensor<float>? b,
        int group, int[] pads, int[] strides, int[]? dilation, TensorExecutionOptions? selected,
        bool expectDirect)
    {
        var options = selected ?? TensorExecutionOptions.Auto;
        var savedX = x.ToArray(); var savedW = w.ToArray(); var savedB = b?.ToArray();
        // The direct path requires one worker. This exercises the existing
        // generic convolution with the same SIMD mode and tensor values.
        var expected = (DenseTensor<float>)Tensor<float>.Conv2D(ReferenceTensor(x), ReferenceTensor(w),
            group, pads, b is null ? null : ReferenceTensor(b), null, strides, dilation,
            options with { MaxDegreeOfParallelism = 2 });
        var reference = expected.Buffer.ToArray();
        var scratch = new ScratchAccountant();
        var result = (DenseTensor<float>)Tensor<float>.Conv2D(x, w, group, pads, b, null, strides, dilation,
            options with { ScratchReporter = scratch });
        EqualBits(reference, result.Buffer.ToArray());
        EqualBits(savedX, x.ToArray()); EqualBits(savedW, w.ToArray());
        if (b is not null) EqualBits(savedB!, b.ToArray());
        if (expectDirect && options.UseIntrinsics && Avx2.IsSupported && Fma.IsSupported) Assert.Equal(0, scratch.TotalScratchBytes);
        else Assert.True(scratch.TotalScratchBytes > 0);
        return result;
    }

    [Fact]
    public void EveryObservedGeometryMatchesGenericBits()
    {
        // Recorded Parakeet geometries, independent of model files and VM paths.
        int[][] geometries =
        {
            new[] { 1, 256, 424, 64, 256, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 212, 32, 256 },
            new[] { 1, 256, 212, 32, 256, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 106, 16, 256 },
            new[] { 1, 1024, 1, 114, 1024, 1, 9, 1, 1, 1, 1, 0, 0, 0, 0, 1, 106, 1024 },
            new[] { 1, 256, 885, 64, 256, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 443, 32, 256 },
            new[] { 1, 256, 443, 32, 256, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 222, 16, 256 },
            new[] { 1, 1024, 1, 230, 1024, 1, 9, 1, 1, 1, 1, 0, 0, 0, 0, 1, 222, 1024 },
            new[] { 1, 256, 447, 64, 256, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 224, 32, 256 },
            new[] { 1, 256, 224, 32, 256, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 112, 16, 256 },
            new[] { 1, 1024, 1, 120, 1024, 1, 9, 1, 1, 1, 1, 0, 0, 0, 0, 1, 112, 1024 },
            new[] { 1, 256, 665, 64, 256, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 333, 32, 256 },
            new[] { 1, 256, 333, 32, 256, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 167, 16, 256 },
            new[] { 1, 1024, 1, 175, 1024, 1, 9, 1, 1, 1, 1, 0, 0, 0, 0, 1, 167, 1024 },
            new[] { 1, 256, 353, 64, 256, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 177, 32, 256 },
            new[] { 1, 256, 177, 32, 256, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 89, 16, 256 },
            new[] { 1, 1024, 1, 97, 1024, 1, 9, 1, 1, 1, 1, 0, 0, 0, 0, 1, 89, 1024 },
            new[] { 1, 256, 628, 64, 256, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 314, 32, 256 },
            new[] { 1, 256, 314, 32, 256, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 157, 16, 256 },
            new[] { 1, 1024, 1, 165, 1024, 1, 9, 1, 1, 1, 1, 0, 0, 0, 0, 1, 157, 1024 },
            new[] { 1, 256, 204, 64, 256, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 102, 32, 256 },
            new[] { 1, 256, 102, 32, 256, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 51, 16, 256 },
            new[] { 1, 1024, 1, 59, 1024, 1, 9, 1, 1, 1, 1, 0, 0, 0, 0, 1, 51, 1024 },
            new[] { 1, 256, 757, 64, 256, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 379, 32, 256 },
            new[] { 1, 256, 379, 32, 256, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 190, 16, 256 },
            new[] { 1, 1024, 1, 198, 1024, 1, 9, 1, 1, 1, 1, 0, 0, 0, 0, 1, 190, 1024 },
            new[] { 1, 256, 352, 64, 256, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 176, 32, 256 },
            new[] { 1, 256, 176, 32, 256, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 88, 16, 256 },
            new[] { 1, 1024, 1, 96, 1024, 1, 9, 1, 1, 1, 1, 0, 0, 0, 0, 1, 88, 1024 },
            new[] { 1, 256, 632, 64, 256, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 316, 32, 256 },
            new[] { 1, 256, 316, 32, 256, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 158, 16, 256 },
            new[] { 1, 1024, 1, 166, 1024, 1, 9, 1, 1, 1, 1, 0, 0, 0, 0, 1, 158, 1024 },
            new[] { 1, 256, 332, 64, 256, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 166, 32, 256 },
            new[] { 1, 256, 166, 32, 256, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 83, 16, 256 },
            new[] { 1, 1024, 1, 91, 1024, 1, 9, 1, 1, 1, 1, 0, 0, 0, 0, 1, 83, 1024 },
            new[] { 1, 256, 623, 64, 256, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 312, 32, 256 },
            new[] { 1, 256, 312, 32, 256, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 156, 16, 256 },
            new[] { 1, 1024, 1, 164, 1024, 1, 9, 1, 1, 1, 1, 0, 0, 0, 0, 1, 156, 1024 },
            new[] { 1, 256, 453, 64, 256, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 227, 32, 256 },
            new[] { 1, 256, 227, 32, 256, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 114, 16, 256 },
            new[] { 1, 1024, 1, 122, 1024, 1, 9, 1, 1, 1, 1, 0, 0, 0, 0, 1, 114, 1024 },
            new[] { 1, 256, 899, 64, 256, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 450, 32, 256 },
            new[] { 1, 256, 450, 32, 256, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 225, 16, 256 },
            new[] { 1, 1024, 1, 233, 1024, 1, 9, 1, 1, 1, 1, 0, 0, 0, 0, 1, 225, 1024 },
            new[] { 1, 256, 242, 64, 256, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 121, 32, 256 },
            new[] { 1, 256, 121, 32, 256, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 61, 16, 256 },
            new[] { 1, 1024, 1, 69, 1024, 1, 9, 1, 1, 1, 1, 0, 0, 0, 0, 1, 61, 1024 },
            new[] { 1, 256, 622, 64, 256, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 311, 32, 256 },
            new[] { 1, 256, 311, 32, 256, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 156, 16, 256 },
            new[] { 1, 256, 407, 64, 256, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 204, 32, 256 },
            new[] { 1, 256, 204, 32, 256, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 102, 16, 256 },
            new[] { 1, 1024, 1, 110, 1024, 1, 9, 1, 1, 1, 1, 0, 0, 0, 0, 1, 102, 1024 },
            new[] { 1, 256, 601, 64, 256, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 301, 32, 256 },
            new[] { 1, 256, 301, 32, 256, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 151, 16, 256 },
            new[] { 1, 1024, 1, 159, 1024, 1, 9, 1, 1, 1, 1, 0, 0, 0, 0, 1, 151, 1024 },
            new[] { 1, 256, 477, 64, 256, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 239, 32, 256 },
            new[] { 1, 256, 239, 32, 256, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 120, 16, 256 },
            new[] { 1, 1024, 1, 128, 1024, 1, 9, 1, 1, 1, 1, 0, 0, 0, 0, 1, 120, 1024 },
            new[] { 1, 256, 675, 64, 256, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 338, 32, 256 },
            new[] { 1, 256, 338, 32, 256, 3, 3, 1, 1, 2, 2, 1, 1, 1, 1, 169, 16, 256 },
            new[] { 1, 1024, 1, 177, 1024, 1, 9, 1, 1, 1, 1, 0, 0, 0, 0, 1, 169, 1024 },
        };
        long values = 0;
        foreach (var g in geometries)
        {
            var x = Data(g[..4], 8171); var w = Data(new[] { g[4], 1, g[5], g[6] }, 1729); var b = Data(new[] { g[4] }, 6131);
            var output = Check(x, w, b, g[17], g[11..15], g[9..11], g[7..9], null, true);
            Assert.Equal(new[] { g[0], g[4], g[15], g[16] }, output.Dimensions.ToArray());
            values += output.Length;
        }
        Assert.Equal(59, geometries.Length);
        Assert.Equal(57332736L, values);
    }

    [Fact]
    public void SpatialBordersAndFlattenedVectorTailsMatch()
    {
        foreach (int h in new[] { 1, 2, 3, 7, 8, 9, 17, 18, 19 })
        for (int width = 1; width <= 35; width++)
            Check(Data(new[] { 1, 2, h, width }, 8171), Data(new[] { 2, 1, 3, 3 }, 13), width % 2 == 0 ? null : Data(new[] { 2 }, 29), 2, new[] { 1, 1, 1, 1 }, new[] { 2, 2 }, null, null, true);
    }

    [Fact]
    public void LineTailsAndMemoryOffsetsMatch()
    {
        for (int outputs = 1; outputs <= 65; outputs++)
        {
            var storage = Values(3 * (outputs + 8) + 10, 17);
            var x = new DenseTensor<float>(storage.AsMemory(5, 3 * (outputs + 8)), new[] { 1, 3, 1, outputs + 8 });
            var weights = Values(33, 29);
            var w = new DenseTensor<float>(weights.AsMemory(3, 27), new[] { 3, 1, 1, 9 });
            Check(x, w, outputs % 2 == 0 ? null : Data(new[] { 3 }, 31), 3, new int[4], new[] { 1, 1 }, null, null, true);
        }
    }

    [Fact]
    public void SpecialValuesIncludingPaddingMatch()
    {
        float[] special = [-0f, 0f, float.Epsilon, -float.Epsilon, float.MaxValue, -float.MaxValue,
            float.PositiveInfinity, float.NegativeInfinity, float.NaN];
        foreach (float value in special)
        {
            var x = Data(new[] { 1, 2, 11, 17 }, 8171); var w = Data(new[] { 2, 1, 3, 3 }, 8171);
            x.Buffer.Span[0] = value; x.Buffer.Span[35] = value; w.Buffer.Span[0] = value;
            Check(x, w, null, 2, new[] { 1, 1, 1, 1 }, new[] { 2, 2 }, null, null, true);
            x = Data(new[] { 1, 2, 1, 29 }, 8171); w = Data(new[] { 2, 1, 1, 9 }, 8171);
            x.Buffer.Span[0] = value; x.Buffer.Span[28] = value; w.Buffer.Span[8] = value;
            Check(x, w, Data(new[] { 2 }, 8171), 2, new int[4], new[] { 1, 1 }, null, null, true);
        }
    }

    [Fact]
    public void UnsupportedOptionsAndGeometriesKeepOriginalResults()
    {
        var x = Data(new[] { 1, 3, 1, 41 }, 8171); var w = Data(new[] { 3, 1, 1, 9 }, 8171);
        foreach (var options in new[] { TensorExecutionOptions.Scalar, TensorExecutionOptions.Simd,
            TensorExecutionOptions.Auto with { MaxDegreeOfParallelism = 2 },
            TensorExecutionOptions.Auto with { UseSegmentedConvolution = true } })
            Check(x, w, null, 3, new int[4], new[] { 1, 1 }, null, options, false);
        Check(x, w, null, 3, new int[4], new[] { 1, 2 }, null, null, false);
        Check(x, w, null, 3, new int[4], new[] { 1, 1 }, new[] { 1, 2 }, null, false);
        Check(x, w, null, 3, new[] { 0, 1, 0, 1 }, new[] { 1, 1 }, null, null, false);
        Check(Data(new[] { 2, 3, 1, 41 }, 8171), w, null, 3, new int[4], new[] { 1, 1 }, null, null, false);
        Check(x, Data(new[] { 3, 1, 1, 7 }, 8171), null, 3, new int[4], new[] { 1, 1 }, null, null, false);
        Check(x, Data(new[] { 3, 3, 1, 9 }, 8171), null, 1, new int[4], new[] { 1, 1 }, null, null, false);
    }

    [Fact]
    public void LogicalViewsAndPublicOutputsAreIndependent()
    {
        var source = Data(new[] { 1, 3, 1, 82 }, 8171);
        var slice = new TensorSlice<float>(source, new[] { new SliceIndex(0, 1), new SliceIndex(0, 3),
            new SliceIndex(0, 1), new SliceIndex(0, 82, 2) });
        var reverse = new DenseTensor<float>(new[] { 1, 3, 1, 41 }, true);
        for (int c = 0; c < 3; c++) for (int j = 0; j < 41; j++) reverse[0,c,0,j] = source[0,c,0,j*2];
        var broadcast = Data(new[] { 1, 1, 1, 41 }, 8171).BroadcastDim(1, 3);
        var weights = Data(new[] { 3, 1, 1, 9 }, 8171);
        foreach (var x in new Tensor<float>[] { slice, reverse, broadcast })
        {
            var first = Check(x, weights, null, 3, new int[4], new[] { 1, 1 }, null, null, true);
            var saved = first.Buffer.ToArray();
            var second = Check(x, weights, null, 3, new int[4], new[] { 1, 1 }, null, null, true);
            second.Buffer.Span.Fill(91f); EqualBits(saved, first.Buffer.ToArray());
        }
    }

    [Fact]
    public void ProviderOneDimensionalAndPooledSpatialOutputsMatch()
    {
        var x = Data(new[] { 1, 3, 1, 41 }, 8171); var w = Data(new[] { 3, 1, 1, 9 }, 8171); var b = Data(new[] { 3 }, 8171);
        var expected = Check(x, w, b, 3, new int[4], new[] { 1, 1 }, null, null, true);
        var result = CPUExecutionProvider.Conv(x.Reshape(1,3,41), w.Reshape(3,1,9), b, null,
            null, 3, null, new[] { 0, 0 }, new[] { 1 }, null);
        Assert.Equal(OpStatus.Success, result.Status);
        EqualBits(expected.Buffer.ToArray(), ((Tensor<float>)result.Outputs![0]).ToArray());
        x = Data(new[] { 1, 3, 17, 19 }, 8171); w = Data(new[] { 3, 1, 3, 3 }, 8171);
        expected = Check(x, w, b, 3, new[] { 1, 1, 1, 1 }, new[] { 2, 2 }, null, null, true);
        var pool = new TensorBufferPool(); int length = (int)expected.Length;
        var dirty = pool.Rent<float>(length); Array.Fill(dirty, float.NaN); pool.Return(dirty);
        var output = (DenseTensor<float>)Tensor<float>.Conv2D(x, w, 3, new[] { 1, 1, 1, 1 }, b,
            null, new[] { 2, 2 }, null, TensorExecutionOptions.Auto, pool);
        EqualBits(expected.Buffer.ToArray(), output.Buffer.ToArray());
        Assert.Equal(1, pool.Reused);
        Assert.True(MemoryMarshal.TryGetArray<float>(output.Buffer, out var memory) && ReferenceEquals(dirty, memory.Array));
    }
}
