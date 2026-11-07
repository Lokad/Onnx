namespace Lokad.Onnx.Backend.Tests;

using System;
using System.Linq;
using System.Numerics;
using Xunit;

public class PreparedDenseConsumerTests
{
    static PreparedFloatTensor Prepared(DenseTensor<float> value) => new PreparedFloatValue(value).CreateTensor();
    static int[] Bits(Tensor<float> value) => value.ToArray().Select(BitConverter.SingleToInt32Bits).ToArray();

    [Fact]
    public void PreparedHarmonicsKeepDenseResultsWithoutExpandedTemporaryArrays()
    {
        const int width = 512, count = 64;
        var p = new DenseTensor<float>(Enumerable.Range(0, 2 * width).Select(i => .125f + (i % 7) / 128f).ToArray(), new[] { 2, 1, width });
        var a = new DenseTensor<float>(Enumerable.Range(0, 2 * width).Select(i => (i % 9 - 4) / 64f).ToArray(), p.Dimensions);
        var ph = new DenseTensor<float>(Enumerable.Range(0, 2 * width).Select(i => (i % 31 - 15) / 64f).ToArray(), p.Dimensions);
        var h = new DenseTensor<float>(Enumerable.Range(0, count).Select(i => i / 16f).ToArray(), new[] { 1, count, 1 });
        var f = new DenseTensor<float>(Enumerable.Range(0, count).Select(i => i / 32f).ToArray(), h.Dimensions);
        var threshold = new DenseTensor<float>(new[] { .5f }, Array.Empty<int>());
        var axes = DenseTensor<long>.OfValues(new[] { 1L });
        var hp = Prepared(h); var fp = Prepared(f); var tp = Prepared(threshold);
        int[] hBefore = Bits(hp), fBefore = Bits(fp);
        Tensor<float> Run(ITensor harmonics, ITensor frequencies, ITensor limit)
        {
            var result = CPUExecutionProvider.HarmonicSum(p, harmonics, limit, a, frequencies, ph, axes, 1, 0, ExecutionOptions.Default);
            Assert.Equal(OpStatus.Success, result.Status);
            return Assert.IsAssignableFrom<Tensor<float>>(Assert.Single(result.Outputs));
        }
        var expected = Run(h, f, threshold);
        Assert.Equal(Bits(expected), Bits(Run(hp, fp, tp)));
        for (int i = 0; i < expected.Length; i++)
        {
            double sum = 0;
            for (int k = 0; k < count; k++)
                if (p.GetValue(i) * h.GetValue(k) < .5f)
                    sum += Math.Sin((double)f.GetValue(k) * ph.GetValue(i)) * a.GetValue(i);
            Assert.True(Math.Abs(expected.GetValue(i) - sum) / Math.Max(1, Math.Abs(sum)) <= 1e-4);
        }
        for (int warm = 0; warm < 3; warm++) { _ = Run(h, f, threshold); _ = Run(hp, fp, tp); }
        long Measure(ITensor harmonics, ITensor frequencies, ITensor limit)
        {
            long before = GC.GetAllocatedBytesForCurrentThread();
            for (int repeat = 0; repeat < 8; repeat++) _ = Run(harmonics, frequencies, limit);
            return GC.GetAllocatedBytesForCurrentThread() - before;
        }
        long denseBytes = Measure(h, f, threshold), preparedBytes = Measure(hp, fp, tp);
        if (Vector.IsHardwareAccelerated)
            Assert.True(preparedBytes <= denseBytes + 65536,
                $"Prepared harmonic inputs expanded avoidable temporaries: dense={denseBytes}, prepared={preparedBytes}.");
        Assert.Equal(hBefore, Bits(hp)); Assert.Equal(fBefore, Bits(fp));
        hp.SetValue(1, 7f); h.SetValue(1, 7f);
        Assert.Equal(Bits(Run(h, f, threshold)), Bits(Run(hp, fp, tp)));
    }

    [Fact]
    public void PreparedAffineViewsAndAliasedDestinationsKeepLogicalValues()
    {
        var prepared = Prepared(new DenseTensor<float>(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, new[] { 2, 3 }));
        var reverse = prepared.Slice(SliceIndex.All, new SliceIndex("::-1"));
        var left = reverse.Reshape(new[] { 2, 1, 3 }).BroadcastDim(1, 5);
        var right = new DenseTensor<float>(Enumerable.Range(0, 30).Select(i => (float)i).ToArray(), new[] { 2, 5, 3 });
        var result = CPUExecutionProvider.Add(left, right, ExecutionOptions.Default, null);
        Assert.Equal(OpStatus.Success, result.Status);
        var actual = Assert.IsAssignableFrom<Tensor<float>>(Assert.Single(result.Outputs));
        for (int batch = 0; batch < 2; batch++)
            for (int row = 0; row < 5; row++)
                for (int col = 0; col < 3; col++)
                    Assert.Equal(batch * 3 + 3 - col + (batch * 15 + row * 3 + col), actual[batch, row, col]);
        var destination = prepared.ToDenseTensor();
        prepared.BroadcastApply(destination, static (x, y) => x + y, destination, TensorExecutionOptions.Scalar);
        Assert.Equal(new[] { 2f, 4f, 6f, 8f, 10f, 12f }, prepared.ToArray());
    }

    [Fact]
    public void PreparedOperandsKeepUserCallbackMutationVisible()
    {
        var value = Prepared(new DenseTensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 4 }));
        var zeros = new DenseTensor<float>(new[] { 4 });
        int calls = 0;
        var output = value.BroadcastApply(zeros, (x, y) =>
        {
            if (calls++ == 0) value.SetValue(3, 99f);
            return x + y;
        }, TensorExecutionOptions.Scalar);
        Assert.Equal(4, calls);
        Assert.Equal(new[] { 1f, 2f, 3f, 99f }, output.ToArray());
        Assert.Equal(99f, value.GetValue(3));
    }
}
