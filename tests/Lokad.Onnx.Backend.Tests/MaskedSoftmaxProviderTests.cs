using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// E69 matcher implementation: the MaskedSoftmax provider entry runs the fused
/// kernel on shared-[S] masks and the exact Add-then-Softmax composite on every
/// other shape, so fused and unfused agree bit for bit in all cases.
/// </summary>
public class MaskedSoftmaxProviderTests
{
    static float[] Composite(float[] scores, int rows, int block, float[] mask, int axis)
    {
        var s = new DenseTensor<float>((float[])scores.Clone(), new[] { rows, block });
        var m = new DenseTensor<float>((float[])mask.Clone(), mask.Length == block ? new[] { block } : new[] { rows, block });
        var add = CPU.Add(s, m, null, null);
        Assert.Equal(OpStatus.Success, add.Status);
        var sm = CPU.Softmax(add.Outputs![0], axis, null, null, 13);
        Assert.Equal(OpStatus.Success, sm.Status);
        return ((Tensor<float>)sm.Outputs![0]).ToDenseTensor().Buffer.Span.ToArray();
    }

    static void FusedEqual(float[] scores, int rows, int block, float[] mask, int axis)
    {
        var s = new DenseTensor<float>((float[])scores.Clone(), new[] { rows, block });
        var m = new DenseTensor<float>((float[])mask.Clone(), mask.Length == block ? new[] { block } : new[] { rows, block });
        var fused = CPU.MaskedSoftmax(s, m, axis, null, null, 13);
        Assert.Equal(OpStatus.Success, fused.Status);
        var got = ((Tensor<float>)fused.Outputs![0]).ToDenseTensor().Buffer.Span.ToArray();
        var want = Composite(scores, rows, block, mask, axis);
        Assert.True(got.AsSpan().SequenceEqual(want),
            $"masked provider diverges on {rows}x{block} axis={axis}.");
    }

    static float[] MixedMask(int block, int seed)
    {
        var rnd = new Random(seed);
        var m = new float[block];
        for (int i = 0; i < m.Length; i++) m[i] = (rnd.NextDouble() < 0.3) ? float.NegativeInfinity : 0f;
        m[0] = 0f;
        return m;
    }

    static float[] RandScores(int rows, int block, int seed)
    {
        var rnd = new Random(seed);
        var x = new float[rows * block];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rnd.NextDouble() * 8 - 4);
        return x;
    }

    [Fact]
    public void FusedSharedMaskMatchesComposite()
    {
        FusedEqual(RandScores(3, 8, 7001), 3, 8, MixedMask(8, 7001), -1);
        FusedEqual(RandScores(2, 30, 7002), 2, 30, MixedMask(30, 7002), -1);
        FusedEqual(RandScores(24, 128, 7003), 24, 128, MixedMask(128, 7003), -1);
    }

    [Fact]
    public void FallbackParallelMaskMatchesComposite()
    {
        int rows = 3, block = 8;
        var scores = RandScores(rows, block, 7011);
        var rnd = new Random(7011);
        var mask = new float[rows * block];
        for (int i = 0; i < mask.Length; i++) mask[i] = (rnd.NextDouble() < 0.3) ? float.NegativeInfinity : 0f;
        FusedEqual(scores, rows, block, mask, -1);
    }

    [Fact]
    public void FallbackNonLastAxisMatchesComposite()
    {
        FusedEqual(RandScores(8, 2, 7021), 8, 2, MixedMask(2, 7021), 0);
    }

    [Fact]
    public void MissingInputsFailCleanly()
    {
        var s = new DenseTensor<float>(new float[8], new[] { 1, 8 });
        var m = new DenseTensor<float>(new float[8], new[] { 8 });
        Assert.Equal(OpStatus.Failure, CPU.MaskedSoftmax(null, m, -1, null, null, 13).Status);
        Assert.Equal(OpStatus.Failure, CPU.MaskedSoftmax(s, null, -1, null, null, 13).Status);
    }
}

