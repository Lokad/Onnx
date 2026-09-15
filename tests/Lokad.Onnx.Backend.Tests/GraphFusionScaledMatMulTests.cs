using System.Collections.Generic;

namespace Lokad.Onnx.Backend.Tests;

// P07 scale fusion: Mul(data, scalar-scale) feeding MatMul rewrites to the
// ScaledMatMul fused op. Composite-first: the fused kernel sequences the exact
// legacy Mul then MatMul, so these gates prove plumbing with identical values
// before any fused kernel arrives.
public class GraphFusionScaledMatMulTests
{
    static DenseTensor<float> Rand(int[] dims, int seed)
    {
        var rnd = new System.Random(seed);
        var n = 1;
        foreach (var d in dims) n *= d;
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)rnd.NextDouble() * 2f - 1f;
        return new DenseTensor<float>(a.AsMemory(), dims);
    }

    static float[] Out(ITensor t) => ((Tensor<float>)t).ToDenseTensor().Buffer.ToArray();

    [Fact]
    public void ScaledMatMul_CompositeMatchesLegacyBitwise()
    {
        var a = Rand(new[] { 201, 384 }, 7);
        var b = Rand(new[] { 384, 384 }, 11);
        var s = DenseTensor<float>.Scalar(0.125f);
        var legacy = CPUExecutionProvider.Mul(a, s, null, null);
        Assert.Equal(OpStatus.Success, legacy.Status);
        var mm = CPUExecutionProvider.MatMul(legacy.Outputs![0]!, b, null, null);
        Assert.Equal(OpStatus.Success, mm.Status);
        var fused = CPUExecutionProvider.ScaledMatMul(a, b, s, null, null);
        Assert.Equal(OpStatus.Success, fused.Status);
        Assert.Equal(Out(mm.Outputs![0]!), Out(fused.Outputs![0]!));
    }

    [Fact]
    public void ScaledMatMul_MissingScaleIsNotSuccess()
    {
        var a = Rand(new[] { 4, 8 }, 7);
        var b = Rand(new[] { 8, 4 }, 11);
        var r = CPUExecutionProvider.ScaledMatMul(a, b, null, null, null);
        Assert.NotEqual(OpStatus.Success, r.Status);
    }
}
