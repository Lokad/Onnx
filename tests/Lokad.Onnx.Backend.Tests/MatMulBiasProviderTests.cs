using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// E71 bias-epilogue fusion: the provider runs the exact MatMul-then-Add
/// composite on shapes the epilogue twin declines (no prepared weights here,
/// so every case below exercises the fallback wiring bit-identically).
/// The fused kernel path is proven by PackedBiasMatchesCompositeBitwise plus
/// model-level agreement where the matcher fires.
/// </summary>
public class MatMulBiasProviderTests
{
    static float[] Composite(float[] a, int[] adims, float[] b, int[] bdims, float[] bias, int[] biasdims)
    {
        var ta = new DenseTensor<float>((float[])a.Clone(), adims);
        var tb = new DenseTensor<float>((float[])b.Clone(), bdims);
        var tbias = new DenseTensor<float>((float[])bias.Clone(), biasdims);
        var mm = CPU.MatMul(ta, tb, null, null);
        Assert.Equal(OpStatus.Success, mm.Status);
        var ad = CPU.Add(mm.Outputs![0]!, tbias, null, null);
        Assert.Equal(OpStatus.Success, ad.Status);
        return ((Tensor<float>)ad.Outputs![0]).ToDenseTensor().Buffer.Span.ToArray();
    }

    static void FusedEqual(float[] a, int[] adims, float[] b, int[] bdims, float[] bias, int[] biasdims, int seed)
    {
        var ta = new DenseTensor<float>((float[])a.Clone(), adims);
        var tb = new DenseTensor<float>((float[])b.Clone(), bdims);
        var tbias = new DenseTensor<float>((float[])bias.Clone(), biasdims);
        var fused = CPU.MatMulBias(ta, tb, tbias, null, null);
        Assert.Equal(OpStatus.Success, fused.Status);
        var got = ((Tensor<float>)fused.Outputs![0]).ToDenseTensor().Buffer.Span.ToArray();
        var want = Composite(a, adims, b, bdims, bias, biasdims);
        Assert.True(got.AsSpan().SequenceEqual(want), "matmulbias provider diverges.");
        _ = seed;
    }

    static float[] Rand(int n, int seed)
    {
        var rnd = new Random(seed);
        var x = new float[n];
        for (int i = 0; i < n; i++) x[i] = (float)(rnd.NextDouble() * 4 - 2);
        return x;
    }

    [Fact]
    public void FallbackMatchesCompositeBitwise()
    {
        FusedEqual(Rand(2 * 4, 8101), new[] { 2, 4 }, Rand(4 * 3, 8101), new[] { 4, 3 }, Rand(3, 8101), new[] { 3 }, 8101);
        FusedEqual(Rand(3 * 5, 8102), new[] { 3, 5 }, Rand(5 * 7, 8102), new[] { 5, 7 }, Rand(7, 8102), new[] { 7 }, 8102);
        FusedEqual(Rand(8 * 16, 8103), new[] { 8, 16 }, Rand(16 * 32, 8103), new[] { 16, 32 }, Rand(32, 8103), new[] { 32 }, 8103);
    }

    [Fact]
    public void MissingInputsFailCleanly()
    {
        var a = new DenseTensor<float>(new float[4], new[] { 2, 2 });
        Assert.Equal(OpStatus.Failure, CPU.MatMulBias(null, a, a, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.MatMulBias(a, null, a, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.MatMulBias(a, a, null, null, null).Status);
    }
}
