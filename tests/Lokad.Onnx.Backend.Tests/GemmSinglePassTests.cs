using System;
using System.Linq;
using Xunit;

namespace Lokad.Onnx.Backend.Tests;

public class GemmSinglePassTests
{
    static float[] NaiveGemm(float[] a, float[] b, float[]? c, int[] cdims, int m, int k, int n, float alpha, float beta)
    {
        var y = new float[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                float p = 0f;
                for (int t = 0; t < k; t++) p += a[i * k + t] * b[t * n + j];
                float cb = 0f;
                if (c is not null && beta != 0f)
                {
                    if (c.Length == 1) cb = c[0];
                    else if (cdims.Length == 1) cb = c[j];
                    else cb = c[i * n + j];
                }
                y[i * n + j] = alpha * p + beta * cb;
            }
        return y;
    }

    [Fact]
    public void EquivalenceMatrix_MatchesNaive()
    {
        var rnd = new System.Random(31);
        int m = 5, k = 4, n = 6;
        var ad = new float[m * k];
        var bd = new float[k * n];
        for (int i = 0; i < ad.Length; i++) ad[i] = (float)rnd.NextDouble() - 0.5f;
        for (int i = 0; i < bd.Length; i++) bd[i] = (float)rnd.NextDouble() - 0.5f;
        var biases = new (string name, float[]? data, int[] dims)[]
        {
            ("none", null, new[] { m, n }),
            ("scalar", new float[] { 2f }, new[] { 1 }),
            ("row", new float[] { 1f, 2f, 3f, 4f, 5f, 6f }, new[] { 6 }),
            ("full", Enumerable.Range(0, m * n).Select(i => 0.25f * i).ToArray(), new[] { m, n }),
        };
        foreach (var alpha in new[] { 1f, 0.5f })
            foreach (var beta in new[] { 0f, 1f, 2f })
                foreach (var (_, data, dims) in biases)
                {
                    var a = new DenseTensor<float>(ad.ToArray(), new[] { m, k });
                    var b = new DenseTensor<float>(bd.ToArray(), new[] { k, n });
                    Tensor<float>? c = data is null ? null : new DenseTensor<float>(data.ToArray(), dims);
                    var r = CPUExecutionProvider.Gemm(a, b, c, alpha, beta, null, 0, 0);
                    Assert.Equal(OpStatus.Success, r.Status);
                    var expected = NaiveGemm(ad, bd, data, dims, m, k, n, alpha, beta);
                    Assert.Equal(expected, ((Tensor<float>)r.Outputs[0]).ToArray());
                }
    }

    [Fact]
    public void TransposedInputs_MatchManualTranspose()
    {
        var a = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        var b = DenseTensor<float>.OfValues(new float[,] { { 7f, 8f }, { 9f, 10f }, { 11f, 12f } });
        var direct = (Tensor<float>)CPUExecutionProvider.Gemm(a, b, null, 1f, 0f, null, 0, 0).Outputs[0];
        var at = Tensor<float>.Transpose(a, null);
        var bt = Tensor<float>.Transpose(b, null);
        var viaA = (Tensor<float>)CPUExecutionProvider.Gemm(at, b, null, 1f, 0f, null, transA: 1, transB: 0).Outputs[0];
        var viaB = (Tensor<float>)CPUExecutionProvider.Gemm(a, bt, null, 1f, 0f, null, transB: 1, transA: 0).Outputs[0];
        Assert.Equal(direct.ToArray(), viaA.ToArray());
        Assert.Equal(direct.ToArray(), viaB.ToArray());
    }

    [Fact]
    public void RepeatedCalls_ReturnIndependentOutputs()
    {
        var a = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f } });
        var b = DenseTensor<float>.OfValues(new float[,] { { 5f, 6f }, { 7f, 8f } });
        var r1 = (Tensor<float>)CPUExecutionProvider.Gemm(a, b, null, 1f, 0f, null, 0, 0).Outputs[0];
        var r2 = (Tensor<float>)CPUExecutionProvider.Gemm(a, b, null, 1f, 0f, null, 0, 0).Outputs[0];
        Assert.NotSame(r1, r2);
        Assert.Equal(r1.ToArray(), r2.ToArray());
    }

    [Fact]
    public void SingleOutput_AllocationBound()
    {
        var rnd = new System.Random(33);
        int m = 64, k = 64, n = 64;
        var ad = new float[m * k];
        var bd = new float[k * n];
        for (int i = 0; i < ad.Length; i++) ad[i] = (float)rnd.NextDouble();
        for (int i = 0; i < bd.Length; i++) bd[i] = (float)rnd.NextDouble();
        var a = new DenseTensor<float>(ad, new[] { m, k });
        var b = new DenseTensor<float>(bd, new[] { k, n });
        for (int i = 0; i < 3; i++) CPUExecutionProvider.Gemm(a, b, null, 1f, 0f, null, 0, 0);
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 20; i++) CPUExecutionProvider.Gemm(a, b, null, 1f, 0f, null, 0, 0);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.True(allocated < 500000L, $"gemm allocated {allocated} bytes for 20x16KB products");
    }

}
