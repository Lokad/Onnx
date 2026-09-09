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
                    else cb = c[(cdims[0] == 1 ? 0 : i) * cdims[1] + (cdims[1] == 1 ? 0 : j)];
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
    public void NonDenseBiasViews_MatchNaiveExactly()
    {
        var rnd = new System.Random(37);
        int m = 2, k = 3, n = 6;
        var ad = new float[m * k];
        var bd = new float[k * n];
        for (int i = 0; i < ad.Length; i++) ad[i] = (float)rnd.NextDouble() - 0.5f;
        for (int i = 0; i < bd.Length; i++) bd[i] = (float)rnd.NextDouble() - 0.5f;
        var a = new DenseTensor<float>(ad.ToArray(), new[] { m, k });
        var b = new DenseTensor<float>(bd.ToArray(), new[] { k, n });
        float alpha = 0.5f, beta = 2f;
        var big2 = new DenseTensor<float>(Enumerable.Range(0, 12).Select(i => 0.25f * i).ToArray(), new[] { 2, 6 });
        Tensor<float> rowView = big2.Slice(new SliceIndex(1, 2));
        var rr = CPUExecutionProvider.Gemm(a, b, rowView, alpha, beta, null, 0, 0);
        Assert.Equal(OpStatus.Success, rr.Status);
        var rowVals = new float[6];
        for (int j = 0; j < 6; j++) rowVals[j] = 0.25f * (6 + j);
        Assert.Equal(NaiveGemm(ad, bd, rowVals, new[] { 1, 6 }, m, k, n, alpha, beta), ((Tensor<float>)rr.Outputs[0]).ToArray());
        var big1 = new DenseTensor<float>(Enumerable.Range(0, 12).Select(i => 0.5f * i).ToArray(), new[] { 12 });
        Tensor<float> vecView = big1.Slice(new SliceIndex(3, 9));
        var rv = CPUExecutionProvider.Gemm(a, b, vecView, 2f, 0.5f, null, 0, 0);
        Assert.Equal(OpStatus.Success, rv.Status);
        var vecVals = new float[6];
        for (int j = 0; j < 6; j++) vecVals[j] = 0.5f * (3 + j);
        Assert.Equal(NaiveGemm(ad, bd, vecVals, new[] { 6 }, m, k, n, 2f, 0.5f), ((Tensor<float>)rv.Outputs[0]).ToArray());
    }

    [Fact]
    public void DoubleEpilogue_MatchesNaiveExactly()
    {
        var rnd = new System.Random(39);
        int m = 3, k = 4, n = 5;
        var ad = new double[m * k];
        var bd = new double[k * n];
        var cd = new double[n];
        for (int j = 0; j < n; j++) cd[j] = rnd.NextDouble();
        var a = new DenseTensor<double>(ad.ToArray(), new[] { m, k });
        var b = new DenseTensor<double>(bd.ToArray(), new[] { k, n });
        var c = new DenseTensor<double>(cd.ToArray(), new[] { n });
        var r = CPUExecutionProvider.Gemm(a, b, c, 0.5f, 2f, null, 0, 0);
        Assert.Equal(OpStatus.Success, r.Status);
        // Zero products isolate the epilogue bit-wise: any summation order
        // accumulates exact zeros, so only the scale/bias pass can differ.
        var expected = new double[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                expected[i * n + j] = 0.5 * 0.0 + 2.0 * cd[j];
            }
        Assert.Equal(expected, ((Tensor<double>)r.Outputs[0]).ToArray());
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
