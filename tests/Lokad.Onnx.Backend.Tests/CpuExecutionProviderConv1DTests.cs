using System;
using System.Linq;
using System.Runtime.Intrinsics.X86;
using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

// One-dimensional Conv/MaxPool coverage adapted from voice commit 13e98bd. Rank-three
// inputs ride the proven two-dimensional kernels through shape
// normalization; expected values are frozen Python onnxruntime 1.29.0
// oracles (ORT_SEQUENTIAL, intra/inter-op 1, ORT_ENABLE_ALL, opset 17)
// retained below with their complete deterministic inputs. Double-precision
// agreement follows the ConvDirectTests pattern (ORT has no double Conv
// CPU kernel): on exactly-representable ints both precisions must agree
// bit for bit.
public class CpuExecutionProviderConv1DTests
{
    const double Tol = 1e-5;

    [SkippableTheory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void GroupedBatchesOffsetsAndFusedReluMatchIndependentRankThreeOracle(int mode)
    {
        Skip.If(mode == 2 && !Fma.IsSupported, "Explicit intrinsic mode requires x86 FMA.");
        var options = mode == 0 ? ExecutionOptions.Scalar : mode == 1 ? ExecutionOptions.Simd : ExecutionOptions.Intrinsics;
        const int batch = 2, channels = 4, length = 9, outputs = 6, group = 2, kernel = 3;
        const int stride = 2, dilation = 2, left = 2, right = 1, offset = 4;
        const int outLength = (length + left + right - ((kernel - 1) * dilation + 1)) / stride + 1;
        var xs = Enumerable.Range(0, batch * channels * length + 2 * offset).Select(i => (i % 17 - 8) * 0.25f).ToArray();
        var ws = Enumerable.Range(0, outputs * (channels / group) * kernel + 2 * offset).Select(i => (i % 11 - 5) * 0.25f).ToArray();
        var originalX = (float[])xs.Clone();
        var originalW = (float[])ws.Clone();
        var x = new DenseTensor<float>(xs.AsMemory(offset, batch * channels * length), new[] { batch, channels, length });
        var w = new DenseTensor<float>(ws.AsMemory(offset, outputs * (channels / group) * kernel), new[] { outputs, channels / group, kernel });
        var bias = DenseTensor<float>.OfValues(new float[] { -2, -1, 0, 1, 2, 3 });
        var expected = new float[batch * outputs * outLength];
        for (int b = 0; b < batch; b++)
        for (int m = 0; m < outputs; m++)
        for (int pos = 0; pos < outLength; pos++)
        {
            float sum = bias.GetValue(m);
            int firstChannel = m / (outputs / group) * (channels / group);
            for (int c = 0; c < channels / group; c++)
            for (int k = 0; k < kernel; k++)
            {
                int tap = pos * stride - left + k * dilation;
                if (tap >= 0 && tap < length)
                    sum += xs[offset + (b * channels + firstChannel + c) * length + tap]
                        * ws[offset + (m * (channels / group) + c) * kernel + k];
            }
            expected[(b * outputs + m) * outLength + pos] = sum;
        }
        var plain = CPU.Conv(x, w, bias, null, new[] { dilation }, group, new[] { kernel }, new[] { left, right }, new[] { stride }, options);
        var fused = CPU.ConvRelu(x, w, bias, null, new[] { dilation }, group, new[] { kernel }, new[] { left, right }, new[] { stride }, options);
        Assert.Equal(OpStatus.Success, plain.Status);
        Assert.Equal(OpStatus.Success, fused.Status);
        Assert.Equal(OpType.Conv, plain.Op);
        Assert.Equal(OpType.ConvRelu, fused.Op);
        Assert.Equal(new[] { batch, outputs, outLength }, plain.Outputs[0].Dims);
        Assert.Equal(expected, ((Tensor<float>)plain.Outputs[0]).ToArray());
        Assert.Equal(expected.Select(v => Math.Max(0f, v)), ((Tensor<float>)fused.Outputs[0]).ToArray());
        Assert.Equal(originalX, xs);
        Assert.Equal(originalW, ws);
    }

    [Fact]
    public void OneDimensionalSameUpperLowerAndPoolPreservePaddingSemantics()
    {
        var x = DenseTensor<float>.OfValues(new float[] { 1, 2, 3, 4, 5 }).Reshape(new[] { 1, 1, 5 });
        var w = DenseTensor<float>.OfValues(new float[] { 1, 1 }).Reshape(new[] { 1, 1, 2 });
        foreach (var (padding, convExpected, poolExpected) in new[]
        {
            ("SAME_UPPER", new float[] { 3, 7, 5 }, new float[] { 2, 4, 5 }),
            ("SAME_LOWER", new float[] { 1, 5, 9 }, new float[] { 1, 3, 5 })
        })
        {
            var conv = CPU.Conv(x, w, null, padding, null, 1, null, null, new[] { 2 }, null);
            var pool = CPU.MaxPool(x, padding, 0, null, new[] { 2 }, null, 0, new[] { 2 }, null);
            Assert.Equal(OpStatus.Success, conv.Status);
            Assert.Equal(OpStatus.Success, pool.Status);
            Assert.Equal(OpType.MaxPool, pool.Op);
            Assert.Equal(convExpected, ((Tensor<float>)conv.Outputs[0]).ToArray());
            Assert.Equal(poolExpected, ((Tensor<float>)pool.Outputs[0]).ToArray());
        }
    }

    [Fact]
    public void TwoDimensionalWeightRankGuardAndConflictingPaddingRemainStrict()
    {
        var x = DenseTensor<float>.OfShape(1, 1, 5);
        var w = DenseTensor<float>.OfShape(1, 1, 2);
        var x4 = DenseTensor<float>.OfShape(1, 1, 1, 5);
        Assert.Equal(OpStatus.Failure, CPU.Conv(x4, w, null, null, null, 1, null, null, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Conv(x, w, null, "SAME_UPPER", null, 1, null, new[] { 0, 0 }, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.MaxPool(x, null, 0, null, new[] { 2 }, null, 1, null, null).Status);
    }

    static float[] Range(float start, float step, int n)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = start + step * i;
        return v;
    }

    static DenseTensor<float> FT(float[] values, params int[] dims) =>
        new DenseTensor<float>(values, dims);

    static void AssertNear(float[] actual, float[] expected, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(actual[i] - expected[i]) < Tol, what + "[" + i + "] drifted: " + actual[i] + " vs " + expected[i] + ".");
    }

    [Fact]
    public void ConvPointwise_MatchesOrt()
    {
        var x = FT(Range(0.5f, 0.5f, 20), 1, 4, 5);
        var w = FT(Range(-2.75f, 0.25f, 24), 6, 4, 1);
        var b = FT(new float[] { -1.25f, -0.75f, -0.25f, 0.25f, 0.75f, 1.25f }, 6);
        var r = CPU.Conv(x, w, b, null, new[] { 1 }, 1, new[] { 1 }, new[] { 0, 0 }, new[] { 1 }, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs![0];
        Assert.Equal(new[] { 1, 6, 5 }, y.Dimensions.ToArray());
        AssertNear(y.ToArray(), new float[]
        {
            -38.5f, -43.25f, -48f, -52.75f, -57.5f, -21f, -23.75f, -26.5f, -29.25f, -32f,
            -3.5f, -4.25f, -5f, -5.75f, -6.5f, 14f, 15.25f, 16.5f, 17.75f, 19f,
            31.5f, 34.75f, 38f, 41.25f, 44.5f, 49f, 54.25f, 59.5f, 64.75f, 70f,
        }, "conv-pointwise");
    }

    [Fact]
    public void ConvDepthwiseGroups_MatchesOrt()
    {
        var x = FT(Range(0.25f, 0.25f, 32), 1, 4, 8);
        var w = FT(Range(-1.25f, 0.25f, 12), 4, 1, 3);
        var r = CPU.Conv(x, w, null, null, new[] { 1 }, 4, new[] { 3 }, new[] { 0, 0 }, new[] { 1 }, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs![0];
        Assert.Equal(new[] { 1, 4, 6 }, y.Dimensions.ToArray());
        AssertNear(y.ToArray(), new float[]
        {
            -1.375f, -2.125f, -2.875f, -3.625f, -4.375f, -5.125f,
            -1.75f, -1.9375f, -2.125f, -2.3125f, -2.5f, -2.6875f,
            6.875f, 7.25f, 7.625f, 8f, 8.375f, 8.75f,
            24.5f, 25.4375f, 26.375f, 27.3125f, 28.25f, 29.1875f,
        }, "conv-depthwise");
    }

    [Fact]
    public void ConvPaddedStridedDilated_MatchesOrt()
    {
        var x = FT(Range(0.5f, 0.5f, 18), 1, 2, 9);
        var w = FT(Range(-2f, 0.25f, 18), 3, 2, 3);
        var b = FT(new float[] { 1f, -1f, 0.5f }, 3);
        var r = CPU.Conv(x, w, b, null, new[] { 2 }, 1, new[] { 3 }, new[] { 1, 1 }, new[] { 2 }, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs![0];
        Assert.Equal(new[] { 1, 3, 4 }, y.Dimensions.ToArray());
        AssertNear(y.ToArray(), new float[]
        {
            -14.125f, -28f, -36.25f, -29.875f, 6.375f, 8.25f, 9f, 2.625f,
            30.375f, 48f, 57.75f, 38.625f,
        }, "conv-pad-stride-dil");
    }

    [Fact]
    public void ConvSameAutoPad_MatchesOrt()
    {
        var x = FT(Range(0.5f, 0.5f, 14), 1, 2, 7);
        var w = FT(Range(-1.25f, 0.25f, 12), 2, 2, 3);
        var r = CPU.Conv(x, w, null, "SAME_UPPER", new[] { 1 }, 1, new[] { 3 }, null, new[] { 2 }, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs![0];
        Assert.Equal(new[] { 1, 2, 4 }, y.Dimensions.ToArray());
        AssertNear(y.ToArray(), new float[] { -2.25f, -7.75f, -11.5f, -12.25f, 12.75f, 21.5f, 26.75f, 17.75f }, "conv-same");
    }

    [Fact]
    public void Conv1D_DoubleMatchesFloatOnExactInts()
    {
        var xf = FT(Range(1f, 1f, 8), 1, 2, 4);
        var wf = FT(Range(1f, 1f, 6), 3, 2, 1);
        var rf = CPU.Conv(xf, wf, null, null, new[] { 1 }, 1, new[] { 1 }, new[] { 0, 0 }, new[] { 1 }, null);
        Assert.Equal(OpStatus.Success, rf.Status);
        var xd = new DenseTensor<double>(Range(1f, 1f, 8).Select(v => (double)v).ToArray(), new[] { 1, 2, 4 });
        var wd = new DenseTensor<double>(Range(1f, 1f, 6).Select(v => (double)v).ToArray(), new[] { 3, 2, 1 });
        var rd = CPU.Conv(xd, wd, null, null, new[] { 1 }, 1, new[] { 1 }, new[] { 0, 0 }, new[] { 1 }, null);
        Assert.Equal(OpStatus.Success, rd.Status);
        Assert.Equal(((Tensor<float>)rf.Outputs![0]).ToArray(), ((Tensor<double>)rd.Outputs![0]).ToArray().Select(v => (float)v).ToArray());
    }

    [Fact]
    public void Conv1D_RankMismatch_Fails()
    {
        var x = FT(Range(0.5f, 0.5f, 20), 1, 4, 5);
        var w4 = FT(Range(0.25f, 0.25f, 24), 6, 4, 1, 1);
        var r = CPU.Conv(x, w4, null, null, new[] { 1 }, 1, new[] { 1 }, new[] { 0, 0 }, new[] { 1 }, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }

    [Fact]
    public void Conv1D_BadAttrRank_Fails()
    {
        var x = FT(Range(0.5f, 0.5f, 20), 1, 4, 5);
        var w = FT(Range(-2.75f, 0.25f, 24), 6, 4, 1);
        var r = CPU.Conv(x, w, null, null, new[] { 1 }, 1, new[] { 1 }, new[] { 0, 0, 0, 0 }, new[] { 1 }, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }

    [Fact]
    public void PoolBasic_MatchesOrt()
    {
        var x = FT(Range(0.5f, 0.5f, 18), 1, 2, 9);
        var r = CPU.MaxPool(x, null, 0, new[] { 1 }, new[] { 3 }, new[] { 0, 0 }, null, new[] { 3 }, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs![0];
        Assert.Equal(new[] { 1, 2, 3 }, y.Dimensions.ToArray());
        AssertNear(y.ToArray(), new float[] { 1.5f, 3f, 4.5f, 6f, 7.5f, 9f }, "pool-basic");
    }

    [Fact]
    public void PoolPaddedCeil_MatchesOrt()
    {
        var x = FT(Range(0.5f, 0.5f, 8), 1, 1, 8);
        var r = CPU.MaxPool(x, null, 1, new[] { 1 }, new[] { 3 }, new[] { 1, 1 }, null, new[] { 2 }, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs![0];
        Assert.Equal(new[] { 1, 1, 5 }, y.Dimensions.ToArray());
        AssertNear(y.ToArray(), new float[] { 1f, 2f, 3f, 4f, 4f }, "pool-pad-ceil");
    }

    [Fact]
    public void PoolDilated_MatchesOrt()
    {
        var x = FT(Range(0.5f, 0.5f, 10), 1, 1, 10);
        var r = CPU.MaxPool(x, null, 0, new[] { 2 }, new[] { 2 }, new[] { 0, 0 }, null, new[] { 1 }, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs![0];
        Assert.Equal(new[] { 1, 1, 8 }, y.Dimensions.ToArray());
        AssertNear(y.ToArray(), new float[] { 1.5f, 2f, 2.5f, 3f, 3.5f, 4f, 4.5f, 5f }, "pool-dilated");
    }

    [Fact]
    public void Pool1D_DoubleMatchesFloatOnExactInts()
    {
        var xf = FT(new float[] { 3f, 1f, 4f, 1f, 5f, 9f }, 1, 1, 6);
        var rf = CPU.MaxPool(xf, null, 0, new[] { 1 }, new[] { 2 }, new[] { 0, 0 }, null, new[] { 2 }, null);
        Assert.Equal(OpStatus.Success, rf.Status);
        var xd = new DenseTensor<double>(new double[] { 3.0, 1.0, 4.0, 1.0, 5.0, 9.0 }, new[] { 1, 1, 6 });
        var rd = CPU.MaxPool(xd, null, 0, new[] { 1 }, new[] { 2 }, new[] { 0, 0 }, null, new[] { 2 }, null);
        Assert.Equal(OpStatus.Success, rd.Status);
        Assert.Equal(((Tensor<float>)rf.Outputs![0]).ToArray(), ((Tensor<double>)rd.Outputs![0]).ToArray().Select(v => (float)v).ToArray());
    }

    [Fact]
    public void Pool1D_MissingKernel_Fails()
    {
        var x = FT(Range(0.5f, 0.5f, 18), 1, 2, 9);
        var r = CPU.MaxPool(x, null, 0, new[] { 1 }, null, new[] { 0, 0 }, null, new[] { 3 }, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }
}
