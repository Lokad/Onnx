using System;
using System.Linq;
using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

// InstanceNormalization/Clip/LeakyRelu/LogSoftmax coverage for the
// pyannote segmentation gaps (PLAN.md Milestone 3). Float expectations are
// frozen Python onnxruntime 1.29.0 oracles (ORT_SEQUENTIAL, intra/inter-op
// 1, ORT_ENABLE_ALL, opset 17 unless noted) from the ignored
// .agent/voice-probe/gen_seg_oracle.py. ORT refuses integer LeakyRelu and
// has no double InstanceNormalization kernel, and the kernels match both.
public class CpuExecutionProviderSegGapTests
{
    const double Tol = 1e-6;

    static float[] Range(float start, float step, int n)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = start + step * i;
        return v;
    }

    static void AssertNear(float[] actual, float[] expected, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(actual[i] - expected[i]) < Tol, what + "[" + i + "] drifted: " + actual[i] + " vs " + expected[i] + ".");
    }

    [Fact]
    public void InstanceNorm_MatchesOrt()
    {
        var x = new DenseTensor<float>(Range(-5.75f, 0.5f, 24), new[] { 2, 3, 4 });
        var s = new DenseTensor<float>(new float[] { 1f, -1f, 0.5f }, new[] { 3 });
        var b = new DenseTensor<float>(new float[] { 0.25f, -0.25f, 0f }, new[] { 3 });
        var r = CPU.InstanceNorm(x, s, b, 1e-5f, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs![0];
        Assert.Equal(new[] { 2, 3, 4 }, y.Dimensions.ToArray());
        AssertNear(y.ToArray(), new float[]
        {
            -1.0916195f, -0.1972065f, 0.6972065f, 1.5916195f, 1.0916195f, 0.1972065f, -0.697206f, -1.591619f,
            -0.6708096f, -0.22360325f, 0.22360325f, 0.6708096f, -1.0916193f, -0.1972065f, 0.6972065f, 1.5916193f,
            1.091619f, 0.19720602f, -0.6972065f, -1.5916195f, -0.67080975f, -0.22360325f, 0.22360325f, 0.67080975f,
        }, "instnorm");
    }

    [Fact]
    public void InstanceNorm4D_MatchesOrt()
    {
        var x = new DenseTensor<float>(Range(-5.75f, 0.5f, 24), new[] { 1, 2, 3, 4 });
        var s = new DenseTensor<float>(new float[] { 2f, 0.5f }, new[] { 2 });
        var b = new DenseTensor<float>(new float[] { -1f, 1f }, new[] { 2 });
        var r = CPU.InstanceNorm(x, s, b, 0.001f, null);
        Assert.Equal(OpStatus.Success, r.Status);
        AssertNear(((Tensor<float>)r.Outputs![0]).ToArray(), new float[]
        {
            -4.185975f, -3.606707f, -3.0274386f, -2.4481707f, -1.8689022f, -1.2896342f, -0.7103658f, -0.1310978f,
            0.44817066f, 1.0274388f, 1.606707f, 2.185975f, 0.2035062f, 0.34832326f, 0.4931403f, 0.63795733f,
            0.78277445f, 0.92759144f, 1.0724086f, 1.2172256f, 1.3620427f, 1.5068597f, 1.6516768f, 1.7964938f,
        }, "instnorm4d");
    }

    [Fact]
    public void InstanceNorm_DoubleRefusedLikeOrt()
    {
        var x = new DenseTensor<double>(new double[] { 1.0, 2.0 }, new[] { 1, 2 });
        var s = new DenseTensor<double>(new double[] { 1.0, 1.0 }, new[] { 2 });
        var b = new DenseTensor<double>(new double[] { 0.0, 0.0 }, new[] { 2 });
        Assert.Equal(OpStatus.Failure, CPU.InstanceNorm(x, s, b, 1e-5f, null).Status);
    }

    [Fact]
    public void InstanceNorm_BadScaleShape_Fails()
    {
        var x = new DenseTensor<float>(Range(-5.75f, 0.5f, 24), new[] { 2, 3, 4 });
        var s = new DenseTensor<float>(new float[] { 1f, 1f }, new[] { 2 });
        var b = new DenseTensor<float>(new float[] { 0f, 0f, 0f }, new[] { 3 });
        Assert.Equal(OpStatus.Failure, CPU.InstanceNorm(x, s, b, 1e-5f, null).Status);
    }

    [Fact]
    public void Clip_MatchesOrt()
    {
        var x = new DenseTensor<float>(new float[] { -1.5f, -0.5f, 0f, 0.5f, 1.5f, 2.5f }, new[] { 6 });
        var lo = new DenseTensor<float>(new float[] { -1f }, new int[0]);
        var hi = new DenseTensor<float>(new float[] { 1f }, new int[0]);
        var r = CPU.Clip(x, lo, hi, null, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        AssertNear(((Tensor<float>)r.Outputs![0]).ToArray(),
            new float[] { -1f, -0.5f, 0f, 0.5f, 1f, 1f }, "clip");
    }

    [Fact]
    public void Clip_OpenEndedAndAttrForm_MatchOrt()
    {
        var x = new DenseTensor<float>(new float[] { -1.5f, -0.5f, 0f, 0.5f, 1.5f, 2.5f }, new[] { 6 });
        var lo = new DenseTensor<float>(new float[] { 0f }, new int[0]);
        var r = CPU.Clip(x, lo, null, null, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        AssertNear(((Tensor<float>)r.Outputs![0]).ToArray(),
            new float[] { 0f, 0f, 0f, 0.5f, 1.5f, 2.5f }, "clip-nomax");
        var ra = CPU.Clip(x, null, null, -1f, 1f, null);
        Assert.Equal(OpStatus.Success, ra.Status);
        AssertNear(((Tensor<float>)ra.Outputs![0]).ToArray(),
            new float[] { -1f, -0.5f, 0f, 0.5f, 1f, 1f }, "clip-attr");
    }

    [Fact]
    public void Clip_Int_MatchesOrt()
    {
        var x = new DenseTensor<int>(new int[] { 1, 5, -3 }, new[] { 3 });
        var lo = new DenseTensor<int>(new int[] { 0 }, new int[0]);
        var hi = new DenseTensor<int>(new int[] { 4 }, new int[0]);
        var r = CPU.Clip(x, lo, hi, null, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new int[] { 1, 4, 0 }, ((Tensor<int>)r.Outputs![0]).ToArray());
        var xd = new DenseTensor<long>(new long[] { 5L, -3L, 0L }, new[] { 3 });
        var lod = new DenseTensor<long>(new long[] { -2L }, new int[0]);
        var hid = new DenseTensor<long>(new long[] { 4L }, new int[0]);
        var rd = CPU.Clip(xd, lod, hid, null, null, null);
        Assert.Equal(OpStatus.Success, rd.Status);
        Assert.Equal(new long[] { 4L, -2L, 0L }, ((Tensor<long>)rd.Outputs![0]).ToArray());
    }

    [Fact]
    public void Clip_DualBoundsSource_Fails()
    {
        var x = new DenseTensor<float>(new float[] { 1f }, new[] { 1 });
        var lo = new DenseTensor<float>(new float[] { 0f }, new int[0]);
        Assert.Equal(OpStatus.Failure, CPU.Clip(x, lo, null, -1f, null, null).Status);
    }

    [Fact]
    public void LeakyRelu_MatchesOrt()
    {
        var x = new DenseTensor<float>(new float[] { -2f, -0.5f, 0f, 0.5f, 2f }, new[] { 5 });
        var r = CPU.LeakyRelu(x, 0.01f, null);
        Assert.Equal(OpStatus.Success, r.Status);
        AssertNear(((Tensor<float>)r.Outputs![0]).ToArray(),
            new float[] { -0.02f, -0.005f, 0f, 0.5f, 2f }, "leaky");
        var xd = new DenseTensor<double>(new double[] { -2.0, -0.5, 0.0, 0.5, 2.0 }, new[] { 5 });
        var rd = CPU.LeakyRelu(xd, 0.1f, null);
        Assert.Equal(OpStatus.Success, rd.Status);
        var actual = ((Tensor<double>)rd.Outputs![0]).ToArray();
        var expected = new double[] { -0.20000000298023224, -0.05000000074505806, 0.0, 0.5, 2.0 };
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(actual[i] - expected[i]) < 1e-9, "leaky-double[" + i + "] drifted.");
    }

    [Fact]
    public void LeakyRelu_IntRefusedLikeOrt()
    {
        var x = new DenseTensor<int>(new int[] { 1, -2, 3 }, new[] { 3 });
        Assert.Equal(OpStatus.Failure, CPU.LeakyRelu(x, 0.01f, null).Status);
    }

    [Fact]
    public void LogSoftmax_MatchesOrt()
    {
        var x = new DenseTensor<float>(new float[] { 1f, 2f, 3f, 1f, 1f, 1f }, new[] { 2, 3 });
        var r = CPU.LogSoftmax(x, -1, null, null, 17);
        Assert.Equal(OpStatus.Success, r.Status);
        AssertNear(((Tensor<float>)r.Outputs![0]).ToArray(),
            new float[] { -2.407606f, -1.4076059f, -0.40760595f, -1.0986123f, -1.0986123f, -1.0986123f }, "logsoftmax");
        var ra = CPU.LogSoftmax(x, 0, null, null, 17);
        Assert.Equal(OpStatus.Success, ra.Status);
        AssertNear(((Tensor<float>)ra.Outputs![0]).ToArray(),
            new float[] { -0.6931472f, -0.31326166f, -0.12692805f, -0.6931472f, -1.3132616f, -2.126928f }, "logsoftmax-axis0");
        var xd = new DenseTensor<double>(new double[] { 1.0, 2.0, 3.0, 0.5, -0.5, 0.0 }, new[] { 2, 3 });
        var rd = CPU.LogSoftmax(xd, -1, null, null, 17);
        Assert.Equal(OpStatus.Success, rd.Status);
        var actual = ((Tensor<double>)rd.Outputs![0]).ToArray();
        var expected = new double[] { -2.40760596444438, -1.4076059644443801, -0.40760596444438024, -0.6802696706417346, -1.6802696706417346, -1.1802696706417346 };
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(actual[i] - expected[i]) < 1e-9, "logsoftmax-double[" + i + "] drifted.");
    }

    [Fact]
    public void LogSoftmax_BadAxis_ThrowsLikeSoftmax()
    {
        // Axis validation throws ArgumentException at the tensor-kernel
        // level (the Softmax precedent); graph dispatch converts it into a
        // clean node failure.
        var x = new DenseTensor<float>(new float[] { 1f, 2f }, new[] { 2 });
        Assert.Throws<ArgumentException>(() => CPU.LogSoftmax(x, 5, null, null, 17));
    }
}
