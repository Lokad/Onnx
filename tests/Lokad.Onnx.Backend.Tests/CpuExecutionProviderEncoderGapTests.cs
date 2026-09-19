using System;
using System.Linq;
using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

// Sigmoid/Floor/And/Not/Pad coverage adapted from voice commit eb36aa5.
// Float expectations are frozen Python
// onnxruntime 1.29.0 oracles (ORT_SEQUENTIAL, intra/inter-op 1,
// ORT_ENABLE_ALL, opset 17) over the complete inputs below. ORT refuses integer Sigmoid,
// Floor, and And inputs, and the kernels match those refusals.
public class CpuExecutionProviderEncoderGapTests
{
    const double Tol = 1e-6;

    static void AssertNear(float[] actual, float[] expected, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(actual[i] - expected[i]) < Tol, what + "[" + i + "] drifted: " + actual[i] + " vs " + expected[i] + ".");
    }

    static void AssertNear(double[] actual, double[] expected, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(actual[i] - expected[i]) < 1e-12, what + "[" + i + "] drifted.");
    }

    [Fact]
    public void Sigmoid_MatchesOrt()
    {
        var x = new DenseTensor<float>(new float[] { -1.5f, -0.5f, 0f, 0.5f, 1.5f, 2.5f }, new[] { 6 });
        var r = CPU.Sigmoid(x, null);
        Assert.Equal(OpStatus.Success, r.Status);
        AssertNear(((Tensor<float>)r.Outputs![0]).ToArray(),
            new float[] { 0.18242556f, 0.3775407f, 0.5f, 0.6224593f, 0.81757444f, 0.9241418f }, "sigmoid");
        var xd = new DenseTensor<double>(new double[] { -1.5, -0.5, 0.0, 0.5, 1.5, 2.5 }, new[] { 6 });
        var rd = CPU.Sigmoid(xd, null);
        Assert.Equal(OpStatus.Success, rd.Status);
        AssertNear(((Tensor<double>)rd.Outputs![0]).ToArray(),
            new double[] { 0.18242552380635635, 0.3775406687981454, 0.5, 0.6224593312018546, 0.8175744761936437, 0.9241418199787566 }, "sigmoid-double");
    }

    [Fact]
    public void Sigmoid_IntRefusedLikeOrt()
    {
        var x = new DenseTensor<int>(new int[] { 1, 2, 3 }, new[] { 3 });
        Assert.Equal(OpStatus.Failure, CPU.Sigmoid(x, null).Status);
    }

    [Fact]
    public void Floor_MatchesOrt()
    {
        var x = new DenseTensor<float>(new float[] { -1.5f, -0.5f, 0f, 0.5f, 1.5f, 2.5f }, new[] { 6 });
        var r = CPU.Floor(x, null);
        Assert.Equal(OpStatus.Success, r.Status);
        AssertNear(((Tensor<float>)r.Outputs![0]).ToArray(),
            new float[] { -2f, -1f, 0f, 0f, 1f, 2f }, "floor");
        var xd = new DenseTensor<double>(new double[] { -1.5, -0.5, 0.0, 0.5, 1.5, 2.5 }, new[] { 6 });
        var rd = CPU.Floor(xd, null);
        Assert.Equal(OpStatus.Success, rd.Status);
        AssertNear(((Tensor<double>)rd.Outputs![0]).ToArray(),
            new double[] { -2.0, -1.0, 0.0, 0.0, 1.0, 2.0 }, "floor-double");
    }

    [Fact]
    public void Floor_IntRefusedLikeOrt()
    {
        var x = new DenseTensor<long>(new long[] { 1L, 2L }, new[] { 2 });
        Assert.Equal(OpStatus.Failure, CPU.Floor(x, null).Status);
    }

    [Fact]
    public void And_MatchesOrtWithBroadcast()
    {
        var a = new DenseTensor<bool>(new bool[] { true, true, false, false }, new[] { 4 });
        var b = new DenseTensor<bool>(new bool[] { true, false, true, false }, new[] { 4 });
        var r = CPU.And(a, b, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new bool[] { true, false, false, false }, ((Tensor<bool>)r.Outputs![0]).ToArray());
        var a2 = new DenseTensor<bool>(new bool[] { true, false }, new[] { 1, 2 });
        var b2 = new DenseTensor<bool>(new bool[] { true, false, true, false }, new[] { 4, 1 });
        var rb = CPU.And(a2, b2, null);
        Assert.Equal(OpStatus.Success, rb.Status);
        var yb = (Tensor<bool>)rb.Outputs![0];
        Assert.Equal(new[] { 4, 2 }, yb.Dimensions.ToArray());
        Assert.Equal(new bool[] { true, false, false, false, true, false, false, false }, yb.ToArray());
    }

    [Fact]
    public void And_NonBoolRefusedLikeOrt()
    {
        var a = new DenseTensor<int>(new int[] { 1, 0, 1 }, new[] { 3 });
        Assert.Equal(OpStatus.Failure, CPU.And(a, a, null).Status);
    }

    [Fact]
    public void And_Unbroadcastable_Fails()
    {
        var a = new DenseTensor<bool>(new bool[] { true, false }, new[] { 2 });
        var b = new DenseTensor<bool>(new bool[] { true, false, true }, new[] { 3 });
        Assert.Equal(OpStatus.Failure, CPU.And(a, b, null).Status);
    }

    [Fact]
    public void Not_MatchesOrt()
    {
        var x = new DenseTensor<bool>(new bool[] { true, true, false, false }, new[] { 4 });
        var r = CPU.Not(x, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new bool[] { false, false, true, true }, ((Tensor<bool>)r.Outputs![0]).ToArray());
    }

    [Fact]
    public void Not_NonBoolRefused()
    {
        var x = new DenseTensor<int>(new int[] { 1 }, new[] { 1 });
        Assert.Equal(OpStatus.Failure, CPU.Not(x, null).Status);
    }

    [Fact]
    public void PadConstant_WithValue_MatchesOrt()
    {
        var d = new DenseTensor<float>(new float[] { 1f, 1.5f, 2f, 2.5f, 3f, 3.5f }, new[] { 2, 3 });
        var p = new DenseTensor<long>(new long[] { 1, 0, 0, 2 }, new[] { 4 });
        var v = new DenseTensor<float>(new float[] { -1f }, new int[0]);
        var r = CPU.Pad(d, p, v, "constant", null, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs![0];
        Assert.Equal(new[] { 3, 5 }, y.Dimensions.ToArray());
        AssertNear(y.ToArray(), new float[]
        {
            -1f, -1f, -1f, -1f, -1f, 1f, 1.5f, 2f, -1f, -1f,
            2.5f, 3f, 3.5f, -1f, -1f,
        }, "pad");
    }

    [Fact]
    public void PadConstant_NegativePadsCropLikeOrt()
    {
        var d = new DenseTensor<float>(new float[] { 1f, 1.5f, 2f, 2.5f, 3f, 3.5f }, new[] { 2, 3 });
        var p = new DenseTensor<long>(new long[] { 0, -1, 0, -1 }, new[] { 4 });
        var r = CPU.Pad(d, p, null, "constant", null, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs![0];
        Assert.Equal(new[] { 2, 1 }, y.Dimensions.ToArray());
        AssertNear(y.ToArray(), new float[] { 1.5f, 3f }, "pad-neg");
    }

    [Fact]
    public void PadConstant_Int64Data_MatchesOrt()
    {
        var d = new DenseTensor<long>(new long[] { 0L, 1L, 2L, 3L, 4L, 5L }, new[] { 2, 3 });
        var p = new DenseTensor<int>(new int[] { 1, 0, 0, 2 }, new[] { 4 });
        var r = CPU.Pad(d, p, null, "constant", null, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<long>)r.Outputs![0];
        Assert.Equal(new[] { 3, 5 }, y.Dimensions.ToArray());
        Assert.Equal(new long[] { 0L, 0L, 0L, 0L, 0L, 0L, 1L, 2L, 0L, 0L, 3L, 4L, 5L, 0L, 0L }, y.ToArray());
    }

    [Fact]
    public void PadConstant_DefaultFillIsZero()
    {
        var d = new DenseTensor<float>(new float[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 });
        var p = new DenseTensor<long>(new long[] { 1, 0, 0, 1 }, new[] { 4 });
        var r = CPU.Pad(d, p, null, null, null, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        AssertNear(((Tensor<float>)r.Outputs![0]).ToArray(),
            new float[] { 0f, 0f, 0f, 1f, 2f, 0f, 3f, 4f, 0f }, "pad-default");
    }

    [Fact]
    public void Pad_ReflectModeMirrorsEndpoints()
    {
        var d = new DenseTensor<float>(new float[] { 1f, 2f }, new[] { 2 });
        var p = new DenseTensor<long>(new long[] { 1, 1 }, new[] { 2 });
        var result = CPU.Pad(d, p, null, "reflect", null, null, null);
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.Equal(new float[] { 2f, 1f, 2f, 1f }, ((Tensor<float>)result.Outputs[0]).ToArray());
        Assert.Equal(OpStatus.Failure, CPU.Pad(d, p, null, "unsupported", null, null, null).Status);
    }

    [Fact]
    public void Pad_BadPadsLength_Fails()
    {
        var d = new DenseTensor<float>(new float[] { 1f, 2f }, new[] { 2 });
        var p = new DenseTensor<long>(new long[] { 1, 1, 1 }, new[] { 3 });
        Assert.Equal(OpStatus.Failure, CPU.Pad(d, p, null, "constant", null, null, null).Status);
    }
}
