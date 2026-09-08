using System;
using System.Linq;
using Xunit;

namespace Lokad.Onnx.Tensors.Tests;

public class BroadcastKernelTests
{
    static DenseTensor<float> Vector100K()
    {
        var data = new float[100000];
        for (int i = 0; i < data.Length; i++) data[i] = i + 1f;
        return new DenseTensor<float>(data, new[] { 100000 });
    }

    static void AssertSuccess(OpResult r) => Assert.Equal(OpStatus.Success, r.Status);

    [Fact]
    public void ScalarDivision_Values_And_Allocation()
    {
        var x = Vector100K();
        var s = new DenseTensor<float>(new float[] { 2f }, new[] { 1 });
        var r = CPUExecutionProvider.Div(x, s, null, null);
        AssertSuccess(r);
        var y = (Tensor<float>)r.Outputs[0];
        Assert.Equal(0.5f, y.GetValue(0));
        Assert.Equal(50000f, y.GetValue(99999));
        foreach (var mode in new[] { OptimizationMode.Speed, OptimizationMode.Memory })
        {
            var opts = new ExecutionOptions(mode, TensorExecutionOptions.Simd);
            for (int i = 0; i < 5; i++) CPUExecutionProvider.Div(x, s, opts, null);
            long before = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 10; i++) CPUExecutionProvider.Div(x, s, opts, null);
            long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            Assert.True(allocated < 6000000L, $"scalar div allocated {allocated} bytes for 10x400KB payloads in {mode} mode");
        }
    }

    [Fact]
    public void ScalarOnLeft_RespectsOperandOrder()
    {
        var t = DenseTensor<float>.OfValues(new float[] { 4f, 2f, 1f });
        var two = new DenseTensor<float>(new float[] { 2f }, new[] { 1 });
        var sub = (Tensor<float>)CPUExecutionProvider.Sub(two, t, null).Outputs[0];
        Assert.Equal(new float[] { -2f, 0f, 1f }, sub.ToArray());
        var div = (Tensor<float>)CPUExecutionProvider.Div(two, t, null, null).Outputs[0];
        Assert.Equal(new float[] { 0.5f, 1f, 2f }, div.ToArray());
        var pow = (Tensor<float>)CPUExecutionProvider.Pow(t, two, null).Outputs[0];
        Assert.Equal(new float[] { 16f, 4f, 1f }, pow.ToArray());
        var add = (Tensor<float>)CPUExecutionProvider.Add(two, t, null, null).Outputs[0];
        Assert.Equal(new float[] { 6f, 4f, 3f }, add.ToArray());
        var mul = (Tensor<float>)CPUExecutionProvider.Mul(two, t, null, null).Outputs[0];
        Assert.Equal(new float[] { 8f, 4f, 2f }, mul.ToArray());
    }

    [Fact]
    public void BiasAdd_Values_And_Allocation()
    {
        var mat = new DenseTensor<float>(Enumerable.Range(0, 25000 * 4).Select(i => (float)i).ToArray(), new[] { 25000, 4 });
        var bias = new DenseTensor<float>(new float[] { 1f, 2f, 3f, 4f }, new[] { 4 });
        var r = CPUExecutionProvider.Add(mat, bias, null, null);
        AssertSuccess(r);
        var y = (Tensor<float>)r.Outputs[0];
        Assert.Equal(new int[] { 25000, 4 }, y.Dimensions.ToArray());
        Assert.Equal(0f + 1f, y.GetValue(0));
        Assert.Equal(7f + 4f, y.GetValue(7));
        var opts = new ExecutionOptions(OptimizationMode.Speed, TensorExecutionOptions.Simd);
        for (int i = 0; i < 5; i++) CPUExecutionProvider.Add(mat, bias, opts, null);
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 10; i++) CPUExecutionProvider.Add(mat, bias, opts, null);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.True(allocated < 5000000L, $"bias add allocated {allocated} bytes for 10x400KB payloads");
    }

    [Fact]
    public void AttentionMask_And_NestedViews_Values()
    {
        var scores = new DenseTensor<float>(Enumerable.Range(0, 2 * 3 * 8).Select(i => (float)i).ToArray(), new[] { 2, 3, 8 });
        var mask = new DenseTensor<float>(Enumerable.Repeat(10f, 8).ToArray(), new[] { 1, 8 });
        var m = (Tensor<float>)CPUExecutionProvider.Mul(scores, mask, null, null).Outputs[0];
        Assert.Equal(new int[] { 2, 3, 8 }, m.Dimensions.ToArray());
        Assert.Equal(0f * 10f, m.GetValue(0));
        Assert.Equal(47f * 10f, m.GetValue(47));
        var x = DenseTensor<float>.OfValues(Enumerable.Range(1, 8).Select(i => (float)i).ToArray());
        var nested = x.InsertDim(0).BroadcastDim(0, 2).InsertDim(1).BroadcastDim(1, 3);
        Assert.Equal(new int[] { 2, 3, 8 }, nested.Dimensions.ToArray());
        var ones = DenseTensor<float>.Ones(2, 3, 8);
        var n = (Tensor<float>)CPUExecutionProvider.Add(nested, ones, null, null).Outputs[0];
        Assert.Equal(1f + 1f, n.GetValue(0));
        Assert.Equal(8f + 1f, n.GetValue(7));
        Assert.Equal(8f + 1f, n.GetValue(47));
    }

    [Fact]
    public void IncompatibleShapes_ReportFailure()
    {
        var a = DenseTensor<float>.OfValues(new float[] { 1f, 2f });
        var b = DenseTensor<float>.OfValues(new float[] { 1f, 2f, 3f });
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Add(a, b, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Div(a, b, null, null).Status);
    }

    [Fact]
    public void DestinationOverload_WritesBroadcastOnce()
    {
        var mat = new DenseTensor<float>(new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f }, new[] { 2, 4 });
        var bias = new DenseTensor<float>(new float[] { 10f, 20f, 30f, 40f }, new[] { 4 });
        var dest = new DenseTensor<float>(Enumerable.Repeat(-1f, 8).ToArray(), new[] { 2, 4 });
        Tensor<float>.Add(mat, bias, dest, TensorExecutionOptions.Simd);
        Assert.Equal(new float[] { 11f, 22f, 33f, 44f, 15f, 26f, 37f, 48f }, dest.ToArray());
        var s = new DenseTensor<float>(new float[] { 2f }, new[] { 1 });
        var dest2 = DenseTensor<float>.Zeros(2, 4);
        Tensor<float>.Multiply(mat, s, dest2, TensorExecutionOptions.Scalar);
        Assert.Equal(new float[] { 2f, 4f, 6f, 8f, 10f, 12f, 14f, 16f }, dest2.ToArray());
    }
}
