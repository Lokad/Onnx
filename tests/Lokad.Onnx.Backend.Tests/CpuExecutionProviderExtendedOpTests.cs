using System;
using System.Linq;
using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

public class CpuExecutionProviderExtendedOpTests
{
    [Fact]
    public void Gemm_AlphaBetaBiasForms()
    {
        var a = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f } });
        var b = DenseTensor<float>.OfValues(new float[,] { { 5f, 6f }, { 7f, 8f } });
        var r = CPU.Gemm(a, b, null, 1f, 1f);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs![0];
        Assert.Equal(new[] { 2, 2 }, y.Dimensions.ToArray());
        Assert.Equal(19f, y[0, 0], 4);
        Assert.Equal(22f, y[0, 1], 4);
        Assert.Equal(43f, y[1, 0], 4);
        Assert.Equal(50f, y[1, 1], 4);

        var bias = DenseTensor<float>.OfValues(new float[] { 100f, 200f });
        var rb = CPU.Gemm(a, b, bias, 1f, 1f);
        Assert.Equal(OpStatus.Success, rb.Status);
        var yb = (Tensor<float>)rb.Outputs![0];
        Assert.Equal(119f, yb[0, 0], 4);
        Assert.Equal(222f, yb[0, 1], 4);

        var rs = CPU.Gemm(a, b, null, 2f, 0f);
        Assert.Equal(OpStatus.Success, rs.Status);
        Assert.Equal(38f, ((Tensor<float>)rs.Outputs![0])[0, 0], 4);
    }

    [Fact]
    public void Tanh_MatchesMath()
    {
        var x = DenseTensor<float>.OfValues(new float[] { 0f, 1f, -1f });
        var r = CPU.Tanh(x);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs![0];
        Assert.Equal(0f, y[0], 5);
        Assert.Equal(MathF.Tanh(1f), y[1], 5);
        Assert.Equal(MathF.Tanh(-1f), y[2], 5);
    }

    [Fact]
    public void Split_ExplicitSizes()
    {
        var x = DenseTensor<float>.OfShape(1, 2, 6);
        for (int i = 0; i < 12; i++) x.SetValue(i, i);
        var sizes = DenseTensor<long>.OfValues(new long[] { 2L, 4L });
        var r = CPU.Split(x, sizes, 2, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(2, r.Outputs!.Length);
        var p0 = (Tensor<float>)r.Outputs[0];
        var p1 = (Tensor<float>)r.Outputs[1];
        Assert.Equal(new[] { 1, 2, 2 }, p0.Dimensions.ToArray());
        Assert.Equal(new[] { 1, 2, 4 }, p1.Dimensions.ToArray());
        Assert.Equal(0f, p0[0, 0, 0], 5);
        Assert.Equal(1f, p0[0, 0, 1], 5);
        Assert.Equal(2f, p1[0, 0, 0], 5);
        Assert.Equal(8f, p1[0, 1, 0], 5);
    }

    [Fact]
    public void Less_Int64()
    {
        var a = DenseTensor<long>.OfValues(new long[] { 1L, 5L, 3L });
        var b = DenseTensor<long>.OfValues(new long[] { 2L, 4L, 3L });
        var r = CPU.Less(a, b);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<bool>)r.Outputs![0];
        Assert.True(y[0]);
        Assert.False(y[1]);
        Assert.False(y[2]);
    }

    [Fact]
    public void ConstantOfShape_DefaultAndTyped()
    {
        var shape = DenseTensor<long>.OfValues(new long[] { 2L, 3L });
        var r = CPU.ConstantOfShape(shape, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs![0];
        Assert.Equal(new[] { 2, 3 }, y.Dimensions.ToArray());
        Assert.Equal(0f, y[1, 2], 5);

        var one = DenseTensor<long>.OfValues(new long[] { 7L });
        var r2 = CPU.ConstantOfShape(shape, one);
        Assert.Equal(OpStatus.Success, r2.Status);
        Assert.Equal(7L, ((Tensor<long>)r2.Outputs![0])[1, 2]);
    }

    [Fact]
    public void GlobalAveragePool_ReducesSpatial()
    {
        var x = DenseTensor<float>.OfShape(1, 2, 2, 2);
        for (int i = 0; i < 8; i++) x.SetValue(i, i);
        var r = CPU.GlobalAveragePool(x);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs![0];
        Assert.Equal(new[] { 1, 2, 1, 1 }, y.Dimensions.ToArray());
        Assert.Equal(1.5f, y[0, 0, 0, 0], 5);
        Assert.Equal(5.5f, y[0, 1, 0, 0], 5);
    }
}
