using System;
using System.Linq;
using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

public class CpuExecutionProviderOpTests
{
    [Fact]
    public void Add_Sub_Mul_Div_Broadcast_Int32()
    {
        var a = DenseTensor<int>.OfValues(new int[,] { { 1, 2 }, { 3, 4 } });
        var b = DenseTensor<int>.OfValues(new int[] { 10, 20 });

        var add = CPU.Add(a, b, null, null);
        var sub = CPU.Sub(a, b, null);
        var mul = CPU.Mul(a, b, null, null);

        Assert.Equal(OpStatus.Success, add.Status);
        Assert.Equal(11, ((Tensor<int>)add.Outputs![0])[0, 0]);
        Assert.Equal(24, ((Tensor<int>)add.Outputs![0])[1, 1]);
        Assert.Equal(-9, ((Tensor<int>)sub.Outputs![0])[0, 0]);
        Assert.Equal(-16, ((Tensor<int>)sub.Outputs![0])[1, 1]);
        Assert.Equal(10, ((Tensor<int>)mul.Outputs![0])[0, 0]);
        Assert.Equal(80, ((Tensor<int>)mul.Outputs![0])[1, 1]);

        var div = CPU.Div(DenseTensor<float>.OfValues(new float[,] { { 8f, 6f }, { 4f, 2f } }),
            DenseTensor<float>.OfValues(new float[,] { { 2f, 3f }, { 4f, 2f } }), null, null);
        Assert.Equal(OpStatus.Success, div.Status);
        Assert.Equal(4f, ((Tensor<float>)div.Outputs![0])[0, 0], 5);
        Assert.Equal(1f, ((Tensor<float>)div.Outputs![0])[1, 0], 5);
    }

    [Fact]
    public void Pow_Sqrt_Erf_Relu_Float()
    {
        var x = DenseTensor<float>.OfValues(new float[,] { { 4f, 9f }, { -1f, 2f } });
        var y = DenseTensor<float>.OfValues(new float[,] { { 0.5f, 0.5f }, { 2f, 3f } });

        var pow = CPU.Pow(x, y, null);
        Assert.Equal(OpStatus.Success, pow.Status);
        Assert.Equal(2f, ((Tensor<float>)pow.Outputs![0])[0, 0], 5);
        Assert.Equal(3f, ((Tensor<float>)pow.Outputs![0])[0, 1], 5);
        Assert.Equal(1f, ((Tensor<float>)pow.Outputs![0])[1, 0], 5);
        Assert.Equal(8f, ((Tensor<float>)pow.Outputs![0])[1, 1], 5);

        var sqrt = CPU.Sqrt(DenseTensor<float>.OfValues(new float[] { 4f, 9f }), null);
        Assert.Equal(OpStatus.Success, sqrt.Status);
        Assert.Equal(2f, ((Tensor<float>)sqrt.Outputs![0])[0], 5);
        Assert.Equal(3f, ((Tensor<float>)sqrt.Outputs![0])[1], 5);

        var relu = CPU.Relu(DenseTensor<float>.OfValues(new float[] { -1f, 0f, 2f }), null);
        Assert.Equal(OpStatus.Success, relu.Status);
        Assert.Equal(0f, ((Tensor<float>)relu.Outputs![0])[0], 5);
        Assert.Equal(2f, ((Tensor<float>)relu.Outputs![0])[2], 5);

        var erf = CPU.Erf(DenseTensor<float>.OfValues(new float[] { 0f, 1f }), null, null);
        Assert.Equal(OpStatus.Success, erf.Status);
        Assert.Equal(0f, ((Tensor<float>)erf.Outputs![0])[0], 5);
        Assert.Equal(0.8427f, ((Tensor<float>)erf.Outputs![0])[1], 3);
    }

    [Fact]
    public void Erf_Gelu_Double()
    {
        // No ORT CPU reference exists (double Erf/Gelu are NOT_IMPLEMENTED);
        // expected values are math.erf facts, and the A&S 7.1.26 kernel stays
        // within 6-decimal agreement (verified against a formula replica).
        var erf = CPU.Erf(DenseTensor<double>.OfValues(new double[] { 0.0, 0.5, 1.0 }), null, null);
        Assert.Equal(OpStatus.Success, erf.Status);
        Assert.Equal(0.0, ((Tensor<double>)erf.Outputs![0])[0], 6);
        Assert.Equal(0.5204998778, ((Tensor<double>)erf.Outputs![0])[1], 6);
        Assert.Equal(0.8427007929, ((Tensor<double>)erf.Outputs![0])[2], 6);
        var gelu = CPU.Gelu(DenseTensor<double>.OfValues(new double[] { 0.0, 0.5, 1.0, -1.0 }), null, null, null);
        Assert.Equal(OpStatus.Success, gelu.Status);
        Assert.Equal(0.0, ((Tensor<double>)gelu.Outputs![0])[0], 6);
        Assert.Equal(0.3457312306, ((Tensor<double>)gelu.Outputs![0])[1], 6);
        Assert.Equal(0.8413447461, ((Tensor<double>)gelu.Outputs![0])[2], 6);
        Assert.Equal(-0.1586552539, ((Tensor<double>)gelu.Outputs![0])[3], 6);
    }

    [Fact]
    public void Tanh_Softmax_Double()
    {
        // ORT 1.29: tanh([0, 0.5, 1, -1]) and softmax rows [0.0900305732, 0.2447284711, 0.6652409558].
        var tanh = CPU.Tanh(DenseTensor<double>.OfValues(new double[] { 0.0, 0.5, 1.0, -1.0 }), null);
        Assert.Equal(OpStatus.Success, tanh.Status);
        Assert.Equal(0.0, ((Tensor<double>)tanh.Outputs![0])[0], 6);
        Assert.Equal(0.4621171573, ((Tensor<double>)tanh.Outputs![0])[1], 6);
        Assert.Equal(0.7615941560, ((Tensor<double>)tanh.Outputs![0])[2], 6);
        Assert.Equal(-0.7615941560, ((Tensor<double>)tanh.Outputs![0])[3], 6);
        var sm = CPU.Softmax(DenseTensor<double>.OfValues(new double[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } }), -1, null, null, 13);
        Assert.Equal(OpStatus.Success, sm.Status);
        var y = ((Tensor<double>)sm.Outputs![0]).ToArray();
        double[] row = new double[] { 0.0900305732, 0.2447284711, 0.6652409558 };
        for (int i = 0; i < 6; i++) Assert.Equal(row[i % 3], y[i], 6);
    }

    [Fact]
    public void MatMul_Transpose_Softmax()
    {
        var a = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f } });
        var b = DenseTensor<float>.OfValues(new float[,] { { 5f, 6f }, { 7f, 8f } });

        var mm = CPU.MatMul(a, b, null, null);
        Assert.Equal(OpStatus.Success, mm.Status);
        var mmOut = (Tensor<float>)mm.Outputs![0];
        Assert.Equal(19f, mmOut[0, 0], 5);
        Assert.Equal(50f, mmOut[1, 1], 5);

        var trans = CPU.Transpose(a, new[] { 1, 0 }, null, null);
        Assert.Equal(OpStatus.Success, trans.Status);
        var t = (Tensor<float>)trans.Outputs![0];
        Assert.Equal(2f, t[1, 0], 5);
        Assert.Equal(3f, t[0, 1], 5);

        var logits = DenseTensor<float>.OfValues(new float[,] { { 0f, 1f }, { 1f, 1f } });
        var sm = CPU.Softmax(logits, 1, null, null, 13);
        Assert.Equal(OpStatus.Success, sm.Status);
        var smOut = (Tensor<float>)sm.Outputs![0];
        Assert.Equal(1f, smOut[0, 0] + smOut[0, 1], 5);
        Assert.Equal(0.5f, smOut[1, 0], 5);
        Assert.Equal(0.5f, smOut[1, 1], 5);
    }

    [Fact]
    public void Conv_MaxPool_Reduce_Concat_Gather_Slice_Unsqueeze_Cast_Constant()
    {
        var x = DenseTensor<float>.OfValues(new float[1, 1, 3, 3] { { {
            { 1f, 2f, 3f }, { 4f, 5f, 6f }, { 7f, 8f, 9f }
        } } });
        var w = DenseTensor<float>.OfValues(new float[1, 1, 2, 2] { { { { 1f, 1f }, { 1f, 1f } } } });
        var b = DenseTensor<float>.OfValues(new float[] { 0f });

        var conv = CPU.Conv(x, w, b, auto_pad: "VALID", dilations: null, group: null, kernel_shape: null, pads: null, strides: null, options: null);
        Assert.Equal(OpStatus.Success, conv.Status);
        var convOut = (Tensor<float>)conv.Outputs![0];
        Assert.Equal(new[] { 1, 1, 2, 2 }, convOut.Dimensions.ToArray());
        Assert.Equal(12f, convOut[0, 0, 0, 0], 5);
        Assert.Equal(16f, convOut[0, 0, 0, 1], 5);

        var pool = CPU.MaxPool(x, auto_pad: "VALID", kernel_shape: new[] { 2, 2 }, ceil_mode: null, dilations: null, pads: null, storage_order: null, strides: null, options: null);
        Assert.Equal(OpStatus.Success, pool.Status);
        var poolOut = (Tensor<float>)pool.Outputs![0];
        Assert.Equal(new[] { 1, 1, 1, 1 }, poolOut.Dimensions.ToArray());
        Assert.Equal(5f, poolOut[0, 0, 0, 0], 5);

        var data = DenseTensor<float>.OfValues(new float[2, 2] { { 1f, 2f }, { 3f, 4f } });
        var axes = DenseTensor<int>.OfValues(new int[] { 1 });
        var rsum = CPU.ReduceSum(data, axes, 0, 0, null);
        Assert.Equal(OpStatus.Success, rsum.Status);
        var rsumOut = (Tensor<float>)rsum.Outputs![0];
        Assert.Equal(new[] { 2 }, rsumOut.Dimensions.ToArray());
        Assert.Equal(3f, rsumOut[0], 5);

        var rmean = CPU.ReduceMean(data, axes, 0, 0, null);
        Assert.Equal(OpStatus.Success, rmean.Status);
        var rmeanOut = (Tensor<float>)rmean.Outputs![0];
        Assert.Equal(1.5f, rmeanOut[0], 5);

        var rmax = CPU.ReduceMax(data, axes, 0, null, null);
        Assert.Equal(OpStatus.Success, rmax.Status);
        var rmaxOut = (Tensor<float>)rmax.Outputs![0];
        Assert.Equal(2f, rmaxOut[0], 5);

        var c1 = DenseTensor<int>.OfValues(new int[,] { { 1, 2 } });
        var c2 = DenseTensor<int>.OfValues(new int[,] { { 3, 4 } });
        var concat = CPU.Concat(new ITensor[] { c1, c2 }, 0, null);
        Assert.Equal(OpStatus.Success, concat.Status);
        var concatOut = (Tensor<int>)concat.Outputs![0];
        Assert.Equal(new[] { 2, 2 }, concatOut.Dimensions.ToArray());
        Assert.Equal(3, concatOut[1, 0]);

        var gdata = DenseTensor<int>.OfValues(new int[,] { { 10, 20 }, { 30, 40 } });
        var gidx = DenseTensor<int>.OfValues(new int[] { 1, 0 });
        var gather = CPU.Gather(gdata, gidx, 0, null);
        Assert.Equal(OpStatus.Success, gather.Status);
        var gatherOut = (Tensor<int>)gather.Outputs![0];
        Assert.Equal(30, gatherOut[0, 0]);
        Assert.Equal(10, gatherOut[1, 0]);

        var sdata = DenseTensor<int>.OfValues(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });
        var starts = DenseTensor<int>.OfValues(new int[] { 0, 1 });
        var ends = DenseTensor<int>.OfValues(new int[] { 2, 3 });
        var slice = CPU.Slice(sdata, starts, ends, null, null, null);
        Assert.Equal(OpStatus.Success, slice.Status);
        var sliceOut = (Tensor<int>)slice.Outputs![0];
        Assert.Equal(new[] { 2, 2 }, sliceOut.Dimensions.ToArray());
        Assert.Equal(2, sliceOut[0, 0]);
        Assert.Equal(6, sliceOut[1, 1]);

        var unsq = CPU.Unsqueeze(DenseTensor<int>.OfValues(new int[] { 7, 8 }), new[] { 0 }, null);
        Assert.Equal(OpStatus.Success, unsq.Status);
        var unsqOut = (Tensor<int>)unsq.Outputs![0];
        Assert.Equal(new[] { 1, 2 }, unsqOut.Dimensions.ToArray());
        Assert.Equal(7, unsqOut[0, 0]);

        var cast = CPU.Cast(DenseTensor<float>.OfValues(new float[] { 1.2f, 2.6f }), TensorElementType.Int32, null); // ONNX truncates toward zero: 2.6 -> 2
        Assert.Equal(OpStatus.Success, cast.Status);
        var castOut = (Tensor<int>)cast.Outputs![0];
        Assert.Equal(1, castOut[0]);
        Assert.Equal(2, castOut[1]);

        var cst = CPU.Constant(42, null);
        Assert.Equal(OpStatus.Success, cst.Status);
        var cstOut = (Tensor<int>)cst.Outputs![0];
        Assert.Equal(42, cstOut[0]);
    }

    [Fact]
    public void Equal_Where_Expand_Resize()
    {
        var a = DenseTensor<long>.OfValues(new long[,] { { 1, 2, 3 } });
        var b = DenseTensor<long>.OfValues(new long[] { 1, 0, 3 });
        var eq = CPU.Equal(a, b, null);
        Assert.Equal(OpStatus.Success, eq.Status);
        var eqOut = (Tensor<bool>)eq.Outputs![0];
        Assert.True(eqOut[0, 0]);
        Assert.False(eqOut[0, 1]);

        var cond = DenseTensor<bool>.OfValues(new bool[,] { { true }, { false } });
        var x = DenseTensor<long>.OfValues(new long[,] { { 7, 8, 9 }, { 10, 11, 12 } });
        var y = DenseTensor<long>.OfValues(new long[] { 1, 1, 1 });
        var where = CPU.Where(cond, x, y, null);
        Assert.Equal(OpStatus.Success, where.Status);
        var whereOut = (Tensor<long>)where.Outputs![0];
        Assert.Equal(7, whereOut[0, 0]);
        Assert.Equal(1, whereOut[1, 2]);

        var expand = CPU.Expand(DenseTensor<float>.OfValues(new float[1, 1, 3] { { { 1f, 2f, 3f } } }),
            DenseTensor<long>.OfValues(new long[] { 2, 1, 3 }), null);
        Assert.Equal(OpStatus.Success, expand.Status);
        var expandOut = (Tensor<float>)expand.Outputs![0];
        Assert.Equal(new[] { 2, 1, 3 }, expandOut.Dimensions.ToArray());
        Assert.Equal(2f, expandOut[1, 0, 1], 5);

        var resizeInput = DenseTensor<float>.OfShape(1, 1, 2, 2);
        resizeInput.Fill(2f);
        var sizes = DenseTensor<int>.OfValues(new int[] { 1, 1, 3, 3 });
        var resize = CPU.Resize(resizeInput, null, null, sizes, "cubic", "half_pixel", "floor", -0.75f, 0f, null);
        Assert.Equal(OpStatus.Success, resize.Status);
        var resizeOut = (Tensor<float>)resize.Outputs![0];
        Assert.Equal(new[] { 1, 1, 3, 3 }, resizeOut.Dimensions.ToArray());
        Assert.Equal(2f, resizeOut[0, 0, 1, 1], 5);
    }
}




