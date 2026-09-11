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
    public void Add_Sub_Mul_Double()
    {
        // ORT 1.29 on [[1,2],[3,4]] and [[5,6],[7,8]], plus row broadcast.
        var a = DenseTensor<double>.OfValues(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });
        var b = DenseTensor<double>.OfValues(new double[,] { { 5.0, 6.0 }, { 7.0, 8.0 } });
        var add = CPU.Add(a, b, null, null);
        Assert.Equal(OpStatus.Success, add.Status);
        Assert.Equal(new double[] { 6.0, 8.0, 10.0, 12.0 }, ((Tensor<double>)add.Outputs![0]).ToArray());
        var sub = CPU.Sub(a, b, null);
        Assert.Equal(OpStatus.Success, sub.Status);
        Assert.Equal(new double[] { -4.0, -4.0, -4.0, -4.0 }, ((Tensor<double>)sub.Outputs![0]).ToArray());
        var mul = CPU.Mul(a, b, null, null);
        Assert.Equal(OpStatus.Success, mul.Status);
        Assert.Equal(new double[] { 5.0, 12.0, 21.0, 32.0 }, ((Tensor<double>)mul.Outputs![0]).ToArray());
        var bc = CPU.Add(a, DenseTensor<double>.OfValues(new double[] { 10.0, 20.0 }), null, null);
        Assert.Equal(OpStatus.Success, bc.Status);
        Assert.Equal(new double[] { 11.0, 22.0, 13.0, 24.0 }, ((Tensor<double>)bc.Outputs![0]).ToArray());
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
        Assert.Equal(new[] { 1, 1, 2, 2 }, poolOut.Dimensions.ToArray());
        Assert.Equal(new float[] { 5f, 6f, 8f, 9f }, poolOut.ToArray());

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

    [Fact]
    public void Expand_UInt_MatchesOrt()
    {
        // ORT 1.29: [[1, 2]] to [2, 2] (u32) and [[5]] to [1, 3] (u64).
        var e32 = CPU.Expand(DenseTensor<uint>.OfValues(new uint[,] { { 1u, 2u } }), DenseTensor<long>.OfValues(new long[] { 2L, 2L }), null);
        Assert.Equal(OpStatus.Success, e32.Status);
        Assert.Equal(new uint[] { 1u, 2u, 1u, 2u }, ((Tensor<uint>)e32.Outputs[0]).ToArray());
        var e64 = CPU.Expand(DenseTensor<ulong>.OfValues(new ulong[,] { { 5ul } }), DenseTensor<long>.OfValues(new long[] { 1L, 3L }), null);
        Assert.Equal(OpStatus.Success, e64.Status);
        Assert.Equal(new ulong[] { 5ul, 5ul, 5ul }, ((Tensor<ulong>)e64.Outputs[0]).ToArray());
    }

    [Fact]
    public void Expand_Sub32_MatchesOrt()
    {
        // ORT 1.29: [[v0, v1]] to [2, 2] across int8/uint8/int16/uint16.
        var sh = DenseTensor<long>.OfValues(new long[] { 2L, 2L });
        var e8 = CPU.Expand(DenseTensor<sbyte>.OfValues(new sbyte[,] { { 1, -2 } }), sh, null);
        Assert.Equal(OpStatus.Success, e8.Status);
        Assert.Equal(new sbyte[] { 1, -2, 1, -2 }, ((Tensor<sbyte>)e8.Outputs[0]).ToArray());
        var eu8 = CPU.Expand(DenseTensor<byte>.OfValues(new byte[,] { { 1, 200 } }), sh, null);
        Assert.Equal(OpStatus.Success, eu8.Status);
        Assert.Equal(new byte[] { 1, 200, 1, 200 }, ((Tensor<byte>)eu8.Outputs[0]).ToArray());
        var e16 = CPU.Expand(DenseTensor<short>.OfValues(new short[,] { { 1, -2000 } }), sh, null);
        Assert.Equal(OpStatus.Success, e16.Status);
        Assert.Equal(new short[] { 1, -2000, 1, -2000 }, ((Tensor<short>)e16.Outputs[0]).ToArray());
        var eu16 = CPU.Expand(DenseTensor<ushort>.OfValues(new ushort[,] { { 1, 60000 } }), sh, null);
        Assert.Equal(OpStatus.Success, eu16.Status);
        Assert.Equal(new ushort[] { 1, 60000, 1, 60000 }, ((Tensor<ushort>)eu16.Outputs[0]).ToArray());
    }

    [Fact]
    public void Expand_Half_MatchesOrt()
    {
        // ORT 1.29 float16: [[1, 2]] to [2, 2].
        var r = CPU.Expand(DenseTensor<Half>.OfValues(new Half[] { (Half)1f, (Half)2f }, new int[] { 1, 2 }), DenseTensor<long>.OfValues(new long[] { 2L, 2L }), null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new Half[] { (Half)1f, (Half)2f, (Half)1f, (Half)2f }, ((Tensor<Half>)r.Outputs[0]).ToArray());
    }

    [Fact]
    public void ReluInt_MatchesOrtWithVersionGate()
    {
        // ORT 1.29 runs int8/int32 Relu at opset 14 (probed) but refuses
        // both at 13; the kernels clamp negatives to zero while the node
        // boundary restores the opset-13 refusal.
        var r8 = CPU.Relu(DenseTensor<sbyte>.OfValues(new sbyte[] { -128, -1, 0, 1, 127 }), null);
        Assert.Equal(OpStatus.Success, r8.Status);
        Assert.Equal(new sbyte[] { 0, 0, 0, 1, 127 }, ((Tensor<sbyte>)r8.Outputs![0]).ToArray());
        var r32 = CPU.Relu(DenseTensor<int>.OfValues(new int[] { -5, 0, 7 }), null);
        Assert.Equal(OpStatus.Success, r32.Status);
        Assert.Equal(new int[] { 0, 0, 7 }, ((Tensor<int>)r32.Outputs![0]).ToArray());
        foreach (var opset in new int[] { 13, 14 })
        {
            var graph = new ComputationalGraph
            {
                Opset = new Dictionary<string, int> { [""] = opset },
                Metadata = new Dictionary<string, object> { ["Name"] = "test" },
            };
            graph.Inputs["x"] = DenseTensor<sbyte>.OfValues(new sbyte[] { 1 });
            var node = new Node
            {
                Name = "r", Op = OpType.Relu, Inputs = new[] { "x" }, Outputs = new[] { "y" },
                Attributes = new Dictionary<string, object>(),
            };
            var r = node.Execute(graph, ExecutionProvider.CPU, null);
            if (opset < 14)
            {
                Assert.Equal(OpStatus.Failure, r.Status);
                Assert.Contains("Int8", r.Message ?? "");
            }
            else
            {
                Assert.Equal(OpStatus.Success, r.Status);
            }
        }
    }

    [Fact]
    public void BinaryOps_RejectMixedDtypes()
    {
        // ORT refuses mixed-dtype binary inputs at load; all seven ops share
        // one identical same-dtype guard returning descriptive Failure.
        var f = DenseTensor<float>.OfValues(new float[] { 1f });
        var i = DenseTensor<int>.OfValues(new int[] { 1 });
        Assert.Equal(OpStatus.Failure, CPU.Add(f, i, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Sub(f, i, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Mul(f, i, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Div(f, i, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Pow(f, i, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Equal(f, i, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Less(f, i, null).Status);
    }

    [Fact]
    public void SoftmaxDefaultAxis_FollowsOpset()
    {
        // ORT 1.29 on arange(6) shaped [1,2,3]: opset 11 defaults to
        // axis 1, opset 13 to axis -1.
        var x = DenseTensor<float>.OfValues(new float[1, 2, 3] { { { 0f, 1f, 2f }, { 3f, 4f, 5f } } });
        var r11 = CPU.Softmax(x, null, null, null, 11);
        Assert.Equal(OpStatus.Success, r11.Status);
        var y11 = ((Tensor<float>)r11.Outputs![0]).ToArray();
        Assert.Equal(0.00427f, y11[0], 5);
        Assert.Equal(0.63369f, y11[5], 5);
        var r13 = CPU.Softmax(x, null, null, null, 13);
        Assert.Equal(OpStatus.Success, r13.Status);
        var y13 = ((Tensor<float>)r13.Outputs![0]).ToArray();
        Assert.Equal(0.09003f, y13[0], 5);
        Assert.Equal(0.66524f, y13[2], 5);
        Assert.Equal(0.09003f, y13[3], 5);
    }

    [Fact]
    public void SoftmaxAxisOutOfRange_Throws()
    {
        // ORT 1.29 refuses out-of-range axes at load (shape inference);
        // the kernel throws the same descriptive ArgumentException.
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f } });
        Assert.Throws<System.ArgumentException>(() => CPU.Softmax(x, 5, null, null, 13));
        Assert.Throws<System.ArgumentException>(() => CPU.Softmax(x, -3, null, null, 13));
    }
    [Fact]
    public void Softmax_RejectsIntegerInputs()
    {
        // ORT 1.29 refuses integer and bool Softmax at load (all of
        // int8/uint8/int16/uint16/int32/uint32/int64/uint64/bool probed
        // refused at opset 13); the provider fails descriptively instead.
        Assert.Equal(OpStatus.Failure, CPU.Softmax(DenseTensor<sbyte>.OfValues(new sbyte[] { 1 }), -1, null, null, 13).Status);
        Assert.Equal(OpStatus.Failure, CPU.Softmax(DenseTensor<byte>.OfValues(new byte[] { 1 }), -1, null, null, 13).Status);
        Assert.Equal(OpStatus.Failure, CPU.Softmax(DenseTensor<short>.OfValues(new short[] { 1 }), -1, null, null, 13).Status);
        Assert.Equal(OpStatus.Failure, CPU.Softmax(DenseTensor<ushort>.OfValues(new ushort[] { 1 }), -1, null, null, 13).Status);
        Assert.Equal(OpStatus.Failure, CPU.Softmax(DenseTensor<int>.OfValues(new int[] { 1 }), -1, null, null, 13).Status);
        Assert.Equal(OpStatus.Failure, CPU.Softmax(DenseTensor<uint>.OfValues(new uint[] { 1 }), -1, null, null, 13).Status);
        Assert.Equal(OpStatus.Failure, CPU.Softmax(DenseTensor<long>.OfValues(new long[] { 1 }), -1, null, null, 13).Status);
        Assert.Equal(OpStatus.Failure, CPU.Softmax(DenseTensor<ulong>.OfValues(new ulong[] { 1 }), -1, null, null, 13).Status);
        Assert.Equal(OpStatus.Failure, CPU.Softmax(DenseTensor<bool>.OfValues(new bool[] { true }), -1, null, null, 13).Status);
    }

    [Fact]
    public void UnaryMath_RejectUnsupportedSub32()
    {
        // Signed sub-32 Neg/Abs moved to supported (values pinned in
        // IntegerArithmeticBoundaryTests); unsigned Neg has no ORT kernel
        // and float-only kernels stay closed to every sub-32 width
        // (all four probed refused at opset 14).
        var u8 = DenseTensor<byte>.OfValues(new byte[] { 1 });
        Assert.Equal(OpStatus.Failure, CPU.Neg(u8, null).Status);
        var u16 = DenseTensor<ushort>.OfValues(new ushort[] { 1 });
        Assert.Equal(OpStatus.Failure, CPU.Neg(u16, null).Status);
        var s8 = DenseTensor<sbyte>.OfValues(new sbyte[] { 1 });
        Assert.Equal(OpStatus.Failure, CPU.Sqrt(s8, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Cos(s8, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Relu(u8, null).Status);
        var s16 = DenseTensor<short>.OfValues(new short[] { 1 });
        foreach (var x in new ITensor[] { u8, s16, u16 })
        {
            Assert.Equal(OpStatus.Failure, CPU.Sqrt(x, null).Status);
            Assert.Equal(OpStatus.Failure, CPU.Cos(x, null).Status);
            Assert.Equal(OpStatus.Failure, CPU.Sin(x, null).Status);
            Assert.Equal(OpStatus.Failure, CPU.Tanh(x, null).Status);
            Assert.Equal(OpStatus.Failure, CPU.Erf(x, null, null).Status);
            Assert.Equal(OpStatus.Failure, CPU.Gelu(x, null, null, null).Status);
        }
        Assert.Equal(OpStatus.Failure, CPU.Sin(s8, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Tanh(s8, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Erf(s8, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Gelu(s8, null, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Relu(u16, null).Status);
    }

    [Fact]
    public void BinaryOps_Sub32NeedsOpset14()
    {
        // ORT 1.29 refuses sub-32 arithmetic at opset 13 (all four widths,
        // probed) but runs it at opset 14 with wraparound; bool is refused
        // at both. The provider entry is version-blind (opset-14 semantics,
        // values pinned in IntegerArithmeticBoundaryTests); the node
        // boundary restores the opset-13 refusal. Pow stays refused: ORT
        // load-fails sub-32 Pow at every opset.
        var s8 = DenseTensor<sbyte>.OfValues(new sbyte[] { 1 });
        var b = DenseTensor<bool>.OfValues(new bool[] { true });
        Assert.Equal(OpStatus.Success, CPU.Add(s8, s8, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Pow(s8, s8, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Add(b, b, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Sub(b, b, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Mul(b, b, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Div(b, b, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Pow(b, b, null).Status);
        foreach (var opset in new int[] { 13, 14 })
        {
            var graph = new ComputationalGraph
            {
                Opset = new Dictionary<string, int> { [""] = opset },
                Metadata = new Dictionary<string, object> { ["Name"] = "test" },
            };
            graph.Inputs["x"] = s8;
            graph.Inputs["y"] = s8;
            OpResult Run(OpType op)
            {
                var node = new Node
                {
                    Name = "n", Op = op, OpTypeName = op.ToString(), Domain = "",
                    OpsetVersion = opset, IsFused = false,
                    Inputs = new[] { "x", "y" }, Outputs = new[] { "z" },
                    Attributes = new Dictionary<string, object>(),
                };
                return node.Execute(graph, ExecutionProvider.CPU, null);
            }
            var expected = opset >= 14 ? OpStatus.Success : OpStatus.Failure;
            Assert.Equal(expected, Run(OpType.Add).Status);
            Assert.Equal(expected, Run(OpType.Sub).Status);
            Assert.Equal(expected, Run(OpType.Mul).Status);
            Assert.Equal(expected, Run(OpType.Div).Status);
        }
    }

    [Fact]
    public void BinaryOps_Sub32GateHoldsThroughImport()
    {
        // Imported nodes carry no per-node version, so the gate must fire
        // off the graph opset: an opset-13 int8 Add model fails at run time
        // while the opset-14 twin computes wrapped values (both probed end
        // to end via OpDump).
        foreach (var opset in new int[] { 13, 14 })
        {
            var mp = new OnnxModel { Name = "add-i8" };
            mp.Opset[""] = opset;
            mp.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Int8, Dims = new[] { 4 } });
            mp.Inputs.Add(new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Int8, Dims = new[] { 4 } });
            mp.Outputs.Add(new OnnxValueInfo { Name = "z", ElementType = TensorElementType.Int8, Dims = new[] { 4 } });
            mp.Nodes.Add(new OnnxNode
            {
                Name = "n", OpType = "Add", Domain = "",
                Inputs = new[] { "x", "y" }, Outputs = new[] { "z" },
                Attributes = new Dictionary<string, object>(),
            });
            var graph = Model.Load(mp)!;
            var inputs = new Dictionary<string, ITensor>
            {
                { "x", DenseTensor<sbyte>.OfValues(new sbyte[] { 100, -100, 50, -128 }) },
                { "y", DenseTensor<sbyte>.OfValues(new sbyte[] { 100, -100, 3, -1 }) },
            };
            if (opset >= 14)
            {
                Assert.True(graph.Execute(inputs, true));
                Assert.Equal(new sbyte[] { -56, 56, 53, 127 }, ((Tensor<sbyte>)graph.Outputs["z"]).ToArray());
            }
            else
            {
                Assert.False(graph.Execute(inputs, true));
                Assert.Contains("Int8", graph.LastErrorMessage ?? "");
            }
        }
    }
    [Fact]
    public void UnaryMath_RejectUnsupportedDtypes()
    {
        // ORT constrains these math kernels to float types at load (Cos-int
        // and Neg-bool already pin theirs); every other unsupported dtype
        // must fail descriptively instead of reaching a kernel cast.
        // (int8/int32 Relu joined at opset 14 instead; see ReluInt test.)
        var i = DenseTensor<int>.OfValues(new int[] { 1 });
        Assert.Equal(OpStatus.Failure, CPU.Sqrt(i, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Tanh(i, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Erf(i, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Sin(i, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Gelu(i, null, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Softmax(i, null, null, null, 13).Status);
        var u = DenseTensor<uint>.OfValues(new uint[] { 1u });
        Assert.Equal(OpStatus.Failure, CPU.Neg(u, null).Status);
        var u64 = DenseTensor<ulong>.OfValues(new ulong[] { 1ul });
        Assert.Equal(OpStatus.Failure, CPU.Neg(u64, null).Status);
        var b = DenseTensor<bool>.OfValues(new bool[] { true });
        Assert.Equal(OpStatus.Failure, CPU.Abs(b, null).Status);
    }

    [Fact]
    public void UnaryMath_RejectUnsupportedWideInts()
    {
        // ORT 1.29 refuses these float-only kernels on every wide integer
        // and bool width at load (uint32/int64/uint64/bool probed refused
        // for Cos/Sin/Tanh/Erf/Sqrt/Relu/Gelu, plus int32 for Cos); the
        // provider fails descriptively instead. (Double Erf/Gelu/Sqrt run
        // on both sides and are pinned separately.)
        var u32 = DenseTensor<uint>.OfValues(new uint[] { 1u });
        var i64 = DenseTensor<long>.OfValues(new long[] { 1L });
        var u64 = DenseTensor<ulong>.OfValues(new ulong[] { 1UL });
        var b = DenseTensor<bool>.OfValues(new bool[] { true });
        foreach (var x in new ITensor[] { u32, i64, u64, b })
        {
            Assert.Equal(OpStatus.Failure, CPU.Cos(x, null).Status);
            Assert.Equal(OpStatus.Failure, CPU.Sin(x, null).Status);
            Assert.Equal(OpStatus.Failure, CPU.Tanh(x, null).Status);
            Assert.Equal(OpStatus.Failure, CPU.Erf(x, null, null).Status);
            Assert.Equal(OpStatus.Failure, CPU.Sqrt(x, null).Status);
            Assert.Equal(OpStatus.Failure, CPU.Relu(x, null).Status);
            Assert.Equal(OpStatus.Failure, CPU.Gelu(x, null, null, null).Status);
        }
        var i32 = DenseTensor<int>.OfValues(new int[] { 1 });
        Assert.Equal(OpStatus.Failure, CPU.Cos(i32, null).Status);
    }
    [Fact]
    public void Pow_RejectsSub32Widths()
    {
        // ORT 1.29 load-fails sub-32 Pow at every opset (int8/bool pinned
        // in BinaryOps_Sub32NeedsOpset14; int16/uint16 probed refused this
        // turn); the provider fails descriptively instead.
        var u8 = DenseTensor<byte>.OfValues(new byte[] { 2 });
        Assert.Equal(OpStatus.Failure, CPU.Pow(u8, u8, null).Status);
        var s16 = DenseTensor<short>.OfValues(new short[] { 2 });
        Assert.Equal(OpStatus.Failure, CPU.Pow(s16, s16, null).Status);
        var u16 = DenseTensor<ushort>.OfValues(new ushort[] { 2 });
        Assert.Equal(OpStatus.Failure, CPU.Pow(u16, u16, null).Status);
    }
    [Fact]
    public void BinaryOps_EmptyInputs_YieldEmpty()
    {
        // ORT 1.29: elementwise kernels over zero elements yield empty
        // outputs (arithmetic and comparison alike), not failures.
        var e = DenseTensor<float>.OfShape(0);
        var add = CPU.Add(e, e, null, null);
        Assert.Equal(OpStatus.Success, add.Status);
        Assert.Empty(((Tensor<float>)add.Outputs![0]).ToArray());
        var mul = CPU.Mul(e, e, null, null);
        Assert.Equal(OpStatus.Success, mul.Status);
        Assert.Empty(((Tensor<float>)mul.Outputs![0]).ToArray());
        var sub = CPU.Sub(e, e, null);
        Assert.Equal(OpStatus.Success, sub.Status);
        Assert.Empty(((Tensor<float>)sub.Outputs![0]).ToArray());
        var div = CPU.Div(e, e, null, null);
        Assert.Equal(OpStatus.Success, div.Status);
        Assert.Empty(((Tensor<float>)div.Outputs![0]).ToArray());
        var pow = CPU.Pow(e, e, null);
        Assert.Equal(OpStatus.Success, pow.Status);
        Assert.Empty(((Tensor<float>)pow.Outputs![0]).ToArray());
        var eq = CPU.Equal(e, e, null);
        Assert.Equal(OpStatus.Success, eq.Status);
        Assert.Empty(((Tensor<bool>)eq.Outputs![0]).ToArray());
        var less = CPU.Less(e, e, null);
        Assert.Equal(OpStatus.Success, less.Status);
        Assert.Empty(((Tensor<bool>)less.Outputs![0]).ToArray());
    }

    [Fact]
    public void UnaryMath_EmptyInputs_YieldEmpty()
    {
        // ORT 1.29: unary math kernels over zero elements yield empty
        // outputs, mirroring the binary empty batch above.
        var e = DenseTensor<float>.OfShape(0);
        var sqrt = CPU.Sqrt(e, null);
        Assert.Equal(OpStatus.Success, sqrt.Status);
        Assert.Empty(((Tensor<float>)sqrt.Outputs![0]).ToArray());
        var relu = CPU.Relu(e, null);
        Assert.Equal(OpStatus.Success, relu.Status);
        Assert.Empty(((Tensor<float>)relu.Outputs![0]).ToArray());
        var tanh = CPU.Tanh(e, null);
        Assert.Equal(OpStatus.Success, tanh.Status);
        Assert.Empty(((Tensor<float>)tanh.Outputs![0]).ToArray());
        var abs = CPU.Abs(e, null);
        Assert.Equal(OpStatus.Success, abs.Status);
        Assert.Empty(((Tensor<float>)abs.Outputs![0]).ToArray());
        var neg = CPU.Neg(e, null);
        Assert.Equal(OpStatus.Success, neg.Status);
        Assert.Empty(((Tensor<float>)neg.Outputs![0]).ToArray());
        var erf = CPU.Erf(e, null, null);
        Assert.Equal(OpStatus.Success, erf.Status);
        Assert.Empty(((Tensor<float>)erf.Outputs![0]).ToArray());
        var sin = CPU.Sin(e, null);
        Assert.Equal(OpStatus.Success, sin.Status);
        Assert.Empty(((Tensor<float>)sin.Outputs![0]).ToArray());
        var cos = CPU.Cos(e, null);
        Assert.Equal(OpStatus.Success, cos.Status);
        Assert.Empty(((Tensor<float>)cos.Outputs![0]).ToArray());
        var gelu = CPU.Gelu(e, null, null, null);
        Assert.Equal(OpStatus.Success, gelu.Status);
        Assert.Empty(((Tensor<float>)gelu.Outputs![0]).ToArray());
    }

    [Fact]
    public void HalfArithmetic_RefusedCleanly()
    {
        // Documented scope boundary, not parity: ORT 1.29 runs float16
        // Add/Sub/Mul/Div/Neg/Abs (all probed) and refuses bfloat16 Add
        // (probed); half kernels do not exist here, so every form fails
        // descriptively instead of reaching a kernel cast.
        var a = DenseTensor<Half>.OfValues(new Half[] { (Half)1f, (Half)2f });
        var b = DenseTensor<Half>.OfValues(new Half[] { (Half)3f, (Half)4f });
        Assert.Equal(OpStatus.Failure, CPU.Add(a, b, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Sub(a, b, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Mul(a, b, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Div(a, b, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Neg(a, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Abs(a, null).Status);
        var bf = Tensor<BFloat16>.Ones(2);
        Assert.Equal(OpStatus.Failure, CPU.Add(bf, bf, null, null).Status);
    }

    [Fact]
    public void BFloat16Arithmetic_RefusedCleanly()
    {
        // Same scope boundary as float16 arithmetic, and ORT 1.29
        // likewise has no CPU kernel for any of them (Sub/Mul/Div/
        // Neg/Abs probed NOT_IMPLEMENTED here; Add probed refused
        // with the float16 set): every form fails descriptively.
        var a = DenseTensor<BFloat16>.OfValues(new BFloat16[] { (BFloat16)1f, (BFloat16)2f });
        var b = DenseTensor<BFloat16>.OfValues(new BFloat16[] { (BFloat16)3f, (BFloat16)4f });
        Assert.Equal(OpStatus.Failure, CPU.Add(a, b, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Sub(a, b, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Mul(a, b, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Div(a, b, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Neg(a, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Abs(a, null).Status);
    }

    [Fact]
    public void Float16Compute_RefusedCleanly()
    {
        // Parity gap, not parity: ORT 1.29 ACCEPTS float16 ReduceSum,
        // Softmax, LayerNorm, MatMul, Conv and Gemm (all probed), but no
        // half compute kernels exist here, so every form fails
        // descriptively. Shapes below are valid, isolating the dtype gate.
        var h = DenseTensor<Half>.OfValues(new Half[] { (Half)1f, (Half)2f, (Half)3f, (Half)4f }, new int[] { 2, 2 });
        var axes = DenseTensor<long>.OfValues(new long[] { 1 });
        Assert.Equal(OpStatus.Failure, CPU.ReduceSum(h, axes, 0, 0, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Softmax(h, -1, null, null, 13).Status);
        var sc = DenseTensor<Half>.OfValues(new Half[] { (Half)1f, (Half)1f });
        Assert.Equal(OpStatus.Failure, CPU.LayerNormalization(h, sc, null, -1, 1e-5f, null, 1, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.MatMul(h, h, null, null).Status);
        var x = DenseTensor<Half>.OfValues(new Half[] { (Half)1f, (Half)1f, (Half)1f, (Half)1f, (Half)1f, (Half)1f, (Half)1f, (Half)1f, (Half)1f, (Half)1f, (Half)1f, (Half)1f, (Half)1f, (Half)1f, (Half)1f, (Half)1f }, new int[] { 1, 1, 4, 4 });
        var w = DenseTensor<Half>.OfValues(new Half[] { (Half)1f, (Half)1f, (Half)1f, (Half)1f }, new int[] { 1, 1, 2, 2 });
        Assert.Equal(OpStatus.Failure, CPU.Conv(x, w, null, auto_pad: null, dilations: null, group: null, kernel_shape: new int[] { 2, 2 }, pads: null, strides: null, options: null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Gemm(h, h, null, 1f, 1f, null, 0, 0).Status);
    }

    [Fact]
    public void BFloat16Compute_RefusedCleanly()
    {
        // ORT 1.29 has no CPU kernel for any of these on bfloat16
        // (all probed NOT_IMPLEMENTED); shapes below are valid,
        // isolating the dtype gate.
        var b = DenseTensor<BFloat16>.OfValues(new BFloat16[] { (BFloat16)1f, (BFloat16)2f, (BFloat16)3f, (BFloat16)4f }, new int[] { 2, 2 });
        var axes = DenseTensor<long>.OfValues(new long[] { 1 });
        Assert.Equal(OpStatus.Failure, CPU.ReduceSum(b, axes, 0, 0, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Softmax(b, -1, null, null, 13).Status);
        var sc = DenseTensor<BFloat16>.OfValues(new BFloat16[] { (BFloat16)1f, (BFloat16)1f });
        Assert.Equal(OpStatus.Failure, CPU.LayerNormalization(b, sc, null, -1, 1e-5f, null, 1, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.MatMul(b, b, null, null).Status);
        var x = DenseTensor<BFloat16>.OfValues(new BFloat16[] { (BFloat16)1f, (BFloat16)1f, (BFloat16)1f, (BFloat16)1f, (BFloat16)1f, (BFloat16)1f, (BFloat16)1f, (BFloat16)1f, (BFloat16)1f, (BFloat16)1f, (BFloat16)1f, (BFloat16)1f, (BFloat16)1f, (BFloat16)1f, (BFloat16)1f, (BFloat16)1f }, new int[] { 1, 1, 4, 4 });
        var w = DenseTensor<BFloat16>.OfValues(new BFloat16[] { (BFloat16)1f, (BFloat16)1f, (BFloat16)1f, (BFloat16)1f }, new int[] { 1, 1, 2, 2 });
        Assert.Equal(OpStatus.Failure, CPU.Conv(x, w, null, auto_pad: null, dilations: null, group: null, kernel_shape: new int[] { 2, 2 }, pads: null, strides: null, options: null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Gemm(b, b, null, 1f, 1f, null, 0, 0).Status);
    }

    [Fact]
    public void ComplexCompute_RefusedCleanly()
    {
        // ORT rejects complex64 compute at schema level (ReduceSum,
        // Softmax, MatMul, Gemm all probed refused); shapes below are
        // valid, isolating the dtype gate.
        var c = DenseTensor<System.Numerics.Complex>.OfValues(new System.Numerics.Complex[] { new System.Numerics.Complex(1, 1), new System.Numerics.Complex(2, 2), new System.Numerics.Complex(3, 3), new System.Numerics.Complex(4, 4) }, new int[] { 2, 2 });
        var axes = DenseTensor<long>.OfValues(new long[] { 1 });
        Assert.Equal(OpStatus.Failure, CPU.ReduceSum(c, axes, 0, 0, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Softmax(c, -1, null, null, 13).Status);
        var sc = DenseTensor<System.Numerics.Complex>.OfValues(new System.Numerics.Complex[] { new System.Numerics.Complex(1, 0), new System.Numerics.Complex(1, 0) });
        Assert.Equal(OpStatus.Failure, CPU.LayerNormalization(c, sc, null, -1, 1e-5f, null, 1, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.MatMul(c, c, null, null).Status);
        var x = DenseTensor<System.Numerics.Complex>.OfValues(new System.Numerics.Complex[16], new int[] { 1, 1, 4, 4 });
        var w = DenseTensor<System.Numerics.Complex>.OfValues(new System.Numerics.Complex[4], new int[] { 1, 1, 2, 2 });
        Assert.Equal(OpStatus.Failure, CPU.Conv(x, w, null, auto_pad: null, dilations: null, group: null, kernel_shape: new int[] { 2, 2 }, pads: null, strides: null, options: null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Gemm(c, c, null, 1f, 1f, null, 0, 0).Status);
    }

    [Fact]
    public void Float16Unary_RefusedCleanly()
    {
        // Parity gap, not parity: ORT 1.29 ACCEPTS float16 Sqrt, Relu,
        // Tanh, Erf and Pow (all probed) but refuses float16 Gelu at
        // schema level; no half unary kernels exist here, so every form
        // fails descriptively. Owner scope decision, like the compute row.
        var h = DenseTensor<Half>.OfValues(new Half[] { (Half)0.5f, (Half)1.5f });
        Assert.Equal(OpStatus.Failure, CPU.Pow(h, h, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Sqrt(h, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Relu(h, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Tanh(h, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Erf(h, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Gelu(h, null, null, null).Status);
    }

    [Fact]
    public void BFloat16Unary_RefusedCleanly()
    {
        // ORT 1.29 refuses every one of these on bfloat16 (Sqrt, Relu,
        // Tanh, Erf NOT_IMPLEMENTED; Gelu and Pow schema-refused; all
        // probed), matching the provider refusal below.
        var b = DenseTensor<BFloat16>.OfValues(new BFloat16[] { (BFloat16)0.5f, (BFloat16)1.5f });
        Assert.Equal(OpStatus.Failure, CPU.Pow(b, b, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Sqrt(b, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Relu(b, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Tanh(b, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Erf(b, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Gelu(b, null, null, null).Status);
    }

    [Fact]
    public void ComplexUnary_RefusedCleanly()
    {
        // ORT rejects complex64 unary at schema level (Pow, Sqrt, Relu,
        // Tanh, Erf, Gelu all probed refused); the provider has no arms.
        var c = DenseTensor<System.Numerics.Complex>.OfValues(new System.Numerics.Complex[] { new System.Numerics.Complex(0.5, 0), new System.Numerics.Complex(1.5, 1) });
        Assert.Equal(OpStatus.Failure, CPU.Pow(c, c, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Sqrt(c, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Relu(c, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Tanh(c, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Erf(c, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Gelu(c, null, null, null).Status);
    }

    [Fact]
    public void UnsupportedDtypePairs_FailCleanlyOnBothSides()
    {
        // Every pair below was probed refused on ORT 1.29 (load or run) and
        // fails descriptively here: Pow is float/double/int32/int64 only,
        // sequences exclude sub-32 payloads,
        // ReduceSum has no sub-32 arm, and Softmax/LayerNorm are float-types
        // only (Gemm-int already pins in IntDtypes_RejectedCleanly).
        Assert.Equal(OpStatus.Failure, CPU.Pow(
            DenseTensor<uint>.OfValues(new uint[] { 2u, 3u }),
            DenseTensor<uint>.OfValues(new uint[] { 2u, 2u }), null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Pow(
            DenseTensor<ulong>.OfValues(new ulong[] { 2ul }),
            DenseTensor<ulong>.OfValues(new ulong[] { 2ul }), null).Status);
        Assert.Equal(OpStatus.Failure, CPU.ReduceSum(
            DenseTensor<sbyte>.OfValues(new sbyte[] { 1, 2 }), null, null, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Softmax(
            DenseTensor<int>.OfValues(new int[,] { { 1, 2 }, { 3, 4 } }), null, null, null, 14).Status);
        Assert.Equal(OpStatus.Failure, CPU.LayerNormalization(
            DenseTensor<int>.OfValues(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } }),
            DenseTensor<int>.OfValues(new int[] { 1, 1, 1 }),
            DenseTensor<int>.OfValues(new int[] { 0, 0, 0 }), -1, null, null, 1, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.SplitToSequence(
            DenseTensor<sbyte>.OfValues(new sbyte[] { 1, 2, 3, 4 }),
            DenseTensor<long>.OfValues(new long[] { 2, 2 }), 0, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.SplitToSequence(
            DenseTensor<byte>.OfValues(new byte[] { 1, 2, 3, 4 }),
            DenseTensor<long>.OfValues(new long[] { 2, 2 }), 0, null, null).Status);
    }

    [Fact]
    public void WhereUnsupportedDtypes_RefusedCleanly()
    {
        // ORT 1.29 refuses every one of these (int8/int16/uint16
        // NOT_IMPLEMENTED; bfloat16 schema; complex64 not registered;
        // all probed); the Where switch stops at uint64/float/double.
        var cond = DenseTensor<bool>.OfValues(new bool[,] { { true, false }, { false, true } });
        Assert.Equal(OpStatus.Failure, CPU.Where(cond, DenseTensor<sbyte>.OfValues(new sbyte[,] { { 1, 2 }, { 3, 4 } }), DenseTensor<sbyte>.OfValues(new sbyte[,] { { 5, 6 }, { 7, 8 } }), null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Where(cond, DenseTensor<short>.OfValues(new short[,] { { 1, 2 }, { 3, 4 } }), DenseTensor<short>.OfValues(new short[,] { { 5, 6 }, { 7, 8 } }), null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Where(cond, DenseTensor<ushort>.OfValues(new ushort[,] { { 1, 2 }, { 3, 4 } }), DenseTensor<ushort>.OfValues(new ushort[,] { { 5, 6 }, { 7, 8 } }), null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Where(cond, DenseTensor<BFloat16>.OfValues(new BFloat16[] { (BFloat16)1f, (BFloat16)2f, (BFloat16)3f, (BFloat16)4f }, new int[] { 2, 2 }), DenseTensor<BFloat16>.OfValues(new BFloat16[] { (BFloat16)5f, (BFloat16)6f, (BFloat16)7f, (BFloat16)8f }, new int[] { 2, 2 }), null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Where(cond, DenseTensor<System.Numerics.Complex>.OfValues(new System.Numerics.Complex[,] { { new System.Numerics.Complex(1, 0), new System.Numerics.Complex(2, 0) }, { new System.Numerics.Complex(3, 0), new System.Numerics.Complex(4, 0) } }), DenseTensor<System.Numerics.Complex>.OfValues(new System.Numerics.Complex[,] { { new System.Numerics.Complex(5, 0), new System.Numerics.Complex(6, 0) }, { new System.Numerics.Complex(7, 0), new System.Numerics.Complex(8, 0) } }), null).Status);
    }

    [Fact]
    public void ReduceSub32_RefusedCleanly()
    {
        // ORT refuses sub-32 and wide-uint reductions (schema or
        // NOT_IMPLEMENTED, all probed); only int8 ReduceSum was pinned
        // before, so the rest of the row joins it here.
        var ax = DenseTensor<long>.OfValues(new long[] { 0 });
        Assert.Equal(OpStatus.Failure, CPU.ReduceSum(DenseTensor<byte>.OfValues(new byte[] { 1, 2 }), ax, 0, 0, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.ReduceSum(DenseTensor<short>.OfValues(new short[] { 1, 2 }), ax, 0, 0, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.ReduceSum(DenseTensor<ushort>.OfValues(new ushort[] { 1, 2 }), ax, 0, 0, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.ReduceSum(DenseTensor<uint>.OfValues(new uint[] { 1, 2 }), ax, 0, 0, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.ReduceSum(DenseTensor<ulong>.OfValues(new ulong[] { 1, 2 }), ax, 0, 0, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.ReduceMean(DenseTensor<sbyte>.OfValues(new sbyte[] { 1, 2 }), ax, 0, 0, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.ReduceMean(DenseTensor<byte>.OfValues(new byte[] { 1, 2 }), ax, 0, 0, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.ReduceMean(DenseTensor<short>.OfValues(new short[] { 1, 2 }), ax, 0, 0, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.ReduceMean(DenseTensor<ushort>.OfValues(new ushort[] { 1, 2 }), ax, 0, 0, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.ReduceMean(DenseTensor<uint>.OfValues(new uint[] { 1, 2 }), ax, 0, 0, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.ReduceMean(DenseTensor<ulong>.OfValues(new ulong[] { 1, 2 }), ax, 0, 0, null).Status);
    }

    [Fact]
    public void NegUnsigned_RefusedCleanly()
    {
        // ORT rejects unsigned Neg at schema level (all four probed);
        // the Neg switch stops at signed sub-32 widths.
        Assert.Equal(OpStatus.Failure, CPU.Neg(DenseTensor<byte>.OfValues(new byte[] { 1, 2 }), null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Neg(DenseTensor<ushort>.OfValues(new ushort[] { 1, 2 }), null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Neg(DenseTensor<uint>.OfValues(new uint[] { 1, 2 }), null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Neg(DenseTensor<ulong>.OfValues(new ulong[] { 1, 2 }), null).Status);
    }

    [Fact]
    public void UintMatMul_RefusedCleanly()
    {
        // Split verdict: ORT 1.29 ACCEPTS uint32/uint64 MatMul (probed)
        // but refuses uint32 Gemm (NOT_IMPLEMENTED, probed). The MatMul
        // gap needs uint GEMM kernels (owner scope, cf. the int32
        // imatmul path); shapes below are valid, isolating dtype gates.
        // Overflow semantics for that future kernel, probed: plain
        // wraparound ([Max]@[2] gives Max-1 for both widths), NOT the
        // saturation integer reductions use.
        var a32 = DenseTensor<uint>.OfValues(new uint[,] { { 1u, 2u }, { 3u, 4u } });
        var b32 = DenseTensor<uint>.OfValues(new uint[,] { { 1u, 0u }, { 0u, 1u } });
        Assert.Equal(OpStatus.Failure, CPU.MatMul(a32, b32, null, null).Status);
        var a64 = DenseTensor<ulong>.OfValues(new ulong[,] { { 1ul, 2ul }, { 3ul, 4ul } });
        var b64 = DenseTensor<ulong>.OfValues(new ulong[,] { { 1ul, 0ul }, { 0ul, 1ul } });
        Assert.Equal(OpStatus.Failure, CPU.MatMul(a64, b64, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Gemm(a32, b32, null, 1f, 1f, null, 0, 0).Status);
    }

    [Fact]
    public void ReluUint_RefusedCleanly()
    {
        // ORT refuses int16 and all unsigned Relu (NOT_IMPLEMENTED or
        // schema, all probed); int8/int32 Relu stay supported elsewhere.
        Assert.Equal(OpStatus.Failure, CPU.Relu(DenseTensor<short>.OfValues(new short[] { -1, 2 }), null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Relu(DenseTensor<byte>.OfValues(new byte[] { 1, 2 }), null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Relu(DenseTensor<ushort>.OfValues(new ushort[] { 1, 2 }), null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Relu(DenseTensor<uint>.OfValues(new uint[] { 1, 2 }), null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Relu(DenseTensor<ulong>.OfValues(new ulong[] { 1, 2 }), null).Status);
    }

    [Fact]
    public void IntScalarMath_RefusedCleanly()
    {
        // ORT schema-refuses every form below (all probed); the float
        // kernels have no integer arms.
        Assert.Equal(OpStatus.Failure, CPU.Sqrt(DenseTensor<int>.OfValues(new int[] { 1, 4 }), null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Sqrt(DenseTensor<sbyte>.OfValues(new sbyte[] { 1, 4 }), null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Pow(DenseTensor<sbyte>.OfValues(new sbyte[] { 2, 3 }), DenseTensor<sbyte>.OfValues(new sbyte[] { 2, 2 }), null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Pow(DenseTensor<short>.OfValues(new short[] { 2, 3 }), DenseTensor<short>.OfValues(new short[] { 2, 2 }), null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Cos(DenseTensor<int>.OfValues(new int[] { 0, 1 }), null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Tanh(DenseTensor<int>.OfValues(new int[] { 0, 1 }), null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Erf(DenseTensor<int>.OfValues(new int[] { 0, 1 }), null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.LayerNormalization(DenseTensor<sbyte>.OfValues(new sbyte[,] { { 1, 2 }, { 3, 4 } }), DenseTensor<sbyte>.OfValues(new sbyte[] { 1, 1 }), null, -1, 1e-5f, null, 1, null, null).Status);
    }

    [Fact]
    public void Float16EqualLess_RefusedCleanly()
    {
        // Parity gap, not parity: ORT 1.29 ACCEPTS float16 Equal/Less
        // (both probed), but no half comparison kernels exist here.
        var x = DenseTensor<Half>.OfValues(new Half[] { (Half)1f, (Half)2f });
        var y = DenseTensor<Half>.OfValues(new Half[] { (Half)1f, (Half)3f });
        Assert.Equal(OpStatus.Failure, CPU.Equal(x, y, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Less(x, y, null).Status);
    }

    [Fact]
    public void BFloat16EqualLess_RefusedCleanly()
    {
        // ORT 1.29 has no CPU kernel for bfloat16 Equal/Less
        // (NOT_IMPLEMENTED, both probed); refused here too.
        var x = DenseTensor<BFloat16>.OfValues(new BFloat16[] { (BFloat16)1f, (BFloat16)2f });
        var y = DenseTensor<BFloat16>.OfValues(new BFloat16[] { (BFloat16)1f, (BFloat16)3f });
        Assert.Equal(OpStatus.Failure, CPU.Equal(x, y, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Less(x, y, null).Status);
    }

    [Fact]
    public void ComplexEqualLess_RefusedCleanly()
    {
        // ORT rejects complex64 Equal/Less at schema level (both
        // probed); the provider has no arms.
        var x = DenseTensor<System.Numerics.Complex>.OfValues(new System.Numerics.Complex[] { new System.Numerics.Complex(1, 0), new System.Numerics.Complex(2, 0) });
        var y = DenseTensor<System.Numerics.Complex>.OfValues(new System.Numerics.Complex[] { new System.Numerics.Complex(1, 0), new System.Numerics.Complex(3, 0) });
        Assert.Equal(OpStatus.Failure, CPU.Equal(x, y, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Less(x, y, null).Status);
    }

    [Fact]
    public void RangeUnsupportedDtypes_RefusedCleanly()
    {
        // ORT schema-refuses Range on every dtype below (all probed);
        // arms exist only for float/double/int16/int32/int64.
        Assert.Equal(OpStatus.Failure, CPU.Range(DenseTensor<sbyte>.OfValues(new sbyte[] { 0 }), DenseTensor<sbyte>.OfValues(new sbyte[] { 3 }), DenseTensor<sbyte>.OfValues(new sbyte[] { 1 }), null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Range(DenseTensor<byte>.OfValues(new byte[] { 0 }), DenseTensor<byte>.OfValues(new byte[] { 3 }), DenseTensor<byte>.OfValues(new byte[] { 1 }), null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Range(DenseTensor<ushort>.OfValues(new ushort[] { 0 }), DenseTensor<ushort>.OfValues(new ushort[] { 3 }), DenseTensor<ushort>.OfValues(new ushort[] { 1 }), null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Range(DenseTensor<uint>.OfValues(new uint[] { 0 }), DenseTensor<uint>.OfValues(new uint[] { 3 }), DenseTensor<uint>.OfValues(new uint[] { 1 }), null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Range(DenseTensor<ulong>.OfValues(new ulong[] { 0 }), DenseTensor<ulong>.OfValues(new ulong[] { 3 }), DenseTensor<ulong>.OfValues(new ulong[] { 1 }), null).Status);
    }

    [Fact]
    public void MatMulNarrowDtypes_RefusedCleanly()
    {
        // ORT schema-refuses int8/uint8 MatMul (both probed); only
        // float/double/int32 kernels exist (uint32/64 gap documented
        // separately).
        var a8 = DenseTensor<sbyte>.OfValues(new sbyte[,] { { 1, 2 }, { 3, 4 } });
        Assert.Equal(OpStatus.Failure, CPU.MatMul(a8, a8, null, null).Status);
        var au8 = DenseTensor<byte>.OfValues(new byte[,] { { 1, 2 }, { 3, 4 } });
        Assert.Equal(OpStatus.Failure, CPU.MatMul(au8, au8, null, null).Status);
    }

    [Fact]
    public void UintScalarMath_RefusedCleanly()
    {
        // ORT schema-refuses every form below (all probed); the float
        // kernels have no unsigned arms.
        Assert.Equal(OpStatus.Failure, CPU.Sqrt(DenseTensor<uint>.OfValues(new uint[] { 1, 4 }), null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Sqrt(DenseTensor<ulong>.OfValues(new ulong[] { 1, 4 }), null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Cos(DenseTensor<byte>.OfValues(new byte[] { 0, 1 }), null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Erf(DenseTensor<byte>.OfValues(new byte[] { 0, 1 }), null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.LayerNormalization(DenseTensor<byte>.OfValues(new byte[,] { { 1, 2 }, { 3, 4 } }), DenseTensor<byte>.OfValues(new byte[] { 1, 1 }), null, -1, 1e-5f, null, 1, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.GlobalAveragePool(DenseTensor<byte>.OfShape(1, 1, 2, 2), null).Status);
    }
}
