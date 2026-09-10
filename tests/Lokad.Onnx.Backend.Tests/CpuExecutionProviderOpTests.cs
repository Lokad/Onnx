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
    public void UnaryMath_RejectSub32()
    {
        // Documented gap, not agreement: ORT accepts int8 Abs (probed
        // [1]), but sub-32 kernels are out of scope, so these fail
        // descriptively instead of reaching a kernel cast.
        var s8 = DenseTensor<sbyte>.OfValues(new sbyte[] { 1 });
        Assert.Equal(OpStatus.Failure, CPU.Abs(s8, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Neg(s8, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Sqrt(s8, null).Status);
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
        var i = DenseTensor<int>.OfValues(new int[] { 1 });
        Assert.Equal(OpStatus.Failure, CPU.Sqrt(i, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Relu(i, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Tanh(i, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Erf(i, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Sin(i, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Gelu(i, null, null, null).Status);
        Assert.Equal(OpStatus.Failure, CPU.Softmax(i, null, null, null, 13).Status);
        var u = DenseTensor<uint>.OfValues(new uint[] { 1u });
        Assert.Equal(OpStatus.Failure, CPU.Neg(u, null).Status);
        var b = DenseTensor<bool>.OfValues(new bool[] { true });
        Assert.Equal(OpStatus.Failure, CPU.Abs(b, null).Status);
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
}




