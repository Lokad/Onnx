using System.Collections.Generic;

namespace Lokad.Onnx.Backend.Tests;

public class ReductionPlanTests
{
    static DenseTensor<float> Data() => DenseTensor<float>.OfValues(new float[,] { { 1f, 2f } });

    [Fact]
    public void FloatAxes_RejectedCleanly()
    {
        // ORT 1.29 refuses float axes at load; reductions must fail
        // descriptively instead of falling through to InvalidCast.
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f } });
        var a = DenseTensor<float>.OfValues(new float[] { 1f });
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.ReduceSum(x, a, 0, 0, null).Status);
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.ReduceMean(x, a, 0, 0, null).Status);
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.ReduceMax(x, a, 0, 0, null).Status);
    }

    [Fact]
    public void OutOfRangeAxis_Throws()
    {
        // ORT 1.29 fails the run (axis 5 outside rank 2); the provider
        // surfaces the same refusal as a descriptive ArgumentException,
        // matching the Unsqueeze out-of-range precedent. Node execution
        // converts it to OpStatus.Failure, as for LayerNormalization.
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        var axes = DenseTensor<long>.OfValues(new long[] { 5L });
        Assert.Throws<System.ArgumentException>(() => CPUExecutionProvider.ReduceMean(x, axes, 0, 0, null));
        var graph = new ComputationalGraph
        {
            Opset = new Dictionary<string, int> { [""] = 18 },
            Metadata = new Dictionary<string, object> { ["Name"] = "test" },
        };
        graph.Inputs["x"] = x;
        graph.Inputs["axes"] = axes;
        var r = new Node
        {
            Name = "r", Op = OpType.ReduceMean, Inputs = new[] { "x", "axes" }, Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object>(),
        }.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }
    [Fact]
    public void IntSum_OverflowsSaturate()
    {
        // ORT 1.29 saturates integer sums (unlike elementwise Add/Sub/Mul,
        // which wrap): [max,1] -> max, [min,-1] -> min.
        var hi = Tensor<int>.ReduceSum(DenseTensor<int>.OfValues(new int[] { 2147483647, 1 }), null, false, false);
        Assert.Equal(new int[] { 2147483647 }, hi.ToArray());
        var lo = Tensor<int>.ReduceSum(DenseTensor<int>.OfValues(new int[] { -2147483648, -1 }), null, false, false);
        Assert.Equal(new int[] { -2147483648 }, lo.ToArray());
        // Discriminating case: exact total clamped (-2), not running
        // saturation (which would give min). ORT 1.29 agrees.
        var mixed = Tensor<int>.ReduceSum(DenseTensor<int>.OfValues(new int[] { 2147483647, 2147483647, -2147483648, -2147483648 }), null, false, false);
        Assert.Equal(new int[] { -2 }, mixed.ToArray());
    }

    [Fact]
    public void IntMean_OverflowsSaturate()
    {
        // ORT 1.29: [max,max] -> max, [min,min] -> min.
        var hi = Tensor<int>.ReduceMean(DenseTensor<int>.OfValues(new int[] { 2147483647, 2147483647 }), null, false, false);
        Assert.Equal(new int[] { 2147483647 }, hi.ToArray());
        var lo = Tensor<int>.ReduceMean(DenseTensor<int>.OfValues(new int[] { -2147483648, -2147483648 }), null, false, false);
        Assert.Equal(new int[] { -2147483648 }, lo.ToArray());
    }

    [Fact]
    public void EmptyAxes_Noop_ReturnsInputUnchanged()
    {
        var empty = new int[0].ToTensor<int>();
        foreach (bool? kd in new bool?[] { true, false, null })
        {
            var r = Tensor<float>.ReduceSum(Data(), empty, kd, true);
            Assert.Equal(new int[] { 1, 2 }, r.Dimensions.ToArray());
            Assert.Equal(new float[] { 1f, 2f }, r.ToArray());
        }
        var ri = Tensor<int>.ReduceSum(DenseTensor<int>.OfValues(new int[,] { { 1, 2 } }), empty, false, true);
        Assert.Equal(new int[] { 1, 2 }, ri.Dimensions.ToArray());
        Assert.Equal(new int[] { 1, 2 }, ri.ToArray());
    }

    [Fact]
    public void AbsentAxes_Noop_ReturnsInputUnchanged()
    {
        var r = Tensor<float>.ReduceSum(Data(), null, false, true);
        Assert.Equal(new int[] { 1, 2 }, r.Dimensions.ToArray());
        Assert.Equal(new float[] { 1f, 2f }, r.ToArray());
        var m = Tensor<float>.ReduceMean(Data(), null, true, true, TensorExecutionOptions.Scalar);
        Assert.Equal(new int[] { 1, 2 }, m.Dimensions.ToArray());
    }

    [Fact]
    public void AbsentOrEmptyAxes_WithoutNoop_ReducesAllAxes()
    {
        var kd = Tensor<float>.ReduceSum(Data(), null, true, false);
        Assert.Equal(new int[] { 1, 1 }, kd.Dimensions.ToArray());
        Assert.Equal(new float[] { 3f }, kd.ToArray());
        var flat = Tensor<float>.ReduceSum(Data(), new int[0].ToTensor<int>(), false, false);
        Assert.Equal(new int[0], flat.Dimensions.ToArray());
        Assert.Equal(new float[] { 3f }, flat.ToArray());
    }

    [Fact]
    public void AbsentAxes_WithoutNoopKeepdims_ReducesToScalar()
    {
        // The one absent-axes combination without a pin: null axes with
        // keepdims=false reduces everything to a rank-0 scalar (1+2=3).
        // Exact small-integer arithmetic needs no ORT reference.
        var s = CPUExecutionProvider.ReduceSum(Data(), null, 0, 0, null);
        Assert.Equal(OpStatus.Success, s.Status);
        var ys = (Tensor<float>)s.Outputs[0];
        Assert.Equal(new int[0], ys.Dimensions.ToArray());
        Assert.Equal(new float[] { 3f }, ys.ToArray());
        var m = CPUExecutionProvider.ReduceMean(Data(), null, 0, 0, null);
        Assert.Equal(OpStatus.Success, m.Status);
        Assert.Equal(new float[] { 1.5f }, ((Tensor<float>)m.Outputs[0]).ToArray());
    }

    [Fact]
    public void NegativeAxes_EquivalentToPositive()
    {
        var pos = Tensor<float>.ReduceSum(Data(), new int[] { 1 }.ToTensor<int>(), false, false);
        var neg = Tensor<float>.ReduceSum(Data(), new int[] { -1 }.ToTensor<int>(), false, false);
        Assert.Equal(pos.Dimensions.ToArray(), neg.Dimensions.ToArray());
        Assert.Equal(pos.ToArray(), neg.ToArray());
        var first = Tensor<float>.ReduceSum(Data(), new int[] { -2 }.ToTensor<int>(), false, false);
        Assert.Equal(new float[] { 1f, 2f }, first.ToArray());
    }

    [Fact]
    public void DuplicateAxes_DeduplicatedSilently()
    {
        var r = Tensor<float>.ReduceSum(Data(), new int[] { 1, -1 }.ToTensor<int>(), false, false);
        Assert.Equal(new int[] { 1 }, r.Dimensions.ToArray());
        Assert.Equal(new float[] { 3f }, r.ToArray());
        var k = Tensor<float>.ReduceSum(Data(), new int[] { 0, 0 }.ToTensor<int>(), true, false);
        Assert.Equal(new int[] { 1, 2 }, k.Dimensions.ToArray());
        Assert.Equal(new float[] { 1f, 2f }, k.ToArray());
    }

    [Fact]
    public void OutOfRangeAxes_Throw()
    {
        Assert.Throws<System.ArgumentException>(() => Tensor<float>.ReduceSum(Data(), new int[] { 5 }.ToTensor<int>()));
        Assert.Throws<System.ArgumentException>(() => Tensor<float>.ReduceSum(Data(), new int[] { -3 }.ToTensor<int>()));
        Assert.Throws<System.ArgumentException>(() => Tensor<double>.ReduceMean(DenseTensor<double>.OfValues(new double[,] { { 1.0 } }), new int[] { 2 }.ToTensor<int>()));
    }

    [Fact]
    public void Scalar_ReducesToScalar()
    {
        var s = DenseTensor<float>.OfShape();
        s.SetValue(0, 5f);
        var r = Tensor<float>.ReduceSum(s, null, false, false);
        Assert.Equal(new int[0], r.Dimensions.ToArray());
        Assert.Equal(new float[] { 5f }, r.ToArray());
    }

    [Fact]
    public void ReduceMean_NoopAndEmpty()
    {
        var noop = Tensor<float>.ReduceMean(Data(), new int[0].ToTensor<int>(), true, true, TensorExecutionOptions.Scalar);
        Assert.Equal(new int[] { 1, 2 }, noop.Dimensions.ToArray());
        Assert.Equal(new float[] { 1f, 2f }, noop.ToArray());
        var all = Tensor<float>.ReduceMean(Data(), new int[0].ToTensor<int>(), false, false, TensorExecutionOptions.Scalar);
        Assert.Equal(new int[0], all.Dimensions.ToArray());
        Assert.Equal(new float[] { 1.5f }, all.ToArray());
        var ax = Tensor<float>.ReduceMean(Data(), new int[] { 0 }.ToTensor<int>(), false, false, TensorExecutionOptions.Scalar);
        Assert.Equal(new float[] { 1f, 2f }, ax.ToArray());
    }

    [Fact]
    public void EmptyExtent_Mean_YieldsZero()
    {
        // ORT 1.29 yields zero for every dtype; without the kernel guard
        // float and double produce NaN and int throws DivideByZero.
        var axes = new int[] { 1 }.ToTensor<int>();
        var f = Tensor<float>.ReduceMean(DenseTensor<float>.OfShape(2, 0), axes, false, false);
        Assert.Equal(new int[] { 2 }, f.Dimensions.ToArray());
        Assert.Equal(new float[] { 0f, 0f }, f.ToArray());
        var fk = Tensor<float>.ReduceMean(DenseTensor<float>.OfShape(2, 0), axes, true, false);
        Assert.Equal(new int[] { 2, 1 }, fk.Dimensions.ToArray());
        Assert.Equal(new float[] { 0f, 0f }, fk.ToArray());
        var d = Tensor<double>.ReduceMean(DenseTensor<double>.OfShape(2, 0), axes, false, false);
        Assert.Equal(new double[] { 0.0, 0.0 }, d.ToArray());
        var i = Tensor<int>.ReduceMean(DenseTensor<int>.OfShape(2, 0), axes, false, false);
        Assert.Equal(new int[] { 0, 0 }, i.ToArray());
    }

    [Fact]
    public void EmptyExtent_MaxSum_Pinned()
    {
        var axes = new int[] { 1 }.ToTensor<int>();
        var max = Tensor<float>.ReduceMax(DenseTensor<float>.OfShape(2, 0), axes, false, false);
        Assert.Equal(new float[] { float.NegativeInfinity, float.NegativeInfinity }, max.ToArray());
        var sum = Tensor<float>.ReduceSum(DenseTensor<float>.OfShape(2, 0), axes, false, false);
        Assert.Equal(new float[] { 0f, 0f }, sum.ToArray());
        var isum = Tensor<int>.ReduceSum(DenseTensor<int>.OfShape(2, 0), axes, false, false);
        Assert.Equal(new int[] { 0, 0 }, isum.ToArray());
    }

    [Fact]
    public void CpuRouting_EmptyMean_YieldsZero()
    {
        var axes = new int[] { 1 }.ToTensor<int>();
        var r = CPUExecutionProvider.ReduceMean(DenseTensor<float>.OfShape(2, 0), axes, 0, 0, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new float[] { 0f, 0f }, ((Tensor<float>)r.Outputs[0]).ToArray());
    }

    [Fact]
    public void ReduceMax_Noop_ReturnsInputUnchanged()
    {
        var r = Tensor<float>.ReduceMax(Data(), new int[0].ToTensor<int>(), true, true);
        Assert.Equal(new int[] { 1, 2 }, r.Dimensions.ToArray());
        Assert.Equal(new float[] { 1f, 2f }, r.ToArray());
        var all = Tensor<float>.ReduceMax(Data(), null, true, false);
        Assert.Equal(new int[] { 1, 1 }, all.Dimensions.ToArray());
        Assert.Equal(new float[] { 2f }, all.ToArray());
    }

    [Fact]
    public void CpuRouting_HonorsNoop()
    {
        var r = CPUExecutionProvider.ReduceSum(Data(), new int[0].ToTensor<int>(), 1, 1, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new int[] { 1, 2 }, ((Tensor<float>)r.Outputs[0]).Dimensions.ToArray());
        var m = CPUExecutionProvider.ReduceMax(Data(), new int[0].ToTensor<int>(), 1, 1, null);
        Assert.Equal(OpStatus.Success, m.Status);
        Assert.Equal(new float[] { 1f, 2f }, ((Tensor<float>)m.Outputs[0]).ToArray());
    }

    static ComputationalGraph NoopMaxGraph(bool noop)
    {
        // C08: version 18 ReduceMax must observe noop_with_empty_axes.
        var mp = new OnnxModel { Name = "reducemax-noop" };
        mp.Opset[""] = 18;
        mp.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new int[] { 1, 2 } });
        mp.Inputs.Add(new OnnxValueInfo { Name = "axes", ElementType = TensorElementType.Int64, Dims = new int[] { 0 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Float, Dims = noop ? new int[] { 1, 2 } : new int[] { 1, 1 } });
        var attrs = new Dictionary<string, object> { { "keepdims", 1L } };
        if (noop) attrs["noop_with_empty_axes"] = 1L;
        mp.Nodes.Add(new OnnxNode { Name = "r", OpType = "ReduceMax", Inputs = new string[] { "x", "axes" }, Outputs = new string[] { "y" }, Attributes = attrs });
        return Model.Load(mp)!;
    }

    static Dictionary<string, ITensor> NoopMaxFeed()
    {
        return new Dictionary<string, ITensor>
        {
            ["x"] = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f } }),
            ["axes"] = DenseTensor<long>.OfValues(new long[0]),
        };
    }

    [Fact]
    public void DispatchV18_ReduceMax_NoopReturnsInput()
    {
        // ORT 1.29: noop=1 with empty axes returns [[1, 2]] unchanged.
        var graph = NoopMaxGraph(true);
        Assert.True(graph.Execute(NoopMaxFeed(), true), graph.LastErrorMessage);
        Assert.Equal(new float[] { 1f, 2f }, ((Tensor<float>)graph.Outputs["y"]).ToArray());
    }

    [Fact]
    public void DispatchV18_ReduceMax_WithoutNoopReducesAll()
    {
        // ORT 1.29: noop=0 with empty axes reduces to [[2]] (keepdims).
        var graph = NoopMaxGraph(false);
        Assert.True(graph.Execute(NoopMaxFeed(), true), graph.LastErrorMessage);
        Assert.Equal(new float[] { 2f }, ((Tensor<float>)graph.Outputs["y"]).ToArray());
    }

    [Fact]
    public void ReduceMax_SkipsNaN()
    {
        // ORT 1.29: [3].
        var x = DenseTensor<float>.OfValues(new float[] { 1f, float.NaN, 3f });
        var r = CPUExecutionProvider.ReduceMax(x, DenseTensor<int>.OfValues(new int[] { 0 }), 1, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new float[] { 3f }, ((Tensor<float>)r.Outputs[0]).ToArray());
    }

    [Fact]
    public void NullKeepDims_KeepsDims()
    {
        // ORT 1.29: omitted keepdims defaults to 1; [[1,2]] summed over
        // axis 0 stays [[1,2]] ([1,2]), not [1,2].
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f } });
        var r = CPUExecutionProvider.ReduceSum(x, DenseTensor<int>.OfValues(new int[] { 0 }), null, 0, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs![0];
        Assert.Equal(new int[] { 1, 2 }, y.Dimensions.ToArray());
        Assert.Equal(new float[] { 1f, 2f }, y.ToArray());
    }

    [Fact]
    public void ReduceMax_AllNaN_YieldsNaN()
    {
        // ORT 1.29: [nan] (the skip-NaN kernel inits from the first
        // element, so a leading NaN persists, as pinned mixed above).
        var x = DenseTensor<float>.OfValues(new float[] { float.NaN, float.NaN });
        var r = CPUExecutionProvider.ReduceMax(x, DenseTensor<int>.OfValues(new int[] { 0 }), 1, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.True(float.IsNaN(((Tensor<float>)r.Outputs[0])[0]));
    }

    [Fact]
    public void ReduceSum_PropagatesNaN()
    {
        // ORT 1.29: [nan].
        var x = DenseTensor<float>.OfValues(new float[] { 1f, float.NaN, 3f });
        var r = CPUExecutionProvider.ReduceSum(x, DenseTensor<int>.OfValues(new int[] { 0 }), 1, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.True(float.IsNaN(((Tensor<float>)r.Outputs[0])[0]));
    }

    [Fact]
    public void ReduceMean_PropagatesNaN()
    {
        // ORT 1.29: [nan].
        var x = DenseTensor<float>.OfValues(new float[] { 1f, float.NaN, 3f });
        var r = CPUExecutionProvider.ReduceMean(x, DenseTensor<int>.OfValues(new int[] { 0 }), 1, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.True(float.IsNaN(((Tensor<float>)r.Outputs[0])[0]));
    }

    [Fact]
    public void DoubleSum_BasicsMatchOrt()
    {
        // ORT 1.29: rows sum to [6, 15]; columns sum to [5, 7, 9].
        var x = DenseTensor<double>.OfValues(new double[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });
        var rows = CPUExecutionProvider.ReduceSum(x, DenseTensor<long>.OfValues(new long[] { 1L }), 0, 0, null);
        Assert.Equal(OpStatus.Success, rows.Status);
        Assert.Equal(new double[] { 6.0, 15.0 }, ((Tensor<double>)rows.Outputs[0]).ToArray());
        var cols = CPUExecutionProvider.ReduceSum(x, DenseTensor<long>.OfValues(new long[] { 0L }), 0, 0, null);
        Assert.Equal(OpStatus.Success, cols.Status);
        Assert.Equal(new double[] { 5.0, 7.0, 9.0 }, ((Tensor<double>)cols.Outputs[0]).ToArray());
    }

    [Fact]
    public void DoubleMean_BasicsMatchOrt()
    {
        // ORT 1.29 (opset 18): rows mean to [2, 5]; kept column means [[2.5, 3.5, 4.5]].
        var x = DenseTensor<double>.OfValues(new double[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });
        var rows = CPUExecutionProvider.ReduceMean(x, DenseTensor<long>.OfValues(new long[] { 1L }), 0, 0, null);
        Assert.Equal(OpStatus.Success, rows.Status);
        Assert.Equal(new double[] { 2.0, 5.0 }, ((Tensor<double>)rows.Outputs[0]).ToArray());
        var cols = CPUExecutionProvider.ReduceMean(x, DenseTensor<long>.OfValues(new long[] { 0L }), 1, 0, null);
        Assert.Equal(OpStatus.Success, cols.Status);
        var y = (Tensor<double>)cols.Outputs[0];
        Assert.Equal(new int[] { 1, 3 }, y.Dimensions.ToArray());
        Assert.Equal(new double[] { 2.5, 3.5, 4.5 }, y.ToArray());
    }

    [Fact]
    public void DoubleMax_BasicsMatchOrt()
    {
        // ORT 1.29 (opset 18): row maxes [5, 6], kept column maxes
        // [[4, 5, 6]], and axis -2 identical to axis 0.
        var x = DenseTensor<double>.OfValues(new double[,] { { 1.0, 5.0, 3.0 }, { 4.0, 2.0, 6.0 } });
        var rows = CPUExecutionProvider.ReduceMax(x, DenseTensor<long>.OfValues(new long[] { 1L }), 0, 0, null);
        Assert.Equal(OpStatus.Success, rows.Status);
        Assert.Equal(new double[] { 5.0, 6.0 }, ((Tensor<double>)rows.Outputs[0]).ToArray());
        var cols = CPUExecutionProvider.ReduceMax(x, DenseTensor<long>.OfValues(new long[] { 0L }), 1, 0, null);
        Assert.Equal(OpStatus.Success, cols.Status);
        var y = (Tensor<double>)cols.Outputs[0];
        Assert.Equal(new int[] { 1, 3 }, y.Dimensions.ToArray());
        Assert.Equal(new double[] { 4.0, 5.0, 6.0 }, y.ToArray());
        var neg = CPUExecutionProvider.ReduceMax(x, DenseTensor<long>.OfValues(new long[] { -2L }), 0, 0, null);
        Assert.Equal(OpStatus.Success, neg.Status);
        Assert.Equal(new double[] { 4.0, 5.0, 6.0 }, ((Tensor<double>)neg.Outputs[0]).ToArray());
    }
}
