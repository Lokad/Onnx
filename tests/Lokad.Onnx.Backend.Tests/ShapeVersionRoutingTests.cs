namespace Lokad.Onnx.Backend.Tests;

public class ShapeVersionRoutingTests
{
    static ComputationalGraph NewGraph(int opset)
    {
        var g = new ComputationalGraph();
        g.Metadata["Name"] = "test";
        g.Opset[""] = opset;
        return g;
    }

    static void Run(ComputationalGraph graph, Dictionary<string, ITensor> inputs)
    {
        foreach (var kv in inputs) graph.Inputs[kv.Key] = kv.Value;
        Assert.True(graph.Execute(inputs, true));
    }

    [Fact]
    public void Squeeze_AttributeForm_OldOpset()
    {
        var g2 = NewGraph(11);
        g2.Inputs["x"] = DenseTensor<float>.OfValues(new float[1, 1, 3] { { { 1f, 2f, 3f } } });
        g2.Outputs["y"] = DenseTensor<float>.OfShape(1, 3);
        g2.Nodes.Add(new Node
        {
            Name = "sq", Op = OpType.Squeeze,
            Inputs = new[] { "x" }, Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object> { { "axes", new int[] { 0 } } },
        });
        g2.RefreshLifetimeAnalysis();
        Run(g2, new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[1, 1, 3] { { { 1f, 2f, 3f } } }) } });
        var y = (Tensor<float>)g2.Outputs["y"];
        Assert.Equal(new int[] { 1, 3 }, y.Dimensions.ToArray());
        Assert.Equal(new float[] { 1f, 2f, 3f }, y.ToArray());
    }

    [Fact]
    public void Squeeze_InputForm_NewOpset_Equivalent()
    {
        var g = NewGraph(13);
        g.Inputs["x"] = DenseTensor<float>.OfValues(new float[1, 1, 3] { { { 1f, 2f, 3f } } });
        g.Inputs["a"] = DenseTensor<long>.OfValues(new long[] { 0 });
        g.Outputs["y"] = DenseTensor<float>.OfShape(1, 3);
        g.Nodes.Add(new Node { Name = "sq", Op = OpType.Squeeze, Inputs = new[] { "x", "a" }, Outputs = new[] { "y" } });
        g.RefreshLifetimeAnalysis();
        Run(g, new Dictionary<string, ITensor>
        {
            { "x", DenseTensor<float>.OfValues(new float[1, 1, 3] { { { 1f, 2f, 3f } } }) },
            { "a", DenseTensor<long>.OfValues(new long[] { 0 }) },
        });
        Assert.Equal(new int[] { 1, 3 }, ((Tensor<float>)g.Outputs["y"]).Dimensions.ToArray());
    }

    [Fact]
    public void Squeeze_EmptyAxes_SqueezesAll()
    {
        var x = DenseTensor<float>.OfValues(new float[1, 1, 2] { { { 1f, 2f } } });
        var r = CPUExecutionProvider.Squeeze(x, new int[0].ToTensor<int>(), null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new int[] { 2 }, ((Tensor<float>)r.Outputs[0]).Dimensions.ToArray());
    }

    [Fact]
    public void Squeeze_ScalarAxes_Fails()
    {
        var x = DenseTensor<float>.OfValues(new float[1, 3] { { 1f, 2f, 3f } });
        var scalar = DenseTensor<long>.OfShape();
        scalar.SetValue(0, 0L);
        var r = CPUExecutionProvider.Squeeze(x, scalar, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }

    [Fact]
    public void ReduceSum_AttributeForm_OldOpset()
    {
        var g = NewGraph(11);
        g.Outputs["y"] = DenseTensor<float>.OfShape(1, 2);
        g.Nodes.Add(new Node
        {
            Name = "rs", Op = OpType.ReduceSum,
            Inputs = new[] { "x" }, Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object> { { "axes", new int[] { 0 } } },
        });
        g.RefreshLifetimeAnalysis();
        Run(g, new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[,] { { 1f, 2f } }) } });
        var y = (Tensor<float>)g.Outputs["y"];
        Assert.Equal(new int[] { 1, 2 }, y.Dimensions.ToArray());
        Assert.Equal(new float[] { 1f, 2f }, y.ToArray());
    }

    [Fact]
    public void Slice_AttributeForm_OldOpset()
    {
        var g = NewGraph(9);
        g.Outputs["y"] = DenseTensor<float>.OfShape(1, 3);
        g.Nodes.Add(new Node
        {
            Name = "sl", Op = OpType.Slice,
            Inputs = new[] { "x" }, Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object>
            {
                { "starts", new int[] { 1 } },
                { "ends", new int[] { 2 } },
                { "axes", new int[] { 0 } },
            },
        });
        g.RefreshLifetimeAnalysis();
        Run(g, new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } }) } });
        var y = (Tensor<float>)g.Outputs["y"];
        Assert.Equal(new int[] { 1, 3 }, y.Dimensions.ToArray());
        Assert.Equal(new float[] { 4f, 5f, 6f }, y.ToArray());
    }

    [Fact]
    public void Slice_OptionalPositions_Retained()
    {
        var g = NewGraph(10);
        g.Outputs["y"] = DenseTensor<float>.OfShape(1, 3);
        g.Nodes.Add(new Node
        {
            Name = "sl", Op = OpType.Slice,
            Inputs = new[] { "x", "s", "e", "", "st" }, Outputs = new[] { "y" },
        });
        g.Initializers["s"] = DenseTensor<long>.OfValues(new long[] { 0 });
        g.Initializers["e"] = DenseTensor<long>.OfValues(new long[] { 2 });
        g.Initializers["st"] = DenseTensor<long>.OfValues(new long[] { 2 });
        g.RefreshLifetimeAnalysis();
        Run(g, new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } }) } });
        var y = (Tensor<float>)g.Outputs["y"];
        Assert.Equal(new int[] { 1, 3 }, y.Dimensions.ToArray());
        Assert.Equal(new float[] { 1f, 2f, 3f }, y.ToArray());
    }

    [Fact]
    public void Split_LegacyEqual_ByOutputCount()
    {
        var g = NewGraph(11);
        g.Outputs["a"] = DenseTensor<float>.OfShape(1, 2);
        g.Outputs["b"] = DenseTensor<float>.OfShape(1, 2);
        g.Nodes.Add(new Node
        {
            Name = "sp", Op = OpType.Split,
            Inputs = new[] { "x" }, Outputs = new[] { "a", "b" },
            Attributes = new Dictionary<string, object> { { "axis", 1 } },
        });
        g.RefreshLifetimeAnalysis();
        Run(g, new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[,] { { 0f, 1f, 2f, 3f } }) } });
        Assert.Equal(new float[] { 0f, 1f }, ((Tensor<float>)g.Outputs["a"]).ToArray());
        Assert.Equal(new float[] { 2f, 3f }, ((Tensor<float>)g.Outputs["b"]).ToArray());
    }

    [Fact]
    public void Split_LegacyUneven_Fails()
    {
        var x = DenseTensor<float>.OfValues(new float[,] { { 0f, 1f, 2f, 3f, 4f } });
        var r = CPUExecutionProvider.Split(x, null, 1, null, null, null, 2);
        Assert.Equal(OpStatus.Failure, r.Status);
    }

    [Fact]
    public void Split_NumOutputs_Distributes()
    {
        var x = DenseTensor<float>.OfValues(new float[,] { { 0f, 1f, 2f, 3f, 4f } });
        var r = CPUExecutionProvider.Split(x, null, 1, null, 2, null, 2);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new float[] { 0f, 1f, 2f }, ((Tensor<float>)r.Outputs[0]).ToArray());
        Assert.Equal(new float[] { 3f, 4f }, ((Tensor<float>)r.Outputs[1]).ToArray());
    }

    [Fact]
    public void Split_UnevenSizes_Work()
    {
        var x = DenseTensor<float>.OfValues(new float[,] { { 0f, 1f, 2f, 3f, 4f, 5f } });
        var r = CPUExecutionProvider.Split(x, new int[] { 4, 2 }.ToTensor<int>(), 1, null, null, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new int[] { 1, 4 }, ((Tensor<float>)r.Outputs[0]).Dimensions.ToArray());
        Assert.Equal(new int[] { 1, 2 }, ((Tensor<float>)r.Outputs[1]).Dimensions.ToArray());
    }

    [Fact]
    public void Split_SplitPlusNumOutputs_Fails()
    {
        var x = DenseTensor<float>.OfValues(new float[,] { { 0f, 1f, 2f, 3f } });
        var r = CPUExecutionProvider.Split(x, new int[] { 2, 2 }.ToTensor<int>(), 1, null, 2, null, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }

    [Fact]
    public void Split_ScalarSplit_Fails()
    {
        var x = DenseTensor<float>.OfValues(new float[,] { { 0f, 1f, 2f, 3f } });
        var scalar = DenseTensor<long>.OfShape();
        scalar.SetValue(0, 4L);
        var r = CPUExecutionProvider.Split(x, scalar, 1, null, null, null, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }

    [Fact]
    public void SplitToSequence_AbsentSplit_SizeOneChunks()
    {
        var x = DenseTensor<float>.OfValues(new float[,] { { 0f, 1f, 2f, 3f } });
        var r = CPUExecutionProvider.SplitToSequence(x, null, 1, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var seq = (TensorSequence)r.Outputs[0];
        Assert.Equal(4, seq.Items.Count);
        Assert.Equal(new int[] { 1, 1 }, ((Tensor<float>)seq.Items[0]).Dimensions.ToArray());
        Assert.Equal(new float[] { 0f }, ((Tensor<float>)seq.Items[0]).ToArray());
        Assert.Equal(new float[] { 3f }, ((Tensor<float>)seq.Items[3]).ToArray());
        var rk = CPUExecutionProvider.SplitToSequence(x, null, 1, 0, null);
        Assert.Equal(OpStatus.Success, rk.Status);
        var seqk = (TensorSequence)rk.Outputs[0];
        Assert.Equal(4, seqk.Items.Count);
        Assert.Equal(new int[] { 1 }, ((Tensor<float>)seqk.Items[0]).Dimensions.ToArray());
    }

    [Fact]
    public void SplitToSequence_ScalarKeepdims0_NonSingleton_Fails()
    {
        var x = DenseTensor<float>.OfValues(new float[5, 1] { { 0f }, { 1f }, { 2f }, { 3f }, { 4f } });
        var scalar = DenseTensor<long>.OfShape();
        scalar.SetValue(0, 3L);
        var r = CPUExecutionProvider.SplitToSequence(x, scalar, 0, 0, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }

    [Fact]
    public void SplitToSequence_ScalarSplit_ChunkSize()
    {
        var x = DenseTensor<float>.OfValues(new float[5, 1] { { 0f }, { 1f }, { 2f }, { 3f }, { 4f } });
        var scalar = DenseTensor<long>.OfShape();
        scalar.SetValue(0, 3L);
        var r = CPUExecutionProvider.SplitToSequence(x, scalar, 0, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var seq = (TensorSequence)r.Outputs[0];
        Assert.Equal(2, seq.Items.Count);
        Assert.Equal(new int[] { 3, 1 }, ((Tensor<float>)seq.Items[0]).Dimensions.ToArray());
        Assert.Equal(new int[] { 2, 1 }, ((Tensor<float>)seq.Items[1]).Dimensions.ToArray());
    }

    [Fact]
    public void SplitToSequence_ScalarSplit_NonPositive_Fails()
    {
        var x = DenseTensor<float>.OfValues(new float[,] { { 0f, 1f } });
        var scalar = DenseTensor<long>.OfShape();
        scalar.SetValue(0, 0L);
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.SplitToSequence(x, scalar, 1, null, null).Status);
    }

    [Fact]
    public void SplitToSequence_VectorSplit_IgnoresKeepdims()
    {
        var x = DenseTensor<float>.OfValues(new float[,] { { 0f, 1f, 2f, 3f } });
        var r = CPUExecutionProvider.SplitToSequence(x, new int[] { 1, 3 }.ToTensor<int>(), 1, 0, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var seq = (TensorSequence)r.Outputs[0];
        Assert.Equal(new int[] { 1, 1 }, ((Tensor<float>)seq.Items[0]).Dimensions.ToArray());
        Assert.Equal(new int[] { 1, 3 }, ((Tensor<float>)seq.Items[1]).Dimensions.ToArray());
    }

    [Fact]
    public void EmptyOutputNames_NotBound()
    {
        var g = NewGraph(13);
        g.Outputs["a"] = DenseTensor<float>.OfShape(1, 2);
        g.Outputs["c"] = DenseTensor<float>.OfShape(1, 2);
        g.Nodes.Add(new Node
        {
            Name = "sp", Op = OpType.Split,
            Inputs = new[] { "x" }, Outputs = new[] { "a", "", "c" },
            Attributes = new Dictionary<string, object> { { "axis", 1 } },
        });
        g.RefreshLifetimeAnalysis();
        Run(g, new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[,] { { 0f, 1f, 2f, 3f, 4f, 5f } }) } });
        Assert.Equal(new float[] { 0f, 1f }, ((Tensor<float>)g.Outputs["a"]).ToArray());
        Assert.Equal(new float[] { 4f, 5f }, ((Tensor<float>)g.Outputs["c"]).ToArray());
        Assert.False(g.Outputs.ContainsKey(""));
        Assert.False(g.IntermediateOutputs.ContainsKey(""));
    }

    [Fact]
    public void Shape_ReversedSlice_NewOpset_ReturnsEmpty()
    {
        var g = NewGraph(15);
        g.Outputs["z"] = DenseTensor<long>.OfShape(0);
        g.Nodes.Add(new Node
        {
            Name = "sh", Op = OpType.Shape,
            Inputs = new[] { "x" }, Outputs = new[] { "z" },
            Attributes = new Dictionary<string, object> { { "start", 2L }, { "end", 1L } },
        });
        g.RefreshLifetimeAnalysis();
        Run(g, new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfShape(2, 3, 4) } });
        var z = (Tensor<long>)g.Outputs["z"];
        Assert.Equal(new int[] { 0 }, z.Dimensions.ToArray());
        Assert.Empty(z.ToArray());
    }

    [Fact]
    public void Shape_NoSlice_OldOpset_ReturnsFullShape()
    {
        var g = NewGraph(13);
        g.Outputs["z"] = DenseTensor<long>.OfShape(3);
        g.Nodes.Add(new Node { Name = "sh", Op = OpType.Shape, Inputs = new[] { "x" }, Outputs = new[] { "z" } });
        g.RefreshLifetimeAnalysis();
        Run(g, new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfShape(2, 3, 4) } });
        Assert.Equal(new long[] { 2L, 3L, 4L }, ((Tensor<long>)g.Outputs["z"]).ToArray());
    }
}
