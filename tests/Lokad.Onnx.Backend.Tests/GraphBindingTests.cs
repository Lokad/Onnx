namespace Lokad.Onnx.Backend.Tests;

public class GraphBindingTests
{
    static ComputationalGraph NewGraph()
    {
        var g = new ComputationalGraph();
        g.Metadata["Name"] = "test";
        return g;
    }

    [Fact]
    public void ResolveInputs_Dict_WrongName_False()
    {
        var g = NewGraph();
        g.Inputs["x"] = DenseTensor<float>.OfShape(2);
        var user = new Dictionary<string, ITensor> { { "z", DenseTensor<float>.OfValues(new float[] { 1f, 2f }) } };
        Assert.False(g.ResolveInputs(user, false));
    }

    [Fact]
    public void ResolveInputs_Dict_ExtraName_False()
    {
        var g = NewGraph();
        g.Inputs["x"] = DenseTensor<float>.OfShape(2);
        var user = new Dictionary<string, ITensor>
        {
            { "x", DenseTensor<float>.OfValues(new float[] { 1f, 2f }) },
            { "z", DenseTensor<float>.OfValues(new float[] { 1f, 2f }) },
        };
        Assert.False(g.ResolveInputs(user, false));
    }

    [Fact]
    public void ResolveInputs_WrongDims_False()
    {
        var g = NewGraph();
        g.Inputs["x"] = DenseTensor<float>.OfShape(2);
        var bad = DenseTensor<float>.OfValues(new float[] { 1f, 2f, 3f });
        Assert.False(g.ResolveInputs(new Dictionary<string, ITensor> { { "x", bad } }, false));
        var g2 = NewGraph();
        g2.Inputs["x"] = DenseTensor<float>.OfShape(2);
        Assert.False(g2.ResolveInputs(new ITensor[] { bad }, false));
    }

    [Fact]
    public void ResolveInputs_SymbolicDims_Accepted()
    {
        var g = NewGraph();
        g.Inputs["x"] = DenseTensor<float>.OfShape(0, 3);
        var user = new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } }) } };
        Assert.True(g.ResolveInputs(user, false));
        var g2 = NewGraph();
        g2.Inputs["x"] = DenseTensor<float>.OfShape(0, 3);
        Assert.True(g2.ResolveInputs(new ITensor[] { DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f } }) }, false));
    }

    [Fact]
    public void ExecuteNode_MissingLabel_False()
    {
        var g = NewGraph();
        g.Inputs["x"] = DenseTensor<float>.OfShape(2);
        g.Outputs["y"] = DenseTensor<float>.OfShape(2);
        g.Nodes.Add(new Node { Name = "r", Op = OpType.Relu, Inputs = new[] { "x" }, Outputs = new[] { "y" } });
        g.RefreshLifetimeAnalysis();
        var user = new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[] { 1f, 2f }) } };
        Assert.False(g.ExecuteNode(user, "nope", false));
    }

    [Fact]
    public void Passthrough_Output_AliasesInput()
    {
        var g = NewGraph();
        g.Inputs["x"] = DenseTensor<float>.OfShape(2);
        g.Outputs["x"] = DenseTensor<float>.OfShape(2);
        g.RefreshLifetimeAnalysis();
        var user = new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[] { 1f, 2f }) } };
        Assert.True(g.Execute(user, false));
        Assert.Equal(new float[] { 1f, 2f }, ((Tensor<float>)g.Outputs["x"]).ToArray());
    }

    [Fact]
    public void Passthrough_Output_AliasesInitializer()
    {
        var g = NewGraph();
        g.Initializers["w"] = DenseTensor<float>.OfValues(new float[] { 7f, 8f });
        g.Outputs["w"] = DenseTensor<float>.OfShape(2);
        g.RefreshLifetimeAnalysis();
        Assert.True(g.Execute(new Dictionary<string, ITensor>(), true));
        Assert.Equal(new float[] { 7f, 8f }, ((Tensor<float>)g.Outputs["w"]).ToArray());
    }

    [Fact]
    public void IntermediateProducedOutput_Resolves()
    {
        var g = NewGraph();
        g.Inputs["x"] = DenseTensor<float>.OfShape(2);
        g.Outputs["t"] = DenseTensor<float>.OfShape(2);
        g.Nodes.Add(new Node { Name = "r", Op = OpType.Relu, Inputs = new[] { "x" }, Outputs = new[] { "t" } });
        g.IntermediateOutputs["t"] = null;
        g.RefreshLifetimeAnalysis();
        var user = new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[] { -1f, 2f }) } };
        Assert.True(g.Execute(user, false));
        Assert.Equal(new float[] { 0f, 2f }, ((Tensor<float>)g.Outputs["t"]).ToArray());
    }

    [Fact]
    public void ResolveNode_DupInputs_SingleTensor()
    {
        var g = NewGraph();
        var t = DenseTensor<float>.OfValues(new float[] { 1f, 2f });
        var node = new Node { Name = "a", Op = OpType.Add, Inputs = new[] { "x", "x" }, Outputs = new[] { "y" } };
        Assert.True(g.ResolveNodeExecuteInputs(node, new ITensor[] { t }, false));
        Assert.Same(t, g.Inputs["x"]);
        var g2 = NewGraph();
        Assert.True(g2.ResolveNodeExecuteInputs(node, new Dictionary<string, ITensor> { { "x", t } }, false));
    }

    [Fact]
    public void ResolveNode_EmptySlot_Ignored()
    {
        var g = NewGraph();
        var t = DenseTensor<float>.OfValues(new float[] { 1f, 2f });
        var node = new Node { Name = "s", Op = OpType.Slice, Inputs = new[] { "x", "" }, Outputs = new[] { "y" } };
        Assert.True(g.ResolveNodeExecuteInputs(node, new ITensor[] { t }, false));
        var g2 = NewGraph();
        Assert.True(g2.ResolveNodeExecuteInputs(node, new Dictionary<string, ITensor> { { "x", t } }, false));
    }

    [Fact]
    public void Failure_InvalidatesOutputs_And_Retry()
    {
        var g = NewGraph();
        g.Inputs["x"] = DenseTensor<float>.OfShape(2);
        g.Outputs["y"] = DenseTensor<float>.OfShape(2);
        g.Nodes.Add(new Node { Name = "r", Op = OpType.Relu, Inputs = new[] { "x" }, Outputs = new[] { "y" } });
        g.RefreshLifetimeAnalysis();
        var good = new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[] { 1f, 2f }) } };
        Assert.True(g.Execute(good, false));
        Assert.Equal(new float[] { 1f, 2f }, ((Tensor<float>)g.Outputs["y"]).ToArray());
        var bad = new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[] { 1f, 2f, 3f }) } };
        Assert.False(g.Execute(bad, false));
        Assert.False(g.Outputs.ContainsKey("y"));
        Assert.True(g.Execute(good, false));
        Assert.Equal(new float[] { 1f, 2f }, ((Tensor<float>)g.Outputs["y"]).ToArray());
    }

    [Fact]
    public void NodeFailure_InvalidatesOutputs()
    {
        var g = NewGraph();
        g.Inputs["x"] = DenseTensor<float>.OfShape(2);
        g.Outputs["y"] = DenseTensor<float>.OfShape(2);
        g.Outputs["z"] = DenseTensor<float>.OfShape(2);
        g.Nodes.Add(new Node { Name = "r", Op = OpType.Relu, Inputs = new[] { "x" }, Outputs = new[] { "y" } });
        g.Nodes.Add(new Node { Name = "a", Op = OpType.Add, Inputs = new[] { "y", "nope" }, Outputs = new[] { "z" } });
        g.IntermediateOutputs["y"] = null;
        g.RefreshLifetimeAnalysis();
        var user = new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[] { 1f, 2f }) } };
        Assert.False(g.Execute(user, false));
        Assert.False(g.Outputs.ContainsKey("y"));
        Assert.False(g.Outputs.ContainsKey("z"));
    }
}
