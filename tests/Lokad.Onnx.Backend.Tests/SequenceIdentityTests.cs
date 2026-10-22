namespace Lokad.Onnx.Backend.Tests;

public class SequenceIdentityTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(2)]
    public void ProviderPreservesItemsWithIndependentSequenceMetadataAndStructure(int count)
    {
        var items = Enumerable.Range(0, count).Select(i => (ITensor)DenseTensor<float>.OfValues([i, -i])).ToArray();
        var input = new TensorSequence(items) { Name = "input" };
        var result = CPUExecutionProvider.Identity(input, null);
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.Equal(OpType.Identity, result.Op);
        var output = Assert.IsType<TensorSequence>(Assert.Single(result.Outputs));
        Assert.NotSame(input, output);
        Assert.Equal(count, output.Items.Count);
        for (int i = 0; i < count; i++) Assert.Same(items[i], output.Items[i]);
        output.Name = "output";
        output.Add(DenseTensor<bool>.Scalar(true));
        if (count > 0) output.SetValue(0, DenseTensor<float>.Scalar(99));
        Assert.Equal("input", input.Name);
        Assert.Equal(count, input.Items.Count);
        for (int i = 0; i < count; i++) Assert.Same(items[i], input.Items[i]);
    }

    [Fact]
    public void ProviderValidatesExecutionOptionsForSequences()
    {
        var invalid = new ExecutionOptions(OptimizationMode.Speed, new TensorExecutionOptions(false, false, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => CPUExecutionProvider.Identity(new TensorSequence(), invalid));
    }

    [Theory]
    [InlineData(14, 0)]
    [InlineData(14, 2)]
    [InlineData(18, 0)]
    [InlineData(18, 2)]
    public void SplitThenIdentityMatchesOrt(int version, int rows)
    {
        // ORT 1.29 accepts these models from opset 14, including empty sequences.
        var model = new OnnxModel { Name = "sequence-identity" };
        model.Opset[""] = version;
        model.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = [rows, 2] });
        model.Outputs.Add(new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Sequence });
        model.Nodes.Add(new OnnxNode { Name = "split", OpType = "SplitToSequence", Inputs = ["x"], Outputs = ["q"],
            Attributes = new() { ["axis"] = 0 } });
        model.Nodes.Add(new OnnxNode { Name = "identity", OpType = "Identity", Inputs = ["q"], Outputs = ["y"] });
        var graph = Model.Load(model);
        var values = Enumerable.Range(0, rows * 2).Select(i => (float)i).ToArray();
        Assert.True(graph.Execute(new ITensor[] { new DenseTensor<float>(values, [rows, 2]) }, true), graph.LastErrorMessage);
        var output = Assert.IsType<TensorSequence>(graph.Outputs["y"]);
        Assert.Equal("y", output.Name);
        Assert.Equal(rows, output.Items.Count);
        for (int i = 0; i < rows; i++)
        {
            Assert.Equal(new[] { 1, 2 }, output.Items[i].Dims);
            Assert.Equal(values.Skip(i * 2).Take(2), ((Tensor<float>)output.Items[i]).ToArray());
        }
    }

    [Theory]
    [InlineData(14)]
    [InlineData(18)]
    public void SequenceAtConsumesIdentityOutput(int version)
    {
        var graph = new ComputationalGraph();
        graph.Metadata["Name"] = "sequence-identity-at";
        graph.Opset[""] = version;
        graph.Inputs["s"] = new TensorSequence();
        graph.InputDescs.Add(new OnnxValueInfo { Name = "s", ElementType = TensorElementType.Sequence });
        graph.Initializers["i"] = DenseTensor<long>.Scalar(-1);
        graph.Outputs["y"] = DenseTensor<int>.OfShape(2);
        graph.IntermediateOutputs["q"] = null;
        graph.Nodes.Add(new Node { Name = "identity", Op = OpType.Identity, Inputs = ["s"], Outputs = ["q"] });
        graph.Nodes.Add(new Node { Name = "at", Op = OpType.SequenceAt, Inputs = ["q", "i"], Outputs = ["y"] });
        graph.RefreshLifetimeAnalysis();
        var input = new TensorSequence(new ITensor[] { DenseTensor<int>.OfValues([1, 2]), DenseTensor<int>.OfValues([3, 4]) });
        Assert.True(graph.Execute(new Dictionary<string, ITensor> { ["s"] = input }, true), graph.LastErrorMessage);
        Assert.Equal(new[] { 3, 4 }, ((Tensor<int>)graph.Outputs["y"]).ToArray());
        Assert.False(graph.Execute(new Dictionary<string, ITensor> { ["s"] = new TensorSequence() }, true));
        Assert.Equal("at", graph.LastFailedNodeName);
    }

    [Theory]
    [InlineData(0, 0, "", 0, false)]
    [InlineData(13, 0, "", 0, false)]
    [InlineData(14, 0, "", 0, true)]
    [InlineData(18, 0, "", 0, true)]
    [InlineData(18, 13, "", 0, false)]
    [InlineData(13, 14, "", 0, true)]
    [InlineData(18, 0, "ai.onnx", 13, false)]
    [InlineData(13, 0, "ai.onnx", 14, true)]
    [InlineData(14, 0, "ai.onnx", 0, true)]
    public void DispatchRespectsResolvedVersion(int graphVersion, int nodeVersion, string domain, int domainVersion, bool supported)
    {
        var graph = new ComputationalGraph();
        if (graphVersion > 0) graph.Opset[""] = graphVersion;
        if (domainVersion > 0) graph.Opset[domain] = domainVersion;
        var input = new TensorSequence(new ITensor[] { DenseTensor<int>.Scalar(7) }) { Name = "original" };
        graph.Inputs["s"] = input;
        var node = new Node { Name = "identity", Op = OpType.Identity, OpTypeName = "Identity", Domain = domain,
            OpsetVersion = nodeVersion, Inputs = ["s"], Outputs = ["y"] };
        var result = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(supported ? OpStatus.Success : OpStatus.Failure, result.Status);
        Assert.Equal(OpType.Identity, result.Op);
        Assert.Equal("original", input.Name);
        if (supported)
        {
            var output = Assert.IsType<TensorSequence>(Assert.Single(result.Outputs));
            Assert.Equal("y", output.Name);
            Assert.Same(input.Items[0], Assert.Single(output.Items));
        }
        else Assert.Empty(result.Outputs);

        // The version guard must only restrict sequences, not older numeric Identity.
        graph.Inputs["s"] = DenseTensor<int>.Scalar(7);
        var numeric = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Success, numeric.Status);
        Assert.Equal(7, ((Tensor<int>)Assert.Single(numeric.Outputs)).GetValue(0));
    }

    [Theory]
    [InlineData(false, false, 0)]
    [InlineData(false, false, 2)]
    [InlineData(false, true, 0)]
    [InlineData(false, true, 2)]
    [InlineData(true, false, 0)]
    [InlineData(true, false, 2)]
    [InlineData(true, true, 0)]
    [InlineData(true, true, 2)]
    public void SequenceOutputsSurviveResetPoolPressureAndLaterExecutions(bool memory, bool reuseContext, int count)
    {
        var options = memory ? ExecutionOptions.Memory : ExecutionOptions.Default;
        var plan = new ComputationalGraph();
        plan.Metadata["Name"] = "sequence-identity-lifetime";
        plan.Opset[""] = 18;
        plan.Inputs["x"] = DenseTensor<float>.OfShape(2);
        plan.Outputs["y"] = new TensorSequence();
        plan.Outputs["z"] = DenseTensor<float>.OfShape(2);
        plan.IntermediateOutputs["t"] = null;
        plan.IntermediateOutputs["s"] = null;
        plan.Nodes.Add(new Node { Name = "sum", Op = OpType.Add, Inputs = ["x", "x"], Outputs = ["t"] });
        if (count > 0)
            plan.Nodes.Add(new Node { Name = "split", Op = OpType.SplitToSequence, Inputs = ["t"], Outputs = ["s"] });
        else
            plan.Initializers["s"] = new TensorSequence();
        plan.Nodes.Add(new Node { Name = "identity", Op = OpType.Identity, Inputs = ["s"], Outputs = ["y"] });
        plan.Nodes.Add(new Node { Name = "pressure", Op = OpType.Relu, Inputs = ["x"], Outputs = ["z"] });
        plan.RefreshLifetimeAnalysis();
        var graph = reuseContext ? plan.CreateExecution(options) : plan;
        var held = new List<(TensorSequence Sequence, float[][] Values)>();
        for (int run = 1; run <= 4; run++)
        {
            graph.Reset();
            var input = DenseTensor<float>.OfValues([run, -run]);
            Assert.True(graph.Execute(new Dictionary<string, ITensor> { ["x"] = input }, true, ExecutionProvider.CPU, options), graph.LastErrorMessage);
            var current = Assert.IsType<TensorSequence>(graph.Outputs["y"]);
            Assert.Equal(count, current.Items.Count);
            for (int i = 0; i < count; i++) Assert.Equal(new[] { (i == 0 ? run : -run) * 2f }, ((Tensor<float>)current.Items[i]).ToArray());
            foreach (var previous in held)
            {
                Assert.NotSame(previous.Sequence, current);
                for (int i = 0; i < count; i++) Assert.Equal(previous.Values[i], ((Tensor<float>)previous.Sequence.Items[i]).ToArray());
            }
            held.Add((current, current.Items.Select(item => ((Tensor<float>)item).ToArray()).ToArray()));
        }
    }
}
