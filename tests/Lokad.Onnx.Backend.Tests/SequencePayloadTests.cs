namespace Lokad.Onnx.Backend.Tests;

public class SequencePayloadTests
{
    static OnnxValueInfo Sequence(string name) => new() { Name = name, ElementType = TensorElementType.Sequence };

    static ComputationalGraph Identity(bool context)
    {
        var graph = Model.Load(new OnnxModel { Name = "sequence-payload", Opset = { [""] = 14 },
            Inputs = { Sequence("x") }, Outputs = { Sequence("y") },
            Nodes = { new OnnxNode { Name = "copy", OpType = "Identity", Inputs = ["x"], Outputs = ["y"] } } });
        return context ? graph.CreateExecution(ExecutionOptions.Memory) : graph;
    }

    [Theory]
    [InlineData(false, "float")] [InlineData(true, "float")]
    [InlineData(false, "mixed")] [InlineData(true, "mixed")]
    [InlineData(false, "alias")] [InlineData(true, "alias")]
    [InlineData(false, "empty")] [InlineData(true, "empty")]
    public void IdentityCountsSequenceElementsPerBindingAcrossResetAndRetry(bool context, string kind)
    {
        var first = DenseTensor<float>.OfValues([1, 2]);
        ITensor second = kind == "mixed" ? DenseTensor<long>.OfValues([3, 4, 5])
            : kind == "alias" ? first : DenseTensor<float>.OfValues([3, 4, 5]);
        var input = new TensorSequence(kind == "empty" ? [] : new ITensor[] { first, second });
        long expected = kind switch { "mixed" => 64, "alias" => 32, "empty" => 0, _ => 40 };
        var graph = Identity(context);
        Assert.True(graph.Execute(new ITensor[] { input }, false), graph.LastErrorMessage);
        Assert.Equal(expected, graph.LastPeakLiveBytes);
        var held = Assert.IsType<TensorSequence>(graph.Outputs["y"]);
        graph.Reset();
        Assert.False(graph.Execute(Array.Empty<ITensor>(), false));
        Assert.True(graph.Execute(new ITensor[] { input }, false), graph.LastErrorMessage);
        Assert.Equal(expected, graph.LastPeakLiveBytes);
        Assert.Equal(input.Items.Count, held.Items.Count);
        if (kind != "empty")
        {
            Assert.Same(first, held.Items[0]);
            Assert.Same(second, held.Items[1]);
            Assert.Equal(new float[] { 1, 2 }, first.ToArray());
        }
    }

    [Theory]
    [InlineData(false)] [InlineData(true)]
    public void SplitAndExtractIncludeSequencePayloadInPeak(bool context)
    {
        var graph = Model.Load(new OnnxModel { Name = "split-payload", Opset = { [""] = 14 },
            Inputs = { new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = [4] } },
            Outputs = { new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Float, Dims = [2] } },
            Initializers = {
                new OnnxTensor { Name = "sizes", ElementType = TensorElementType.Int64, Dims = [2], Data = new long[] { 2, 2 } },
                new OnnxTensor { Name = "index", ElementType = TensorElementType.Int64, Dims = [], Data = new long[] { 1 } } },
            Nodes = {
                new OnnxNode { Name = "split", OpType = "SplitToSequence", Inputs = ["x", "sizes"], Outputs = ["parts"] },
                new OnnxNode { Name = "extract", OpType = "SequenceAt", Inputs = ["parts", "index"], Outputs = ["y"] } } });
        if (context) graph = graph.CreateExecution(ExecutionOptions.Memory);
        var input = DenseTensor<float>.OfValues([1, 2, 3, 4]);
        Assert.True(graph.Execute(new ITensor[] { input }, true, ExecutionProvider.CPU, ExecutionOptions.Memory), graph.LastErrorMessage);
        Assert.Equal(32, graph.LastPeakLiveBytes);
        var held = (Tensor<float>)graph.Outputs["y"];
        Assert.Equal(new float[] { 3, 4 }, held.ToArray());
        graph.Reset();
        Assert.True(graph.Execute(new ITensor[] { input }, true, ExecutionProvider.CPU, ExecutionOptions.Memory), graph.LastErrorMessage);
        Assert.Equal(32, graph.LastPeakLiveBytes);
        Assert.Equal(new float[] { 3, 4 }, held.ToArray());
        Assert.Equal(new float[] { 1, 2, 3, 4 }, input.ToArray());
    }

    [Theory]
    [InlineData(false, false)] [InlineData(false, true)]
    [InlineData(true, false)] [InlineData(true, true)]
    public void NestedAndCyclicInputsAreRejectedBeforePublishingBindings(bool context, bool cycle)
    {
        var graph = Identity(context);
        var bad = new TensorSequence();
        bad.Add(cycle ? bad : new TensorSequence());
        // Resolve first: a failing baseline stops here before an unsafe recursive
        // ownership traversal could be reached by the malformed cyclic value.
        Assert.False(graph.ResolveInputs(new ITensor[] { bad }, false));
        Assert.Contains("sequence element", graph.LastErrorMessage);
        Assert.False(graph.Execute(new ITensor[] { bad }, false));
        var good = new TensorSequence(new ITensor[] { DenseTensor<float>.OfValues([1, 2]) });
        Assert.True(graph.Execute(new ITensor[] { good }, false), graph.LastErrorMessage);
        Assert.Equal(16, graph.LastPeakLiveBytes);
    }

    [Theory]
    [InlineData(false)] [InlineData(true)]
    public void HandBuiltGraphsRejectCyclicSequenceInputs(bool context)
    {
        var graph = new ComputationalGraph();
        graph.Opset[""] = 14;
        graph.Metadata["Name"] = "hand-built-sequence";
        graph.Inputs["x"] = new TensorSequence();
        graph.Outputs["y"] = new TensorSequence();
        graph.Nodes.Add(new Node { Name = "copy", Op = OpType.Identity, Inputs = ["x"], Outputs = ["y"] });
        if (context) graph = graph.CreateExecution(ExecutionOptions.Memory);
        var cycle = new TensorSequence(); cycle.Add(cycle);
        Assert.False(graph.Execute(new ITensor[] { cycle }, false));
        Assert.Contains("sequence element", graph.LastErrorMessage);
    }

    [Theory]
    [InlineData(false, false)] [InlineData(false, true)]
    [InlineData(true, false)] [InlineData(true, true)]
    public void StaticNestedSequenceValuesAreRejectedBeforeTraversal(bool attribute, bool cycle)
    {
        var bad = new TensorSequence(); bad.Add(cycle ? bad : new TensorSequence());
        var graph = new ComputationalGraph();
        graph.Opset[""] = 14;
        graph.Metadata["Name"] = "static-sequence";
        graph.OutputDescs.Add(Sequence("y"));
        if (attribute)
            graph.Nodes.Add(new Node { Name = "constant", Op = OpType.Constant, Inputs = [], Outputs = ["y"], Attributes = new() { ["value"] = bad } });
        else
        {
            graph.Initializers["s"] = bad;
            graph.Nodes.Add(new Node { Name = "copy", Op = OpType.Identity, Inputs = ["s"], Outputs = ["y"] });
        }
        Assert.False(graph.Execute(Array.Empty<ITensor>(), true));
        Assert.Contains("sequence element", graph.LastErrorMessage);
        var good = new TensorSequence(new ITensor[] { DenseTensor<float>.OfValues([1, 2]) });
        if (attribute) graph.Nodes[0].Attributes!["value"] = good;
        else graph.Initializers["s"] = good;
        graph.InvalidatePreparation();
        Assert.True(graph.Execute(Array.Empty<ITensor>(), true), graph.LastErrorMessage);
        Assert.Equal(8, graph.LastPeakLiveBytes);
    }
}
