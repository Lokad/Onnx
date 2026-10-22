using Lokad.Onnx.Optimization;

namespace Lokad.Onnx.Backend.Tests;

public class DeadNodeSweepTests
{
    [Theory]
    [InlineData(8)]
    [InlineData(512)]
    [InlineData(1024)]
    public void OneSweepRemovesEntireUnusedChain(int count)
    {
        var graph = Model.Load(UnusedChain(count), runOptimizer: false);
        var facts = GraphFacts.Build(graph);
        ConstFold.SweepDeadNodes(graph, facts);
        Assert.Empty(graph.Nodes);
        ConstFold.SweepDeadNodes(graph, GraphFacts.Build(graph));
        Assert.Empty(graph.Nodes);
    }

    [Theory]
    [InlineData(512)]
    [InlineData(1024)]
    public void ModelLoadRemovesEntireUnusedChain(int count)
    {
        var graph = Model.Load(UnusedChain(count));
        Assert.Empty(graph.Nodes);
        Assert.True(graph.Execute(Feed(), true), graph.LastErrorMessage);
        Assert.Equal(new[] { 3f }, ((Tensor<float>)graph.Outputs["x"]).ToArray());
    }

    [Fact]
    public void LiveSharedProducersKeepTheirOriginalOrder()
    {
        var model = UnusedChain(16);
        model.Outputs.Clear();
        model.Outputs.Add(Desc("y")); model.Outputs.Add(Desc("z"));
        model.Nodes.Insert(0, Op("a", "Neg", "x"));
        model.Nodes.Insert(2, Op("y", "Neg", "a"));
        model.Nodes.Add(Op("z", "Identity", "a"));
        var graph = Model.Load(model);
        Assert.Equal(new[] { "a", "y", "z" }, graph.Nodes.Select(n => n.Name));
        Assert.True(graph.Execute(Feed(), true), graph.LastErrorMessage);
        Assert.Equal(new[] { 3f }, ((Tensor<float>)graph.Outputs["y"]).ToArray());
        Assert.Equal(new[] { -3f }, ((Tensor<float>)graph.Outputs["z"]).ToArray());
    }

    [Fact]
    public void CapturesOfLiveBranchesSurviveWhileUnusedBranchesAndTheirCapturesDisappear()
    {
        var model = UnusedChain(0);
        model.Inputs.Add(new OnnxValueInfo { Name = "cond", ElementType = TensorElementType.Bool, Dims = [] });
        model.Outputs.Clear(); model.Outputs.Add(Desc("y"));
        model.Nodes.Add(Op("liveCapture", "Neg", "x"));
        model.Nodes.Add(Op("deadCapture", "Neg", "x"));
        model.Nodes.Add(Choose("unused", "deadCapture"));
        model.Nodes.Add(Choose("y", "liveCapture"));
        var graph = Model.Load(model);
        Assert.Equal(new[] { "liveCapture", "y" }, graph.Nodes.Select(n => n.Name));
        var feed = Feed(); feed["cond"] = new DenseTensor<bool>(new[] { true }, Array.Empty<int>());
        Assert.True(graph.Execute(feed, true), graph.LastErrorMessage);
        Assert.Equal(new[] { -3f }, ((Tensor<float>)graph.Outputs["y"]).ToArray());
    }

    [Fact]
    public void DuplicateProducersRemainAvailableForPreparationValidation()
    {
        var model = UnusedChain(0);
        model.Outputs.Clear(); model.Outputs.Add(Desc("y"));
        model.Nodes.Add(Op("y", "Identity", "x"));
        model.Nodes.Add(Op("y", "Neg", "x"));
        var graph = Model.Load(model);
        Assert.Equal(2, graph.Nodes.Count);
        Assert.False(graph.Execute(Feed(), true));
        Assert.Contains("duplicate producers", graph.LastErrorMessage);
    }

    static OnnxNode Choose(string output, string capture)
    {
        OnnxSubgraph Branch() => new() { Outputs = { Desc(capture) } };
        return new OnnxNode { Name = output, OpType = "If", Inputs = ["cond"], Outputs = [output],
            Attributes = new() { ["then_branch"] = Branch(), ["else_branch"] = Branch() } };
    }

    static OnnxNode Op(string output, string op, string input) =>
        new() { Name = output, OpType = op, Inputs = [input], Outputs = [output] };

    static OnnxValueInfo Desc(string name) => new() { Name = name, ElementType = TensorElementType.Float, Dims = [1] };

    static Dictionary<string, ITensor> Feed() => new() { ["x"] = DenseTensor<float>.OfValues(new[] { 3f }) };

    static OnnxModel UnusedChain(int count)
    {
        var model = new OnnxModel { Name = "dead-chain", Opset = { [""] = 17 }, Inputs = { Desc("x") }, Outputs = { Desc("x") } };
        for (int i = 0; i < count; i++) model.Nodes.Add(Op("dead" + i, "Identity", i == 0 ? "x" : "dead" + (i - 1)));
        return model;
    }
}
