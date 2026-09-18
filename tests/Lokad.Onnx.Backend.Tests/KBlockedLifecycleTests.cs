namespace Lokad.Onnx.Backend.Tests;

public class KBlockedLifecycleTests
{
    static ComputationalGraph BuildGraph()
    {
        var x = DenseTensor<float>.OfShape(new int[] { 16, 1024 });
        var w = DenseTensor<float>.OfShape(new int[] { 1024, 1024 });
        var graph = new ComputationalGraph();
        graph.Metadata["Name"] = "kb-lifecycle";
        graph.Inputs["x"] = x;
        graph.Initializers["w"] = w;
        graph.Outputs["z"] = DenseTensor<float>.OfShape(new int[] { 16, 1024 });
        graph.Nodes.Add(new Node { Name = "mm", Op = OpType.MatMul, Inputs = new[] { "x", "w" }, Outputs = new[] { "z" } });
        graph.Options = new ExecutionOptions(OptimizationMode.Speed, TensorExecutionOptions.Auto with { UseKBlockedPanels = true });
        return graph;
    }

    [Fact]
    public void ReplacedInitializer_DropsStaleBlockedClone()
    {
        var graph = BuildGraph();
        graph.RefreshLifetimeAnalysis();
        Assert.True(graph.Initializers.ContainsKey("kbpacked:w"));
        var oldClone = graph.Initializers["kbpacked:w"];
        var w2 = DenseTensor<float>.OfShape(new int[] { 1024, 1024 });
        var ws = w2.Buffer.Span;
        for (int i = 0; i < ws.Length; i++) ws[i] = 0.03f;
        graph.Initializers["w"] = w2;
        graph.RefreshLifetimeAnalysis();
        Assert.True(graph.Initializers.ContainsKey("kbpacked:w"));
        Assert.NotSame(oldClone, graph.Initializers["kbpacked:w"]);
        Assert.Equal(1, graph.PackingReport.KBlockedLive);
    }

    [Fact]
    public void InvalidatePreparation_DropsBlockedClones()
    {
        var graph = BuildGraph();
        graph.RefreshLifetimeAnalysis();
        Assert.True(graph.Initializers.ContainsKey("kbpacked:w"));
        graph.InvalidatePreparation();
        Assert.False(graph.Initializers.ContainsKey("kbpacked:w"));
        Assert.Equal(0, graph.PackingReport.KBlockedLive);
        Assert.Equal(0, graph.PackingReport.KBlockedRetainedBytes);
    }
}
