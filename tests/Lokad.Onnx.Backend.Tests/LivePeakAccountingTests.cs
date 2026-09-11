namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Live payload peak accounting (P28): bind and release deltas plus a per-run
/// recompute report bit-identical peaks to the former per-node full scan.
/// </summary>
public class LivePeakAccountingTests
{
    static ComputationalGraph Chain()
    {
        var graph = new ComputationalGraph { Opset = new System.Collections.Generic.Dictionary<string, int> { [""] = 13 } };
        graph.Metadata["Name"] = "test";
        graph.Inputs["x"] = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f } });
        graph.Inputs["y"] = DenseTensor<float>.OfValues(new float[,] { { 5f, 6f }, { 7f, 8f } });
        graph.Inputs["w"] = DenseTensor<float>.OfValues(new float[,] { { 1f, 1f }, { 1f, 1f } });
        graph.Outputs["z"] = DenseTensor<float>.OfShape(2, 2);
        graph.Nodes.Add(new Node { Name = "add1", Op = OpType.Add, Inputs = new[] { "x", "y" }, Outputs = new[] { "t" } });
        graph.Nodes.Add(new Node { Name = "add2", Op = OpType.Add, Inputs = new[] { "t", "w" }, Outputs = new[] { "z" } });
        graph.IntermediateOutputs["t"] = null;
        graph.RefreshLifetimeAnalysis();
        return graph;
    }

    static System.Collections.Generic.Dictionary<string, ITensor> Feed(ComputationalGraph graph)
    {
        return new System.Collections.Generic.Dictionary<string, ITensor>
        {
            { "x", graph.Inputs["x"] },
            { "y", graph.Inputs["y"] },
            { "w", graph.Inputs["w"] },
        };
    }

    static ExecutionOptions Speed1() => new ExecutionOptions(OptimizationMode.Speed, TensorExecutionOptions.Auto with { MaxDegreeOfParallelism = 1 });

    // Peak 80 on the first run counts the caller-seeded output placeholder alongside
    // live tensors; after Reset the placeholder is gone so the rerun peaks at 64.
    // Both values were verified identical against the former per-node full scan.
    [Fact]
    public void LivePeak_IsExactAndDeterministic()
    {
        var graph = Chain();
        var opts = Speed1();
        Assert.True(graph.Execute(Feed(graph), true, ExecutionProvider.CPU, opts), graph.LastErrorMessage);
        Assert.Equal(new float[] { 7f, 9f, 11f, 13f }, ((Tensor<float>)graph.Outputs["z"]).ToArray());
        Assert.Equal(80L, graph.LastPeakLiveBytes);
        graph.Reset();
        Assert.True(graph.Execute(Feed(graph), true, ExecutionProvider.CPU, opts), graph.LastErrorMessage);
        Assert.Equal(new float[] { 7f, 9f, 11f, 13f }, ((Tensor<float>)graph.Outputs["z"]).ToArray());
        Assert.Equal(64L, graph.LastPeakLiveBytes);
    }
}
