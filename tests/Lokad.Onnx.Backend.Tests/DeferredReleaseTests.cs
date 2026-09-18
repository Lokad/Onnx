namespace Lokad.Onnx.Backend.Tests;

public class DeferredReleaseTests
{
    static ComputationalGraph SharedViews(bool keepFirst, bool reverse)
    {
        var graph = new ComputationalGraph();
        graph.Metadata["Name"] = "deferred-release-alias-changes";
        graph.Inputs["x"] = DenseTensor<float>.OfShape(4);
        graph.Initializers["w"] = DenseTensor<float>.OfValues(new float[] { 10, 20, 30, 40 });
        graph.Initializers["shape2"] = DenseTensor<long>.OfValues(new long[] { 2, 4 });
        graph.Initializers["shape3"] = DenseTensor<long>.OfValues(new long[] { 3, 4 });
        graph.Outputs["y2"] = DenseTensor<float>.OfShape(2, 4);
        graph.Outputs["y3"] = DenseTensor<float>.OfShape(3, 4);
        graph.Outputs["z"] = DenseTensor<float>.OfShape(4);
        if (keepFirst) graph.Outputs["v2"] = DenseTensor<float>.OfShape(2, 4);
        void Add(string name, OpType op, string[] inputs, string output)
        {
            graph.Nodes.Add(new Node { Name = name, Op = op, Inputs = inputs, Outputs = new[] { output } });
            if (!graph.Outputs.ContainsKey(output)) graph.IntermediateOutputs[output] = null;
        }
        Add("source", OpType.Add, new[] { "x", "w" }, "t");
        Add("first-view", OpType.Expand, new[] { "t", "shape2" }, "v2");
        Add("second-view", OpType.Expand, new[] { "t", "shape3" }, "v3");
        // These writes use the same pool size as t, but different values. Returning
        // t before either remaining view dies would corrupt a later output.
        for (int i = 0; i < 12; i++) Add("unrelated" + i, OpType.Add, new[] { "x", "x" }, "dead" + i);
        string first = reverse ? "3" : "2", second = reverse ? "2" : "3";
        Add("consume-first", OpType.Relu, new[] { "v" + first }, "y" + first);
        for (int i = 12; i < 24; i++) Add("unrelated" + i, OpType.Add, new[] { "x", "x" }, "dead" + i);
        Add("consume-second", OpType.Relu, new[] { "v" + second }, "y" + second);
        Add("reuse", OpType.Add, new[] { "x", "x" }, "z");
        graph.RefreshLifetimeAnalysis();
        return graph;
    }

    [Theory]
    [InlineData(false, false, false)]
    [InlineData(false, true, false)]
    [InlineData(true, false, false)]
    [InlineData(true, true, false)]
    [InlineData(false, false, true)]
    [InlineData(false, true, true)]
    [InlineData(true, false, true)]
    [InlineData(true, true, true)]
    public void MultipleAliasesDropInEitherOrderAndOutputsSurviveLaterRuns(bool keepFirst, bool reverse, bool reuseContext)
    {
        var plan = SharedViews(keepFirst, reverse);
        ComputationalGraph graph = reuseContext ? plan.CreateExecution(null) : plan;
        var retained = new List<(Tensor<float> Tensor, float[] Values)>();
        for (int run = 0; run < 3; run++)
        {
            graph.Reset();
            var input = DenseTensor<float>.OfValues(new float[] { 1 + run, 2 + run, 3 + run, 4 + run });
            var values = input.ToArray();
            var inputs = new Dictionary<string, ITensor> { ["x"] = input };
            Assert.True(graph.Execute(inputs, true), graph.LastErrorMessage);
            var sum = new float[] { 11 + run, 22 + run, 33 + run, 44 + run };
            Assert.Equal(Enumerable.Range(0, 2).SelectMany(_ => sum), ((Tensor<float>)graph.Outputs["y2"]!).ToArray());
            Assert.Equal(Enumerable.Range(0, 3).SelectMany(_ => sum), ((Tensor<float>)graph.Outputs["y3"]!).ToArray());
            Assert.Equal(values.Select(x => x * 2), ((Tensor<float>)graph.Outputs["z"]!).ToArray());
            Assert.Equal(values, input.ToArray());
            if (keepFirst)
            {
                Assert.NotNull(graph.IntermediateOutputs["t"]);
                Assert.Equal(Enumerable.Range(0, 2).SelectMany(_ => sum), ((Tensor<float>)graph.Outputs["v2"]!).ToArray());
            }
            else Assert.Null(graph.IntermediateOutputs["t"]);
            Assert.Null(graph.IntermediateOutputs["v3"]);
            Assert.True(graph.LastPoolReused >= 23);
            foreach (var old in retained) Assert.Equal(old.Values, old.Tensor.ToArray());
            foreach (var output in graph.Outputs.Values.Cast<Tensor<float>>()) retained.Add((output, output.ToArray()));
            // A rejected run must not leave release-cache state for the next call.
            graph.Reset();
            Assert.False(graph.Execute(new Dictionary<string, ITensor>(), false));
            foreach (var old in retained) Assert.Equal(old.Values, old.Tensor.ToArray());
        }
    }
}
