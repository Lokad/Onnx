namespace Lokad.Onnx.Backend.Tests;

public class PackedPreparationLookupTests
{
    static Node Transpose(string name, string input) => new()
    {
        Name = name, Op = OpType.Transpose, Inputs = [input], Outputs = ["b"]
    };

    static ComputationalGraph Graph(long budget)
    {
        var graph = new ComputationalGraph(budget);
        graph.Metadata["Name"] = "packed-fold-lookup";
        graph.Inputs["x"] = DenseTensor<float>.OfValues(new float[,] { { 1, 2 }, { 3, 4 } });
        graph.Initializers["w"] = DenseTensor<float>.OfValues(new float[,] { { 1, 2 }, { 3, 4 } });
        graph.Nodes.Add(Transpose("transpose", "w"));
        graph.Nodes.Add(new Node { Name = "mm", Op = OpType.MatMul, Inputs = ["x", "b"], Outputs = ["y"] });
        graph.Outputs["y"] = Tensor<float>.Zeros(2, 2);
        return graph;
    }

    [Theory]
    [InlineData(0)]
    [InlineData(16)]
    public void FoldedSource_ReusesReplacesAndInvalidatesWithinBudget(long budget)
    {
        var graph = Graph(budget);
        var feeds = new Dictionary<string, ITensor> { ["x"] = graph.Inputs["x"]! };
        graph.Prepare();
        Assert.Equal(budget, graph.RetainedPackedWeightBytes);
        var original = graph.PackedWeights.Values.SingleOrDefault()?.Packed;
        graph.Prepare();
        Assert.Same(original, graph.PackedWeights.Values.SingleOrDefault()?.Packed);
        Assert.Equal(new float[] { 1, 3, 2, 4 }, ((Tensor<float>)graph.Initializers["folded:transpose"]).ToArray());

        graph.Initializers["w"] = DenseTensor<float>.OfValues(new float[,] { { 10, 20 }, { 30, 40 } });
        graph.Prepare();
        Assert.Equal(budget, graph.RetainedPackedWeightBytes);
        if (budget != 0) Assert.NotSame(original, Assert.Single(graph.PackedWeights).Value.Packed);
        Assert.True(graph.Execute(feeds, true), graph.LastErrorMessage);
        Assert.Equal(new float[] { 50, 110, 110, 250 }, ((Tensor<float>)graph.Outputs["y"]!).ToArray());

        graph.InvalidatePreparation();
        Assert.Empty(graph.PackedWeights);
        Assert.Equal(0, graph.RetainedPackedWeightBytes);
        Assert.True(graph.Execute(feeds, true), graph.LastErrorMessage);
        Assert.Equal(new float[] { 50, 110, 110, 250 }, ((Tensor<float>)graph.Outputs["y"]!).ToArray());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DuplicateTransposeOutputs_KeepFirstMatchAndValidation(bool firstIsConstant)
    {
        var graph = Graph(16);
        graph.Inputs["dynamic"] = Tensor<float>.Zeros(2, 2);
        graph.Nodes[0] = Transpose("first", firstIsConstant ? "w" : "dynamic");
        graph.Nodes.Insert(1, Transpose("second", firstIsConstant ? "dynamic" : "w"));
        graph.Prepare();
        Assert.Equal(firstIsConstant ? 16 : 0, graph.RetainedPackedWeightBytes);
        Assert.Equal(firstIsConstant ? 1 : 0, graph.PackedWeights.Count);
        Assert.False(graph.Execute(new Dictionary<string, ITensor>
        {
            ["x"] = graph.Inputs["x"]!, ["dynamic"] = graph.Inputs["dynamic"]!
        }, true));
        Assert.Contains("duplicate producers", graph.LastErrorMessage);
        Assert.Contains("first", graph.LastErrorMessage);
        Assert.Contains("second", graph.LastErrorMessage);
    }
}
