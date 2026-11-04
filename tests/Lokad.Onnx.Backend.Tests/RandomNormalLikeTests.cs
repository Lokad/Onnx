namespace Lokad.Onnx.Backend.Tests;

public class RandomNormalLikeTests
{
    static ComputationalGraph Graph(float? seed = 17f, int? dtype = null, float? mean = null, float? scale = null)
    {
        var graph = new ComputationalGraph { Opset = new() { [""] = 17 }, Metadata = new() { ["Name"] = "random-test" } };
        graph.Inputs["x"] = new DenseTensor<float>(new[] { 1 });
        graph.Outputs["y"] = dtype == (int)TensorElementType.Double
            ? new DenseTensor<double>(new[] { 1 }) : new DenseTensor<float>(new[] { 1 });
        graph.InputDescs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { -1 } });
        graph.OutputDescs.Add(new OnnxValueInfo { Name = "y", ElementType = dtype.HasValue ? (TensorElementType)dtype : TensorElementType.Float, Dims = new[] { -1 } });
        var attributes = new Dictionary<string, object>();
        if (seed.HasValue) attributes["seed"] = seed.Value;
        if (dtype.HasValue) attributes["dtype"] = dtype.Value;
        if (mean.HasValue) attributes["mean"] = mean.Value;
        if (scale.HasValue) attributes["scale"] = scale.Value;
        graph.Nodes.Add(new Node { Name = "normal", Op = OpType.RandomNormalLike, OpTypeName = "RandomNormalLike", OpsetVersion = 17,
            Inputs = new[] { "x" }, Outputs = new[] { "y" }, Attributes = attributes });
        return graph;
    }

    static float[] Draw(ComputationalGraph graph, int length)
    {
        graph.Reset();
        Assert.True(graph.Execute(new Dictionary<string, ITensor> { ["x"] = new DenseTensor<float>(new[] { length }) }, false), graph.LastErrorMessage);
        return ((Tensor<float>)graph.Outputs["y"]).ToArray();
    }

    [Fact]
    public void FacadeStreamsAdvanceAcrossExecuteResetAndOddLengthBoundaries()
    {
        var graph = Graph();
        var first = Draw(graph, 31);
        var held = first.ToArray();
        var second = Draw(graph, 32);
        Assert.Equal(Draw(Graph(), 63), first.Concat(second));
        Assert.Equal(held, first);
        Assert.False(first.SequenceEqual(second.Take(31)));
    }

    [Fact]
    public void ExplicitExecutionContextsOwnIndependentAdvancingStreams()
    {
        var prepared = Graph();
        var first = prepared.CreateExecution(null);
        var second = prepared.CreateExecution(null);
        Assert.Equal(Draw(first, 16), Draw(second, 16));
        Assert.Equal(Draw(first, 16), Draw(second, 16));
        Assert.Equal(Draw(Graph(), 16), Draw(prepared, 16));
    }

    [Fact]
    public void ChangedSeedRestartsStreamAndUnseededCallsRemainFinite()
    {
        var graph = Graph(1f);
        _ = Draw(graph, 8);
        graph.Nodes[0].Attributes!["seed"] = 7f;
        Assert.Equal(Draw(Graph(7f), 16), Draw(graph, 16));
        var random = Draw(Graph(null), 20000);
        Assert.All(random, value => Assert.True(float.IsFinite(value)));
    }

    [Fact]
    public void NormalDistributionHasExpectedMomentsAndZeroScaleReturnsMean()
    {
        var values = Draw(Graph(123f), 20000);
        double average = values.Average(v => (double)v);
        double variance = values.Average(v => ((double)v - average) * ((double)v - average));
        Assert.InRange(average, -.05, .05);
        Assert.InRange(variance, .9, 1.1);
        var graph = Graph(dtype: (int)TensorElementType.Double, mean: 4f, scale: 0f);
        graph.Reset();
        Assert.True(graph.Execute(new Dictionary<string, ITensor> { ["x"] = new DenseTensor<float>(new[] { 7 }) }, false), graph.LastErrorMessage);
        Assert.Equal(Enumerable.Repeat(4.0, 7), ((Tensor<double>)graph.Outputs["y"]).ToArray());
    }

    [Theory]
    [InlineData("scale", -1f)]
    [InlineData("mean", float.NaN)]
    [InlineData("seed", float.PositiveInfinity)]
    public void InvalidDistributionAttributesFailWithoutDrawing(string name, float value)
    {
        var graph = Graph();
        graph.Nodes[0].Attributes![name] = value;
        var result = graph.Nodes[0].Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, result.Status);
        Assert.Contains(name, result.Message);
    }
}
