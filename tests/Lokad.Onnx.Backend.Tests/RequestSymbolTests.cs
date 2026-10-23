namespace Lokad.Onnx.Backend.Tests;

public class RequestSymbolTests
{
    static OnnxValueInfo Desc(string name, string? symbol) => new()
    {
        Name = name, ElementType = TensorElementType.Float, Dims = [-1], DimParams = [symbol]
    };

    static ComputationalGraph Graph(bool duplicate, string? outputSymbol = "N", bool withDefault = false)
    {
        var model = new OnnxModel
        {
            Name = "request-symbols", Opset = { [""] = 13 },
            Inputs = { Desc("x", "N") }, Outputs = { Desc("y", outputSymbol) },
            Nodes = { new OnnxNode
            {
                Name = "copy", OpType = duplicate ? "Concat" : "Identity",
                Inputs = duplicate ? ["x", "x"] : ["x"], Outputs = ["y"],
                Attributes = duplicate ? new() { ["axis"] = 0 } : new()
            } }
        };
        if (withDefault) model.Initializers.Add(new OnnxTensor
        {
            Name = "x", ElementType = TensorElementType.Float, Dims = [2], Data = new float[] { 1, 2 }
        });
        return Model.Load(model);
    }

    static object Feed(bool positional, params ITensor[] tensors) => positional ? tensors
        : tensors.Length == 0 ? new Dictionary<string, ITensor>() : new Dictionary<string, ITensor> { ["x"] = tensors[0] };

    [Theory]
    [InlineData(false, false)] [InlineData(false, true)]
    [InlineData(true, false)] [InlineData(true, true)]
    public void SharedInputOutputSymbolRejectsConflictAndRecovers(bool positional, bool context)
    {
        var plan = Graph(true);
        ComputationalGraph graph = context ? plan.CreateExecution(ExecutionOptions.Memory) : plan;
        var empty = DenseTensor<float>.OfShape(0);
        Assert.True(graph.Execute(Feed(positional, empty), false), graph.LastErrorMessage);
        var held = (Tensor<float>)graph.Outputs["y"];
        var input = DenseTensor<float>.OfValues([1, 2]);
        for (int retry = 0; retry < 2; retry++)
        {
            Assert.False(graph.Execute(Feed(positional, input), false));
            Assert.Contains("Shared symbolic dimension N", graph.LastErrorMessage);
            Assert.Contains("Graph output y", graph.LastErrorMessage);
            Assert.Empty(graph.Outputs);
            Assert.True(graph.Execute(Feed(positional, empty), false), graph.LastErrorMessage);
            Assert.Null(graph.LastErrorMessage);
            graph.Reset();
        }
        Assert.Empty(held.ToArray());
        Assert.Equal(new float[] { 1, 2 }, input.ToArray());
    }

    [Theory]
    [InlineData(false, false)] [InlineData(false, true)]
    [InlineData(true, false)] [InlineData(true, true)]
    public void SelectedDefaultsAlsoConstrainOutputSymbols(bool positional, bool context)
    {
        var plan = Graph(true, withDefault: true);
        ComputationalGraph graph = context ? plan.CreateExecution(ExecutionOptions.Scalar) : plan;
        Assert.False(graph.Execute(Feed(positional), true));
        Assert.Contains("Shared symbolic dimension N", graph.LastErrorMessage);
        Assert.Empty(graph.Outputs);
        Assert.Equal(new float[] { 1, 2 }, ((Tensor<float>)graph.Initializers["x"]).ToArray());
    }

    [Theory]
    [InlineData(false, false)] [InlineData(false, true)]
    [InlineData(true, false)] [InlineData(true, true)]
    public void MatchingIndependentAndAnonymousSymbolsAllowChangingSizes(bool positional, bool context)
    {
        foreach (var (duplicate, symbol) in new (bool, string?)[] { (false, "N"), (true, "M"), (true, null) })
        {
            var plan = Graph(duplicate, symbol);
            ComputationalGraph graph = context ? plan.CreateExecution(ExecutionOptions.Scalar) : plan;
            foreach (int length in new[] { 2, 0, 3 })
            {
                var input = DenseTensor<float>.OfValues(Enumerable.Range(1, length).Select(i => (float)i).ToArray());
                Assert.True(graph.Execute(Feed(positional, input), false), graph.LastErrorMessage);
                Assert.Equal(duplicate ? input.ToArray().Concat(input.ToArray()) : input.ToArray(), ((Tensor<float>)graph.Outputs["y"]).ToArray());
            }
        }
    }

    [Fact]
    public async Task ConcurrentContextsKeepIndependentRequestSymbols()
    {
        var plan = Graph(false);
        await Task.WhenAll(Enumerable.Range(1, 8).Select(length => Task.Run(() =>
        {
            var context = plan.CreateExecution(ExecutionOptions.Memory);
            for (int retry = 0; retry < 4; retry++)
            {
                var input = DenseTensor<float>.OfShape(length + retry);
                Assert.True(context.Execute(new ITensor[] { input }, false), context.LastErrorMessage);
                Assert.Equal(new[] { length + retry }, context.Outputs["y"].Dims);
            }
        })));
    }
}
