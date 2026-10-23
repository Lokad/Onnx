namespace Lokad.Onnx.Backend.Tests;

public class NestedSymbolTests
{
    static OnnxValueInfo Desc(string name, string? symbol) => new()
    {
        Name = name, ElementType = TensorElementType.Float, Dims = [-1], DimParams = [symbol]
    };

    static OnnxNode If(OnnxSubgraph then, OnnxSubgraph otherwise, string output) => new()
    {
        Name = "choose-" + output, OpType = "If", Inputs = ["cond"], Outputs = [output],
        Attributes = new() { ["then_branch"] = then, ["else_branch"] = otherwise }
    };

    static OnnxSubgraph Branch(bool duplicate, string? symbol, int depth)
    {
        string output = "branch-" + depth;
        var branch = new OnnxSubgraph { Name = output, Outputs = { Desc(output, symbol) } };
        if (depth > 1)
            branch.Nodes.Add(If(Branch(duplicate, symbol, depth - 1), Branch(duplicate, symbol, depth - 1), output));
        else
            branch.Nodes.Add(new OnnxNode
            {
                Name = "produce", OpType = duplicate ? "Concat" : "Identity",
                Inputs = duplicate ? ["x", "x"] : ["x"], Outputs = [output],
                Attributes = duplicate ? new() { ["axis"] = 0 } : new()
            });
        return branch;
    }

    static OnnxModel Description(bool duplicate, string? symbol, int depth) => new()
    {
        Name = "nested-symbols", Opset = { [""] = 14 },
        Inputs = { new OnnxValueInfo { Name = "cond", ElementType = TensorElementType.Bool }, Desc("x", "N") },
        Outputs = { Desc("y", null) },
        Nodes = { If(Branch(duplicate, symbol, depth), Branch(duplicate, symbol, depth), "y") }
    };

    static object Feed(bool positional, bool condition, Tensor<float>? value) => positional
        ? value is null ? new ITensor[] { DenseTensor<bool>.Scalar(condition) } : new ITensor[] { DenseTensor<bool>.Scalar(condition), value }
        : value is null ? new Dictionary<string, ITensor> { ["cond"] = DenseTensor<bool>.Scalar(condition) }
        : new Dictionary<string, ITensor> { ["cond"] = DenseTensor<bool>.Scalar(condition), ["x"] = value };

    public static IEnumerable<object[]> Cases =>
        from positional in new[] { false, true }
        from context in new[] { false, true }
        from condition in new[] { false, true }
        from depth in new[] { 1, 2 }
        select new object[] { positional, context, condition, depth };

    [Theory]
    [MemberData(nameof(Cases))]
    public void SelectedBranchCannotContradictParentSymbolAndRetriesStartFresh(bool positional, bool context, bool condition, int depth)
    {
        var plan = Model.Load(Description(true, "N", depth));
        ComputationalGraph graph = context ? plan.CreateExecution(ExecutionOptions.Memory) : plan;
        var empty = DenseTensor<float>.OfShape(0);
        Assert.True(graph.Execute(Feed(positional, condition, empty), false), graph.LastErrorMessage);
        var held = (Tensor<float>)graph.Outputs["y"];
        var input = DenseTensor<float>.OfValues([1, 2]);
        Assert.False(graph.Execute(Feed(positional, condition, input), false));
        Assert.Contains("Shared symbolic dimension N", graph.LastErrorMessage);
        Assert.Empty(graph.Outputs);
        Assert.True(graph.Execute(Feed(positional, !condition, empty), false), graph.LastErrorMessage);
        Assert.Null(graph.LastErrorMessage);
        graph.Reset();
        Assert.Empty(held.ToArray());
        Assert.Equal(new float[] { 1, 2 }, input.ToArray());
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void MatchingIndependentAndAnonymousSymbolsKeepWorking(bool positional, bool context, bool condition, int depth)
    {
        foreach (var (duplicate, symbol) in new (bool, string?)[] { (false, "N"), (true, "M"), (true, null) })
        {
            var plan = Model.Load(Description(duplicate, symbol, depth));
            ComputationalGraph graph = context ? plan.CreateExecution(ExecutionOptions.Scalar) : plan;
            foreach (int length in new[] { 2, 0, 3 })
            {
                var input = DenseTensor<float>.OfShape(length);
                Assert.True(graph.Execute(Feed(positional, condition, input), false), graph.LastErrorMessage);
                Assert.Equal(duplicate ? length * 2 : length, graph.Outputs["y"].Length);
            }
        }
    }

    [Theory]
    [InlineData(false)] [InlineData(true)]
    public void ChildCreatedSymbolConstrainsLaterParentOutput(bool context)
    {
        var model = Description(true, "M", 2);
        model.Outputs.Add(Desc("z", "M"));
        model.Nodes.Add(new OnnxNode { Name = "after", OpType = "Identity", Inputs = ["x"], Outputs = ["z"] });
        var plan = Model.Load(model);
        ComputationalGraph graph = context ? plan.CreateExecution(ExecutionOptions.Scalar) : plan;
        Assert.False(graph.Execute(Feed(false, true, DenseTensor<float>.OfShape(2)), false));
        Assert.Contains("Shared symbolic dimension M", graph.LastErrorMessage);
        Assert.Empty(graph.Outputs);
    }

    [Theory]
    [InlineData(false)] [InlineData(true)]
    public void DefaultsConstrainSelectedBranches(bool positional)
    {
        var model = Description(true, "N", 1);
        model.Initializers.Add(new OnnxTensor { Name = "x", ElementType = TensorElementType.Float, Dims = [2], Data = new float[] { 1, 2 } });
        var graph = Model.Load(model);
        Assert.False(graph.Execute(Feed(positional, true, null), true));
        Assert.Contains("Shared symbolic dimension N", graph.LastErrorMessage);
        Assert.Equal(new float[] { 1, 2 }, ((Tensor<float>)graph.Initializers["x"]).ToArray());
    }

    [Fact]
    public void InactiveBranchDoesNotBindDimensions()
    {
        var model = Description(false, "M", 1);
        model.Nodes[0].Attributes["else_branch"] = Branch(true, "M", 1);
        model.Outputs[0] = Desc("y", "M");
        var graph = Model.Load(model);
        foreach (bool condition in new[] { true, false, true })
        {
            Assert.True(graph.Execute(Feed(false, condition, DenseTensor<float>.OfShape(2)), false), graph.LastErrorMessage);
            Assert.Equal(condition ? 2 : 4, graph.Outputs["y"].Length);
        }
    }

    [Fact]
    public async Task ConcurrentContextsDoNotShareRequestSymbols()
    {
        var plan = Model.Load(Description(false, "N", 2));
        await Task.WhenAll(Enumerable.Range(1, 8).Select(length => Task.Run(() =>
        {
            var context = plan.CreateExecution(ExecutionOptions.Memory);
            for (int retry = 0; retry < 3; retry++)
                Assert.True(context.Execute(Feed(false, retry % 2 == 0, DenseTensor<float>.OfShape(length + retry)), false), context.LastErrorMessage);
        })));
    }
}
