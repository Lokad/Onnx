namespace Lokad.Onnx.Backend.Tests;

public class InitializerDefaultTests
{
    static OnnxValueInfo Desc(string name, bool symbolic = true) => new()
    {
        Name = name, ElementType = TensorElementType.Float,
        Dims = [-1], DimParams = [symbolic ? "N" : null]
    };

    static OnnxModel ModelWithDefault() => new()
    {
        Name = "initializer-default", Opset = { [""] = 13 },
        Inputs = { Desc("x") }, Outputs = { Desc("y") },
        Initializers = { new OnnxTensor { Name = "x", ElementType = TensorElementType.Float, Dims = [2], Data = new float[] { 1, 2 } } },
        Nodes = { new OnnxNode { Name = "copy", OpType = "Identity", Inputs = ["x"], Outputs = ["y"] } }
    };

    static object Feed(bool positional, params (string Name, ITensor Value)[] values) => positional
        ? values.Select(value => value.Value).ToArray()
        : values.ToDictionary(value => value.Name, value => value.Value);

    static Tensor<float> Values(ComputationalGraph graph) => (Tensor<float>)graph.Outputs["y"];

    [Theory]
    [InlineData(false, false, false)] [InlineData(false, false, true)]
    [InlineData(false, true, false)] [InlineData(false, true, true)]
    [InlineData(true, false, false)] [InlineData(true, false, true)]
    [InlineData(true, true, false)] [InlineData(true, true, true)]
    public void DefaultDoesNotDependOnPreviousRequests(bool positional, bool context, bool symbolic)
    {
        var model = ModelWithDefault();
        model.Inputs[0].DimParams = [symbolic ? "N" : null];
        model.Outputs[0].DimParams = [symbolic ? "N" : null];
        var plan = Model.Load(model);
        ComputationalGraph graph = context ? plan.CreateExecution(ExecutionOptions.Scalar) : plan;
        var initializer = graph.Initializers["x"];
        Assert.Empty(graph.GetRequiredInputs(true));
        Assert.True(graph.Execute(Feed(positional), true), graph.LastErrorMessage);
        var heldDefault = Values(graph);
        Assert.Equal(new float[] { 1, 2 }, heldDefault.ToArray());
        Assert.NotSame(initializer, heldDefault);

        foreach (int length in new[] { 2, 3 })
        {
            var input = DenseTensor<float>.OfValues(Enumerable.Range(8, length).Select(value => (float)value).ToArray());
            var saved = input.ToArray();
            Assert.True(graph.Execute(Feed(positional, ("x", input)), false), graph.LastErrorMessage);
            var heldExplicit = Values(graph);
            Assert.Equal(saved, heldExplicit.ToArray());
            Assert.True(graph.Execute(Feed(positional), true), graph.LastErrorMessage);
            Assert.Equal(new float[] { 1, 2 }, Values(graph).ToArray());
            Assert.Equal(saved, input.ToArray());
            Assert.Equal(saved, heldExplicit.ToArray());
            graph.Reset();
            Assert.True(graph.Execute(Feed(positional), true), graph.LastErrorMessage);
        }
        Assert.False(graph.Execute(Feed(positional, ("x", DenseTensor<int>.OfValues(new[] { 5, 6 }))), false));
        Assert.True(graph.Execute(Feed(positional), true), graph.LastErrorMessage);
        Assert.Equal(new float[] { 1, 2 }, heldDefault.ToArray());
        Assert.Equal(new float[] { 1, 2 }, ((Tensor<float>)initializer).ToArray());
        Assert.Same(initializer, graph.Initializers["x"]);
        Assert.Equal(new[] { -1 }, graph.InputDescs[0].Dims);
        Assert.Equal(symbolic ? "N" : null, graph.InputDescs[0].DimParams![0]);
    }

    static bool Resolve(ComputationalGraph graph, object inputs, bool useInitializers) => inputs is ITensor[] positional
        ? graph.ResolveInputs(positional, useInitializers)
        : graph.ResolveInputs((Dictionary<string, ITensor>)inputs, useInitializers);

    [Theory]
    [InlineData(false)] [InlineData(true)]
    public void DefaultsParticipateInValidationBeforeAnyBindingChanges(bool positional)
    {
        var model = ModelWithDefault();
        model.Inputs.Add(Desc("b"));
        var graph = Model.Load(model);
        var a = DenseTensor<float>.OfValues(new float[] { 20, 21 });
        var b = DenseTensor<float>.OfValues(new float[] { 8, 9 });
        Assert.True(Resolve(graph, Feed(positional, ("x", a), ("b", b)), false));
        Assert.False(Resolve(graph, Feed(positional, ("b", DenseTensor<float>.OfShape(3))), true));
        Assert.Contains("Shared symbolic dimension N", graph.LastErrorMessage);
        Assert.Same(a, graph.Inputs["x"]);
        Assert.Same(b, graph.Inputs["b"]);
        Assert.True(Resolve(graph, Feed(positional, ("b", b)), true), graph.LastErrorMessage);
        Assert.Same(graph.Initializers["x"], graph.Inputs["x"]);
    }

    [Theory]
    [InlineData(false)] [InlineData(true)]
    public void ConflictingDefaultsAreRejectedTogether(bool positional)
    {
        var model = ModelWithDefault();
        model.Inputs.Add(Desc("b"));
        model.Initializers.Add(new OnnxTensor { Name = "b", ElementType = TensorElementType.Float, Dims = [3], Data = new float[3] });
        var graph = Model.Load(model);
        Assert.False(Resolve(graph, Feed(positional), true));
        Assert.Contains("Shared symbolic dimension N", graph.LastErrorMessage);
        Assert.All(graph.Inputs.Values, value => Assert.Null(value));
    }

    [Theory]
    [InlineData(false)] [InlineData(true)]
    public void InvalidDefaultStillRequiresAnExplicitValidInput(bool wrongType)
    {
        var model = ModelWithDefault();
        model.Inputs[0].Dims = [2]; model.Inputs[0].DimParams = [null];
        model.Outputs[0].Dims = [2]; model.Outputs[0].DimParams = [null];
        model.Initializers[0] = wrongType
            ? new OnnxTensor { Name = "x", ElementType = TensorElementType.Int32, Dims = [2], Data = new int[2] }
            : new OnnxTensor { Name = "x", ElementType = TensorElementType.Float, Dims = [3], Data = new float[3] };
        var graph = Model.Load(model);
        Assert.False(graph.Execute(new Dictionary<string, ITensor>(), true));
        Assert.True(graph.Execute(new Dictionary<string, ITensor> { ["x"] = DenseTensor<float>.OfValues(new float[] { 8, 9 }) }, true), graph.LastErrorMessage);
        Assert.Equal(new float[] { 8, 9 }, Values(graph).ToArray());
    }
}
