using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx.Backend.Tests;

public partial class ConvBlockedSpatialTests
{
    static float[] Values(int count) => Values(count, 0);

    static float[] Values(int count, int seed) => Enumerable.Range(0, count).Select(i => ((i * 17 + seed) % 47 - 23) / 127f).ToArray();

    static ComputationalGraph Graph() => Graph(16, 32, 3, 11, 1, long.MaxValue, false);
    static ComputationalGraph Graph(long budget) => Graph(16, 32, 3, 11, 1, budget, false);
    static ComputationalGraph Graph(int m, long budget) => Graph(16, m, 3, 11, 1, budget, false);
    static ComputationalGraph Graph(bool relu) => Graph(16, 32, 3, 11, 1, long.MaxValue, relu);
    static ComputationalGraph Graph(int c, int stride) => Graph(c, 32, 3, 11, stride, long.MaxValue, false);

    static ComputationalGraph Graph(int c, int m, int h, int w, int stride, long budget, bool relu)
    {
        var graph = new ComputationalGraph(budget); graph.Metadata["Name"] = "blocked-convolution";
        graph.Opset[""] = 18;
        graph.Inputs["x"] = new DenseTensor<float>(Values(c * h * w), new[] { 1, c, h, w });
        graph.Initializers["w"] = new DenseTensor<float>(Values(m * c * 9, 3), new[] { m, c, 3, 3 });
        graph.Initializers["b"] = new DenseTensor<float>(Values(m, 11), new[] { m });
        graph.Outputs["y"] = Tensor<float>.Zeros(1, m, (h + stride - 1) / stride, (w + stride - 1) / stride);
        graph.Nodes.Add(new Node { Name = "conv", Op = relu ? OpType.ConvRelu : OpType.Conv,
            OpTypeName = relu ? "ConvRelu" : "Conv", Domain = "", OpsetVersion = 18, IsFused = relu,
            Inputs = new[] { "x", "w", "b" }, Outputs = new[] { "y" },
            Attributes = new() { ["pads"] = new[] { 1, 1, 1, 1 }, ["strides"] = new[] { stride, stride },
                ["dilations"] = new[] { 1, 1 }, ["kernel_shape"] = new[] { 3, 3 }, ["group"] = 1 } });
        return graph;
    }

    static DenseTensor<float> Operand(ComputationalGraph graph, string name) =>
        Assert.IsType<DenseTensor<float>>(name == "x" ? graph.Inputs[name] : graph.Initializers[name]);

    static float[] Reference(ComputationalGraph graph) => Reference(graph, null, null);
    static float[] Reference(ComputationalGraph graph, TensorExecutionOptions options) => Reference(graph, options, null);
    static float[] Reference(ComputationalGraph graph, DenseTensor<float> weight) => Reference(graph, null, weight);

    static float[] Reference(ComputationalGraph graph, TensorExecutionOptions? options, DenseTensor<float>? weight)
    {
        int s = graph.Nodes[0].Ints("strides")![0];
        var result = Lokad.Onnx.Tensor<float>.Conv2D(Operand(graph, "x"), weight ?? Operand(graph, "w"), graph.Nodes[0].GetInt("group", 1)!.Value,
            graph.Nodes[0].Ints("pads")!, Operand(graph, "b"), graph.Nodes[0].Ints("kernel_shape"), new[] { s, s },
            graph.Nodes[0].Ints("dilations"), options ?? TensorExecutionOptions.Auto);
        if (graph.Nodes[0].Op == OpType.ConvRelu) result = Lokad.Onnx.Tensor<float>.Relu(result, options ?? TensorExecutionOptions.Auto);
        return result.ToArray();
    }

    static DenseTensor<float> Run(ComputationalGraph graph) => Run(graph, ExecutionOptions.Default);

    static DenseTensor<float> Run(ComputationalGraph graph, ExecutionOptions options)
    {
        Assert.True(graph.Execute(new Dictionary<string, ITensor> { ["x"] = graph.Inputs["x"] }, true, ExecutionProvider.CPU, options), graph.LastErrorMessage);
        return Assert.IsType<DenseTensor<float>>(graph.Outputs["y"]);
    }

    static void Equal(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            if (float.IsNaN(expected[i])) Assert.True(float.IsNaN(actual[i]));
            else Assert.Equal(BitConverter.SingleToInt32Bits(expected[i]), BitConverter.SingleToInt32Bits(actual[i]));
    }

    static long Bytes() => Bytes(16, 32);

    static long Bytes(int c, int m)
    {
        int lanes = GraphConvPacking.Lanes;
        return lanes == 0 ? 0 : (long)((m + 2 * lanes - 1) / (2 * lanes) * 2 * lanes) * c * 9 * 4;
    }

    [Theory]
    [InlineData(16, 32, 1, 1, 1, false)]
    [InlineData(16, 48, 3, 7, 2, true)]
    [InlineData(32, 48, 7, 13, 1, false)]
    [InlineData(32, 64, 7, 11, 2, true)]
    [InlineData(64, 64, 3, 13, 1, true)]
    [InlineData(128, 128, 3, 7, 1, false)]
    [InlineData(256, 256, 3, 13, 1, true)]
    public void ActualGraphUsesPreparedConvolutionAndPreservesValues(int c, int m, int h, int w, int stride, bool relu)
    {
        var graph = Graph(c, m, h, w, stride, long.MaxValue, relu);
        var x = Operand(graph, "x").ToArray(); var weights = Operand(graph, "w").ToArray(); var expected = Reference(graph);
        var first = Run(graph); var held = first.ToArray();
        if (stride == 1 && GraphConvPacking.Lanes != 0) Close(expected, held); else Equal(expected, held);
        if (GraphConvPacking.Lanes != 0)
        {
            Assert.Single(graph.PackedConvWeights); Assert.Equal(Bytes(c, m) + (stride == 1 ? WinogradBytes(c, m) : 0), graph.RetainedPackedWeightBytes);
            Assert.True(Lokad.Onnx.Tensor<float>.PlanConvBlockedScratch(c, m, h, w, (h + stride - 1) / stride, (w + stride - 1) / stride, out int a, out int b));
            Assert.Equal(stride == 1 ? WinogradScratch(c, m, h, w) : (long)(a + b) * 4, graph.LastScratchBytes);
        }
        else Assert.Empty(graph.PackedConvWeights);
        var again = Run(graph); Assert.NotSame(first, again); Equal(held, first.ToArray()); Equal(held, again.ToArray());
        Equal(x, Operand(graph, "x").ToArray()); Equal(weights, Operand(graph, "w").ToArray());
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(1)]
    public void SharedBudgetIncludesMatrixAndRoundedConvolution(int delta)
    {
        long bytes = Bytes(16, 48); var graph = Graph(m: 48, budget: Math.Max(16, bytes + 16 + delta));
        graph.Inputs["mx"] = DenseTensor<float>.OfValues(new float[,] { { 1, 2 } });
        graph.Initializers["mw"] = DenseTensor<float>.OfValues(new float[,] { { 1, 2 }, { 3, 4 } });
        graph.Outputs["my"] = Lokad.Onnx.Tensor<float>.Zeros(1, 2);
        graph.Nodes.Add(new Node { Name = "mm", Op = OpType.MatMul, OpTypeName = "MatMul", Domain = "", OpsetVersion = 18,
            Inputs = new[] { "mx", "mw" }, Outputs = new[] { "my" } });
        graph.Prepare(); long expected = 16 + (bytes > 0 && delta >= 0 ? bytes : 0);
        Assert.Equal(expected, graph.RetainedPackedWeightBytes); Assert.True(expected <= graph.MaximumPackedWeightBytes);
        var old = graph.PackedConvWeights.Values.SingleOrDefault(); graph.Prepare();
        Assert.Equal(expected, graph.RetainedPackedWeightBytes); Assert.Same(old, graph.PackedConvWeights.Values.SingleOrDefault());
        graph.InvalidatePreparation(); Assert.Empty(graph.PackedConvWeights); Assert.Equal(0, graph.RetainedPackedWeightBytes);
    }

    [Fact]
    public void ZeroBudgetFallsBackWithoutChangingValues()
    {
        var graph = Graph(budget: 0); var expected = Reference(graph); Equal(expected, Run(graph).ToArray());
        Assert.Empty(graph.PackedConvWeights); Assert.Equal(0, graph.RetainedPackedWeightBytes);
    }

    [Fact]
    public void OversizedFirstCandidateDoesNotHideSmallerLaterCandidate()
    {
        var graph = Graph(m: 48, budget: Bytes());
        graph.Initializers["small"] = new DenseTensor<float>(Values(32 * 16 * 9), new[] { 32, 16, 3, 3 });
        var node = graph.Nodes[0]; node.Name = "small"; node.Inputs = new[] { "x", "small" }; node.Outputs = new[] { "small-y" };
        graph.Nodes.Add(node); graph.Prepare();
        if (GraphConvPacking.Lanes != 0) Assert.Equal("small", Assert.Single(graph.PackedConvWeights).Value.SourceName);
        Assert.Equal(Bytes(), graph.RetainedPackedWeightBytes);
    }

    [Fact]
    public void ReplacementAndExplicitMutationInvalidationRefreshValues()
    {
        var graph = Graph(); Run(graph); var old = graph.PackedConvWeights.Values.SingleOrDefault();
        var replaced = new DenseTensor<float>(Values(32 * 16 * 9, 31), new[] { 32, 16, 3, 3 });
        graph.Initializers["w"] = replaced;
        Assert.Null(GraphConvPacking.Resolve(graph.PackedConvWeights, replaced, GraphConvPacking.Lanes));
        Equal(Reference(graph), Run(graph).ToArray());
        // Replacement is safe through fallback; explicit preparation rebuilds the clone.
        graph.Prepare();
        if (old is not null) Assert.NotSame(old, Assert.Single(graph.PackedConvWeights).Value);
        Close(Reference(graph), Run(graph).ToArray());
        replaced.Buffer.Span[17] += .25f; graph.InvalidatePreparation();
        Close(Reference(graph), Run(graph).ToArray());
    }

    [Fact]
    public void SameArrayAliasesUseOneCloneAndForeignTensorFallsBack()
    {
        var graph = Graph(); var source = Operand(graph, "w");
        var alias = new DenseTensor<float>(source.Buffer, source.Dimensions.ToArray()); graph.Initializers["alias"] = alias;
        var node = graph.Nodes[0]; node.Name = "alias"; node.Inputs = new[] { "x", "alias", "b" }; node.Outputs = new[] { "alias-y" }; graph.Nodes.Add(node);
        graph.Prepare();
        if (GraphConvPacking.Lanes != 0)
        {
            Assert.Single(graph.PackedConvWeights); Assert.Equal(Bytes() + WinogradBytes(16, 32), graph.RetainedPackedWeightBytes);
            Assert.Null(GraphConvPacking.Resolve(graph.PackedConvWeights, alias, GraphConvPacking.Lanes));
        }
        Close(Reference(graph), Run(graph).ToArray());
    }

    [Fact]
    public void RuntimeWeightInputIsNeverPrepared()
    {
        var graph = Graph(); graph.Inputs["w"] = graph.Initializers["w"];
        var overrideWeight = new DenseTensor<float>(Values(32 * 16 * 9, 39), new[] { 32, 16, 3, 3 });
        var expected = Reference(graph, weight: overrideWeight);
        Assert.True(graph.Execute(new Dictionary<string, ITensor> { ["x"] = graph.Inputs["x"], ["w"] = overrideWeight },
            false, ExecutionProvider.CPU, ExecutionOptions.Default), graph.LastErrorMessage);
        Equal(expected, Assert.IsType<DenseTensor<float>>(graph.Outputs["y"]).ToArray());
        Assert.Empty(graph.PackedConvWeights);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ScalarAndSimdPoliciesKeepTheirOriginalArithmetic(bool simd)
    {
        var graph = Graph(); var options = simd ? ExecutionOptions.Simd : ExecutionOptions.Scalar;
        Equal(Reference(graph, options.Tensor), Run(graph, options).ToArray());
    }

    [Theory]
    [InlineData("x", false)]
    [InlineData("w", false)]
    [InlineData("b", false)]
    [InlineData("x", true)]
    [InlineData("w", true)]
    [InlineData("b", true)]
    public void NonfiniteOperandsRetainSelectedFallback(string operand, bool relu)
    {
        var graph = Graph(relu: relu); Operand(graph, operand).Buffer.Span[0] = BitConverter.Int32BitsToSingle(unchecked((int)0xffc00123));
        Equal(Reference(graph), Run(graph).ToArray());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ContextsShareOnlyPreparedWeightsAndKeepHeldOutputs(bool disablePool)
    {
        var graph = Graph(); graph.Prepare(); var expected = Reference(graph);
        var options = ExecutionOptions.Memory with { Tensor = TensorExecutionOptions.Auto with { DisableBufferPool = disablePool } };
        var a = graph.CreateExecution(options); var b = graph.CreateExecution(options);
        Assert.Same(graph.PackedConvWeights, a.PackedConvWeights); Assert.Same(a.PackedConvWeights, b.PackedConvWeights);
        var results = await Task.WhenAll(Task.Run(() => Run(a, options)), Task.Run(() => Run(b, options)));
        Assert.NotSame(results[0], results[1]); Close(expected, results[0].ToArray()); Close(expected, results[1].ToArray());
        var held0 = results[0].ToArray(); var held1 = results[1].ToArray(); Equal(held0, held1);
        var again = Run(a, options); Equal(held0, results[0].ToArray()); Equal(held1, results[1].ToArray()); Equal(held0, again.ToArray());
    }

    [Fact]
    public void SlicedWeightStorageRetainsFallback()
    {
        var graph = Graph(); var bytes = Values(32 * 16 * 9 + 2);
        graph.Initializers["w"] = new DenseTensor<float>(bytes.AsMemory(1, bytes.Length - 2), new[] { 32, 16, 3, 3 });
        Equal(Reference(graph), Run(graph).ToArray()); Assert.Empty(graph.PackedConvWeights);
    }

    [Fact]
    public void ScratchBoundRejectsOversizeAndOverflowWithoutAllocation()
    {
        Assert.True(Lokad.Onnx.Tensor<float>.PlanConvBlockedScratch(32, 32, 80, 998, 80, 998, out int a, out int b));
        Assert.Equal(32 * 82 * 1000, a); Assert.Equal(32 * 80 * 998, b);
        Assert.False(Lokad.Onnx.Tensor<float>.PlanConvBlockedScratch(256, 256, 1024, 1024, 1024, 1024, out _, out _));
        Assert.False(Lokad.Onnx.Tensor<float>.PlanConvBlockedScratch(int.MaxValue, 32, int.MaxValue, int.MaxValue, 1, 1, out _, out _));
        Assert.False(Lokad.Onnx.Tensor<float>.PlanConvBlockedScratch(16, 32, 0, 1, 1, 1, out _, out _));
    }

    [Theory]
    [InlineData(8, 1, 1)]
    [InlineData(16, 3, 1)]
    [InlineData(16, 1, 2)]
    public void UnsupportedChannelsStridesAndDilationKeepFallback(int channels, int stride, int dilation)
    {
        var graph = Graph(c: channels, stride: stride);
        graph.Nodes[0].Attributes!["dilations"] = new[] { dilation, dilation };
        Equal(Reference(graph), Run(graph).ToArray());
    }

    [Fact]
    public void FailedRequestDoesNotCorruptPreparedWeightsOrHeldOutputs()
    {
        var graph = Graph(); var held = Run(graph); var expected = held.ToArray(); Close(Reference(graph), expected);
        Assert.False(graph.Execute(new Dictionary<string, ITensor>(), true));
        Equal(expected, held.ToArray()); Equal(expected, Run(graph).ToArray());
    }
}
