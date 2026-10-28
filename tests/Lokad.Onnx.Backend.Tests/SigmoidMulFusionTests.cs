using Lokad.Onnx.Optimization;

namespace Lokad.Onnx.Backend.Tests;

public class SigmoidMulFusionTests
{
    static Node Node(OpType op, string[] inputs, string output) => new()
    {
        Name = output, Op = op, OpTypeName = op.ToString(), Domain = "", OpsetVersion = 18,
        Inputs = inputs, Outputs = new[] { output }
    };

    static ComputationalGraph Graph(ITensor input)
    {
        var graph = new ComputationalGraph(); graph.Opset[""] = 18; graph.Metadata["Name"] = "sigmoid-mul";
        graph.Inputs["x"] = input; graph.Outputs["y"] = input;
        graph.Nodes.Add(Node(OpType.Sigmoid, new[] { "x" }, "s"));
        graph.Nodes.Add(Node(OpType.Mul, new[] { "x", "s" }, "y"));
        return graph;
    }

    static ITensor Run(ComputationalGraph graph, ITensor input, ExecutionOptions options)
    {
        Assert.True(graph.Execute(new Dictionary<string, ITensor> { ["x"] = input }, true, ExecutionProvider.CPU, options), graph.LastErrorMessage);
        return graph.Outputs["y"];
    }

    static ComputationalGraph UnknownGraph(ITensor input)
    {
        var graph = Graph(input);
        var sigmoid = graph.Nodes[0]; sigmoid.Inputs = new[] { "q" }; graph.Nodes[0] = sigmoid;
        var mul = graph.Nodes[1]; mul.Inputs = new[] { "q", "s" }; graph.Nodes[1] = mul;
        graph.Nodes.Insert(0, Node(OpType.Identity, new[] { "x" }, "q"));
        return graph;
    }

    static void Equal(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            if (float.IsNaN(expected[i])) Assert.True(float.IsNaN(actual[i]));
            else Assert.Equal(BitConverter.SingleToInt32Bits(expected[i]), BitConverter.SingleToInt32Bits(actual[i]));
    }

    [Theory]
    [InlineData(0)] [InlineData(1)] [InlineData(7)] [InlineData(8)] [InlineData(15)]
    [InlineData(16)] [InlineData(17)] [InlineData(31)] [InlineData(32)] [InlineData(33)]
    [InlineData(257)]
    public void FusedFloatMatchesOriginalArithmeticTailsAndOwnership(int length)
    {
        float[] special = [-float.MaxValue, float.NegativeInfinity, -18f, -1f, -0f, 0f,
            float.Epsilon, 1f, 18f, float.PositiveInfinity, float.MaxValue, float.NaN];
        var values = Enumerable.Range(0, length).Select(i => special[i % special.Length]).ToArray();
        foreach (var options in new[] { ExecutionOptions.Default, ExecutionOptions.Simd, ExecutionOptions.Scalar, ExecutionOptions.Memory })
        {
            var input = new DenseTensor<float>(values.ToArray(), new[] { length });
            var plain = Graph(input); var fused = Graph(input);
            Assert.Equal(1, GraphFusion.FuseSigmoidMul(fused, GraphFacts.Build(fused)));
            Assert.Equal(OpType.FusedSigmoidMul, Assert.Single(fused.Nodes).Op);
            var expected = ((Tensor<float>)Run(plain, input, options)).ToArray();
            var held = (Tensor<float>)Run(fused, input, options); var snapshot = held.ToArray();
            Equal(expected, snapshot); Equal(snapshot, ((Tensor<float>)Run(fused, input, options)).ToArray());
            Equal(snapshot, held.ToArray()); Equal(values, input.ToArray());
        }
    }

    [Fact]
    public void LogicalLayoutsAndOffsetsPreserveTheOriginalResult()
    {
        var values = Enumerable.Range(0, 77).Select(i => (i - 38) * .125f).ToArray();
        var reversed = new DenseTensor<float>(new[] { 7, 11 }, true);
        for (int i = 0; i < 7; i++) for (int j = 0; j < 11; j++) reversed[i, j] = values[i * 11 + j];
        var source = new DenseTensor<float>(values, new[] { 7, 11 });
        var offset = new DenseTensor<float>(values.AsMemory(2, 66), new[] { 6, 11 });
        var slice = new TensorSlice<float>(source, new[] { new SliceIndex(1, 6), new SliceIndex(1, 10, 2) });
        foreach (var input in new Tensor<float>[] { reversed, offset, slice })
        {
            var fused = Graph(input); Assert.Equal(1, GraphFusion.FuseSigmoidMul(fused, GraphFacts.Build(fused)));
            Equal(((Tensor<float>)Run(Graph(input), input, ExecutionOptions.Default)).ToArray(),
                ((Tensor<float>)Run(fused, input, ExecutionOptions.Default)).ToArray());
        }
    }

    [Fact]
    public void DoubleAndOrdinaryFiniteValuesMatchBitwise()
    {
        var input = new DenseTensor<double>(new[] { -1000d, -18, -1, -0d, 0, 1, 18, 1000 }, new[] { 8 });
        var fused = Graph(input); Assert.Equal(1, GraphFusion.FuseSigmoidMul(fused, GraphFacts.Build(fused)));
        var expected = ((Tensor<double>)Run(Graph(input), input, ExecutionOptions.Default)).ToArray();
        var actual = ((Tensor<double>)Run(fused, input, ExecutionOptions.Default)).ToArray();
        for (int i = 0; i < expected.Length; i++) Assert.Equal(BitConverter.DoubleToInt64Bits(expected[i]), BitConverter.DoubleToInt64Bits(actual[i]));
        var floats = new DenseTensor<float>(Enumerable.Range(0, 65539).Select(i => (i % 1001 - 500) * .125f).ToArray(), new[] { 65539 });
        var fast = Graph(floats); Assert.Equal(1, GraphFusion.FuseSigmoidMul(fast, GraphFacts.Build(fast)));
        Equal(((Tensor<float>)Run(Graph(floats), floats, ExecutionOptions.Default)).ToArray(),
            ((Tensor<float>)Run(fast, floats, ExecutionOptions.Default)).ToArray());
    }

    [Theory]
    [InlineData("visible")] [InlineData("shared")] [InlineData("captured")]
    [InlineData("domain")] [InlineData("attribute")] [InlineData("commuted")]
    [InlineData("intervening")] [InlineData("integer")] [InlineData("in-place")]
    public void UnsafeOrUnprovedPatternsRemainUnchanged(string reason)
    {
        var input = new DenseTensor<float>(new[] { 1f, 2f }, new[] { 2 }); var graph = Graph(input);
        switch (reason)
        {
            case "visible": graph.Outputs["s"] = input; break;
            case "shared": graph.Nodes.Add(Node(OpType.Identity, new[] { "s" }, "other")); break;
            case "captured":
                var branch = new ComputationalGraph(); branch.Nodes.Add(Node(OpType.Identity, new[] { "s" }, "z")); branch.Outputs["z"] = input;
                var conditional = Node(OpType.If, new[] { "condition" }, "other");
                conditional.Attributes = new() { ["then_branch"] = branch, ["else_branch"] = branch };
                graph.Nodes.Add(conditional); break;
            case "domain": var node = graph.Nodes[0]; node.Domain = "custom"; graph.Nodes[0] = node; break;
            case "attribute": node = graph.Nodes[0]; node.Attributes = new() { ["unexpected"] = 1 }; graph.Nodes[0] = node; break;
            case "commuted": node = graph.Nodes[1]; node.Inputs = new[] { "s", "x" }; graph.Nodes[1] = node; break;
            case "intervening": graph.Nodes.Insert(1, Node(OpType.Identity, new[] { "x" }, "other")); break;
            case "integer": graph.Inputs["x"] = new DenseTensor<int>(new[] { 1, 2 }, new[] { 2 }); break;
            case "in-place": node = graph.Nodes[0]; node.Outputs = new[] { "x" }; graph.Nodes[0] = node; node = graph.Nodes[1]; node.Inputs = new[] { "x", "x" }; graph.Nodes[1] = node; break;
        }
        Assert.Equal(0, GraphFusion.FuseSigmoidMul(graph, GraphFacts.Build(graph)));
    }

    [Fact]
    public void UnknownProducerDtypeKeepsRuntimeValidationAndTheSameResult()
    {
        var input = new DenseTensor<float>(new[] { -2f, 0f, 2f }, new[] { 3 });
        var plain = UnknownGraph(input); var fused = UnknownGraph(input);
        Assert.False(GraphFacts.Build(fused).Dtypes.ContainsKey("q"));
        Assert.Equal(1, GraphFusion.FuseSigmoidMul(fused, GraphFacts.Build(fused)));
        Equal(((Tensor<float>)Run(plain, input, ExecutionOptions.Default)).ToArray(),
            ((Tensor<float>)Run(fused, input, ExecutionOptions.Default)).ToArray());
        var integer = new DenseTensor<int>(new[] { 1, 2, 3 }, new[] { 3 });
        plain = UnknownGraph(integer); fused = UnknownGraph(integer);
        Assert.Equal(1, GraphFusion.FuseSigmoidMul(fused, GraphFacts.Build(fused)));
        foreach (var graph in new[] { plain, fused })
            Assert.False(graph.Execute(new Dictionary<string, ITensor> { ["x"] = integer }, true, ExecutionProvider.CPU, ExecutionOptions.Default));
        Assert.Equal(new[] { 1, 2, 3 }, integer.ToArray());
    }

    [Fact]
    public void RewriteIsOptionalIdempotentAndCannotBeImportedAsAnOnnxOperator()
    {
        var input = new DenseTensor<float>(new[] { 1f }, new[] { 1 });
        var disabled = Graph(input); GraphOptimizer.Run(disabled, new[] { "sigmoid-mul" }); Assert.Equal(2, disabled.Nodes.Count);
        var enabled = Graph(input); GraphOptimizer.Run(enabled); Assert.Single(enabled.Nodes);
        Assert.Equal(TensorElementType.Float, GraphFacts.Build(enabled).Dtypes["y"]);
        Assert.Equal(TensorElementType.Float, GraphFacts.Build(Graph(input)).Dtypes["s"]);
        Assert.Equal(0, GraphFusion.FuseSigmoidMul(enabled, GraphFacts.Build(enabled)));
        var imported = enabled.Nodes[0]; imported.IsFused = false;
        Assert.False(CPUExecutionProvider.SupportsNode(imported));
    }
}
