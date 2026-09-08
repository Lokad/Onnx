using System.Collections.Generic;

namespace Lokad.Onnx.Backend.Tests;

// R6: one schema-aware operator definition serves support queries, dispatch,
// fusion eligibility, and CLI reporting. Fused-only operations reject imports,
// arity and versions resolve once, and every rejection names its reason.
public class OperatorSchemaTests
{
    static ComputationalGraph Graph(int opset)
    {
        return new ComputationalGraph
        {
            Opset = new Dictionary<string, int> { [""] = opset },
            Metadata = new Dictionary<string, object> { ["Name"] = "test" },
        };
    }

    static Node Nod(OpType op, string? domain, int version, string[] inputs, string[] outputs, bool fused)
    {
        return NodAttrs(op, domain, version, inputs, outputs, fused, new Dictionary<string, object>());
    }

    static Node NodAttrs(OpType op, string? domain, int version, string[] inputs, string[] outputs, bool fused, Dictionary<string, object> attrs)
    {
        return new Node
        {
            Name = "n", Op = op, OpTypeName = op.ToString(), Domain = domain ?? "",
            OpsetVersion = version, IsFused = fused, Inputs = inputs, Outputs = outputs,
            Attributes = attrs,
        };
    }

    static void Bind(ComputationalGraph graph, string name, ITensor tensor)
    {
        graph.Inputs[name] = tensor;
    }

    [Fact]
    public void ImportedRotaryEmbedding_Rejected()
    {
        var node = Nod(OpType.RotaryEmbedding, "", 20,
            new[] { "x", "c", "s" }, new[] { "y" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(20);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f, 4f } }));
        Bind(graph, "c", DenseTensor<float>.OfValues(new float[,] { { 1f, 0f, 1f, 0f } }));
        Bind(graph, "s", DenseTensor<float>.OfValues(new float[,] { { 0f, 1f, 0f, 1f } }));
        node = NodAttrs(OpType.RotaryEmbedding, "", 20,
            new[] { "x", "c", "s" }, new[] { "y" }, false,
            new Dictionary<string, object> { ["half"] = 2 });
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("fused", r.Message ?? "");
    }

    [Fact]
    public void FusedRotaryEmbedding_Supported()
    {
        var node = Nod(OpType.RotaryEmbedding, "", 20,
            new[] { "x", "c", "s" }, new[] { "y" }, true);
        Assert.True(CPUExecutionProvider.SupportsNode(node));
    }

    [Fact]
    public void AddThreeInputs_Rejected()
    {
        var node = Nod(OpType.Add, "", 11, new[] { "a", "b", "c" }, new[] { "z" }, false);
        var graph = Graph(11);
        Bind(graph, "a", DenseTensor<float>.OfValues(new float[] { 1f }));
        Bind(graph, "b", DenseTensor<float>.OfValues(new float[] { 2f }));
        Bind(graph, "c", DenseTensor<float>.OfValues(new float[] { 3f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("inputs", r.Message ?? "");
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(0)]
    public void UnversionedNode_Dispatches(int version)
    {
        var node = Nod(OpType.Add, "", version, new[] { "a", "b" }, new[] { "z" }, false);
        Assert.True(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(11);
        Bind(graph, "a", DenseTensor<float>.OfValues(new float[] { 1f }));
        Bind(graph, "b", DenseTensor<float>.OfValues(new float[] { 2f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new float[] { 3f }, ((Tensor<float>)r.Outputs[0]).ToArray());
    }

    [Theory]
    [InlineData(16, false)]
    [InlineData(17, true)]
    [InlineData(20, true)]
    public void LayerNormVersionFloor(int version, bool supported)
    {
        var node = Nod(OpType.LayerNormalization, "", version,
            new[] { "x", "s" }, new[] { "y" }, false);
        Assert.Equal(supported, CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(version);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[,] { { 1f, 2f } }));
        Bind(graph, "s", DenseTensor<float>.OfValues(new float[] { 1f, 1f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(supported ? OpStatus.Success : OpStatus.Failure, r.Status);
    }

    [Fact]
    public void FusedLayerNorm_OldVersion_Supported()
    {
        var node = Nod(OpType.LayerNormalization, "", 11,
            new[] { "x", "s", "b" }, new[] { "y" }, true);
        Assert.True(CPUExecutionProvider.SupportsNode(node));
    }

    [Fact]
    public void CustomDomain_RejectionNamesDomain()
    {
        var node = Nod(OpType.Add, "custom", 1, new[] { "a", "b" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        Assert.False(OperatorSchemas.TryResolve(node, out _, out var rejection));
        Assert.Contains("custom", rejection ?? "");
    }

    [Fact]
    public void SqueezeSingleInput_Dispatches()
    {
        var node = Nod(OpType.Squeeze, "", 20, new[] { "x" }, new[] { "z" }, false);
        Assert.True(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(20);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[1, 1, 2] { { { 1f, 2f } } }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new[] { 2 }, ((Tensor<float>)r.Outputs[0]).Dimensions.ToArray());
    }

    [Fact]
    public void LayerNormFourOutputs_Rejected()
    {
        var node = Nod(OpType.LayerNormalization, "", 20,
            new[] { "x", "s" }, new[] { "y", "m", "v", "w" }, false);
        var graph = Graph(20);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[,] { { 1f, 2f } }));
        Bind(graph, "s", DenseTensor<float>.OfValues(new float[] { 1f, 1f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("outputs", r.Message ?? "");
    }
}
