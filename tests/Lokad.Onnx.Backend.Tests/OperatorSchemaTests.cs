using System.Reflection;
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

    [Fact]
    public void ModHonestlyUnsupported_FailsCleanly()
    {
        // C11: no Mod kernel exists; the dead OpType.Mod enum member must stay
        // honestly unsupported with a clean reason-naming failure, never a throw.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.Mod));
        var node = Nod(OpType.Mod, "", 13,
            new[] { "x", "y" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { 7f, -7f }));
        Bind(graph, "y", DenseTensor<float>.OfValues(new float[] { 3f, 3f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("Mod", r.Message ?? "");
    }

    [Fact]
    public void PadHonestlyUnsupported_FailsCleanly()
    {
        // C11: no Pad kernel exists; like OpType.Mod it must stay honestly
        // unsupported with a clean reason-naming failure, never a throw.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.Pad));
        var node = Nod(OpType.Pad, "", 13,
            new[] { "x", "pads" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { 1f, 2f }));
        Bind(graph, "pads", DenseTensor<long>.OfValues(new long[] { 1L, 1L }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("Pad", r.Message ?? "");
    }

    [Fact]
    public void IdentityHonestlyUnsupported_FailsCleanly()
    {
        // C11: Identity has no schema, provider, or import folding; like
        // Mod/Pad it must stay honestly unsupported with a clean
        // reason-naming failure, never a throw.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.Identity));
        var node = Nod(OpType.Identity, "", 13,
            new[] { "x" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { 1f, 2f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("Identity", r.Message ?? "");
    }

    [Fact]
    public void ClipHonestlyUnsupported_FailsCleanly()
    {
        // C11: no Clip kernel exists anywhere in src; same honest contract.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.Clip));
        var node = Nod(OpType.Clip, "", 13,
            new[] { "x", "lo", "hi" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { -1f, 0.5f, 2f }));
        Bind(graph, "lo", DenseTensor<float>.OfValues(new float[] { 0f }));
        Bind(graph, "hi", DenseTensor<float>.OfValues(new float[] { 1f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("Clip", r.Message ?? "");
    }

    [Fact]
    public void SigmoidHonestlyUnsupported_FailsCleanly()
    {
        // C11: no Sigmoid kernel exists anywhere in src; same honest contract
        // as Mod/Pad/Identity/Clip.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.Sigmoid));
        var node = Nod(OpType.Sigmoid, "", 13,
            new[] { "x" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { 0f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("Sigmoid", r.Message ?? "");
    }

    [Fact]
    public void LogHonestlyUnsupported_FailsCleanly()
    {
        // C11: no Log kernel exists anywhere in src; same honest contract.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.Log));
        var node = Nod(OpType.Log, "", 13,
            new[] { "x" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { 1f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("Log", r.Message ?? "");
    }

    [Fact]
    public void FlattenHonestlyUnsupported_FailsCleanly()
    {
        // C11: no Flatten kernel exists anywhere in src; same honest contract.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.Flatten));
        var node = Nod(OpType.Flatten, "", 13,
            new[] { "x" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[,] { { 1f, 2f } }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("Flatten", r.Message ?? "");
    }

    [Fact]
    public void GreaterHonestlyUnsupported_FailsCleanly()
    {
        // C11: no Greater schema, provider, kernel, or dispatch arm exists
        // anywhere in src; same honest contract as Sigmoid/Log/Flatten.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.Greater));
        var node = Nod(OpType.Greater, "", 13,
            new[] { "x", "y" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { 2f }));
        Bind(graph, "y", DenseTensor<float>.OfValues(new float[] { 1f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("Greater", r.Message ?? "");
    }

    [Fact]
    public void AndHonestlyUnsupported_FailsCleanly()
    {
        // C11: no And schema, provider, kernel, or dispatch arm exists
        // anywhere in src; same honest contract.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.And));
        var node = Nod(OpType.And, "", 13,
            new[] { "x", "y" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<bool>.OfValues(new bool[] { true }));
        Bind(graph, "y", DenseTensor<bool>.OfValues(new bool[] { false }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("And", r.Message ?? "");
    }

    [Fact]
    public void OrHonestlyUnsupported_FailsCleanly()
    {
        // C11: no Or schema, provider, kernel, or dispatch arm exists
        // anywhere in src; same honest contract.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.Or));
        var node = Nod(OpType.Or, "", 13,
            new[] { "x", "y" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<bool>.OfValues(new bool[] { true }));
        Bind(graph, "y", DenseTensor<bool>.OfValues(new bool[] { false }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("Or", r.Message ?? "");
    }

    [Fact]
    public void ReciprocalHonestlyUnsupported_FailsCleanly()
    {
        // C11: no Reciprocal schema, provider, kernel, or dispatch arm
        // exists anywhere in src; same honest contract as Greater/And/Or.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.Reciprocal));
        var node = Nod(OpType.Reciprocal, "", 13,
            new[] { "x" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { 2f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("Reciprocal", r.Message ?? "");
    }

    [Fact]
    public void XorHonestlyUnsupported_FailsCleanly()
    {
        // C11: no Xor schema, provider, kernel, or dispatch arm exists
        // anywhere in src; same honest contract.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.Xor));
        var node = Nod(OpType.Xor, "", 13,
            new[] { "x", "y" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<bool>.OfValues(new bool[] { true }));
        Bind(graph, "y", DenseTensor<bool>.OfValues(new bool[] { false }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("Xor", r.Message ?? "");
    }

    [Fact]
    public void NotHonestlyUnsupported_FailsCleanly()
    {
        // C11: no Not schema, provider, kernel, or dispatch arm exists
        // anywhere in src; same honest contract.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.Not));
        var node = Nod(OpType.Not, "", 13,
            new[] { "x" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<bool>.OfValues(new bool[] { true }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("Not", r.Message ?? "");
    }

    [Fact]
    public void GreaterOrEqualHonestlyUnsupported_FailsCleanly()
    {
        // C11: no GreaterOrEqual schema, provider, kernel, or dispatch arm
        // exists anywhere in src; same honest contract.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.GreaterOrEqual));
        var node = Nod(OpType.GreaterOrEqual, "", 13,
            new[] { "x", "y" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { 2f }));
        Bind(graph, "y", DenseTensor<float>.OfValues(new float[] { 1f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("GreaterOrEqual", r.Message ?? "");
    }

    [Fact]
    public void LessOrEqualHonestlyUnsupported_FailsCleanly()
    {
        // C11: no LessOrEqual schema, provider, kernel, or dispatch arm
        // exists anywhere in src; same honest contract.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.LessOrEqual));
        var node = Nod(OpType.LessOrEqual, "", 13,
            new[] { "x", "y" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { 1f }));
        Bind(graph, "y", DenseTensor<float>.OfValues(new float[] { 2f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("LessOrEqual", r.Message ?? "");
    }

    [Fact]
    public void ArgMaxHonestlyUnsupported_FailsCleanly()
    {
        // C11: no ArgMax schema, provider, kernel, or dispatch arm exists
        // anywhere in src; same honest contract as the earlier batches.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.ArgMax));
        var node = Nod(OpType.ArgMax, "", 13,
            new[] { "x" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { 1f, 3f, 2f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("ArgMax", r.Message ?? "");
    }

    [Fact]
    public void ArgMinHonestlyUnsupported_FailsCleanly()
    {
        // C11: no ArgMin schema, provider, kernel, or dispatch arm exists
        // anywhere in src; same honest contract.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.ArgMin));
        var node = Nod(OpType.ArgMin, "", 13,
            new[] { "x" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { 1f, 3f, 2f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("ArgMin", r.Message ?? "");
    }

    [Fact]
    public void AveragePoolHonestlyUnsupported_FailsCleanly()
    {
        // C11: no AveragePool schema, provider, kernel, or dispatch arm
        // exists anywhere in src; same honest contract.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.AveragePool));
        var node = Nod(OpType.AveragePool, "", 13,
            new[] { "x" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f } }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("AveragePool", r.Message ?? "");
    }

    [Fact]
    public void BatchNormalizationHonestlyUnsupported_FailsCleanly()
    {
        // C11: no BatchNormalization schema, provider, kernel, or dispatch
        // arm exists anywhere in src; same honest contract.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.BatchNormalization));
        var node = Nod(OpType.BatchNormalization, "", 13,
            new[] { "x", "scale", "bias", "mean", "var" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { 1f }));
        Bind(graph, "scale", DenseTensor<float>.OfValues(new float[] { 1f }));
        Bind(graph, "bias", DenseTensor<float>.OfValues(new float[] { 0f }));
        Bind(graph, "mean", DenseTensor<float>.OfValues(new float[] { 0f }));
        Bind(graph, "var", DenseTensor<float>.OfValues(new float[] { 1f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("BatchNormalization", r.Message ?? "");
    }

    [Fact]
    public void DropoutHonestlyUnsupported_FailsCleanly()
    {
        // C11: no Dropout schema, provider, kernel, or dispatch arm exists
        // anywhere in src; same honest contract.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.Dropout));
        var node = Nod(OpType.Dropout, "", 13,
            new[] { "x" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { 1f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("Dropout", r.Message ?? "");
    }

    [Fact]
    public void TopKHonestlyUnsupported_FailsCleanly()
    {
        // C11: no TopK schema, provider, kernel, or dispatch arm exists
        // anywhere in src; same honest contract.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.TopK));
        var node = Nod(OpType.TopK, "", 13,
            new[] { "x", "k" }, new[] { "values", "indices" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { 1f, 3f, 2f }));
        Bind(graph, "k", DenseTensor<long>.OfValues(new long[] { 2L }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("TopK", r.Message ?? "");
    }

    [Fact]
    public void ExpHonestlyUnsupported_FailsCleanly()
    {
        // C11: no Exp schema, provider, kernel, or dispatch arm exists
        // anywhere in src (only MathF.Exp call sites); same honest contract.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.Exp));
        var node = Nod(OpType.Exp, "", 13,
            new[] { "x" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { 0f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("Exp", r.Message ?? "");
    }

    [Fact]
    public void MaxHonestlyUnsupported_FailsCleanly()
    {
        // C11: no Max schema, provider, kernel, or dispatch arm exists
        // anywhere in src (only Math.Max call sites); same honest contract.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.Max));
        var node = Nod(OpType.Max, "", 13,
            new[] { "x", "y" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { 1f }));
        Bind(graph, "y", DenseTensor<float>.OfValues(new float[] { 2f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("Max", r.Message ?? "");
    }

    [Fact]
    public void MinHonestlyUnsupported_FailsCleanly()
    {
        // C11: no Min schema, provider, kernel, or dispatch arm exists
        // anywhere in src (only Math.Min call sites); same honest contract.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.Min));
        var node = Nod(OpType.Min, "", 13,
            new[] { "x" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { 1f }));
        Bind(graph, "y", DenseTensor<float>.OfValues(new float[] { 2f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("Min", r.Message ?? "");
    }

    [Fact]
    public void MeanHonestlyUnsupported_FailsCleanly()
    {
        // C11: no Mean schema, provider, kernel, or dispatch arm exists
        // anywhere in src; same honest contract.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.Mean));
        var node = Nod(OpType.Mean, "", 13,
            new[] { "x", "y" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { 1f }));
        Bind(graph, "y", DenseTensor<float>.OfValues(new float[] { 2f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("Mean", r.Message ?? "");
    }

    [Fact]
    public void SumHonestlyUnsupported_FailsCleanly()
    {
        // C11: no Sum schema, provider, kernel, or dispatch arm exists
        // anywhere in src (only LINQ/Vector call sites); same honest contract.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.Sum));
        var node = Nod(OpType.Sum, "", 13,
            new[] { "x", "y" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { 1f }));
        Bind(graph, "y", DenseTensor<float>.OfValues(new float[] { 2f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("Sum", r.Message ?? "");
    }

    [Fact]
    public void CastLikeHonestlyUnsupported_FailsCleanly()
    {
        // C11: no CastLike schema, provider, kernel, or dispatch arm exists
        // anywhere in src (unlike Cast); same honest contract (verified end
        // to end via OpDump: ORT 1.29 runs float->int64, Lokad fails cleanly).
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.CastLike));
        var node = Nod(OpType.CastLike, "", 19,
            new[] { "x", "like" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(19);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { 1.5f, 2.5f }));
        Bind(graph, "like", DenseTensor<long>.OfValues(new long[] { 1L }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("CastLike", r.Message ?? "");
    }

    [Fact]
    public void GlobalMaxPoolHonestlyUnsupported_FailsCleanly()
    {
        // C11: GlobalMaxPool has an enum entry but no schema, provider,
        // kernel, or dispatch arm anywhere in src (unlike
        // GlobalAveragePool); same honest contract (verified end to end
        // via OpDump: clean load refusal).
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.GlobalMaxPool));
        var node = Nod(OpType.GlobalMaxPool, "", 14,
            new[] { "x" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(14);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[1, 1, 2, 2] { { { { 1f, 2f }, { 3f, 4f } } } }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("GlobalMaxPool", r.Message ?? "");
    }

    [Fact]
    public void BitShiftHonestlyUnsupported_FailsCleanly()
    {
        // C11: no BitShift schema, provider, kernel, or dispatch arm exists
        // anywhere in src; same honest contract (verified end to end via
        // OpDump: ORT 1.29 runs uint8 shifts, Lokad fails cleanly).
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.BitShift));
        var node = Nod(OpType.BitShift, "", 11,
            new[] { "x", "y" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(11);
        Bind(graph, "x", DenseTensor<byte>.OfValues(new byte[] { 1, 2, 3 }));
        Bind(graph, "y", DenseTensor<byte>.OfValues(new byte[] { 1, 1, 1 }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("BitShift", r.Message ?? "");
    }

    [Fact]
    public void IsNaNHonestlyUnsupported_FailsCleanly()
    {
        // C11: no IsNaN schema, provider, kernel, or dispatch arm exists
        // anywhere in src (unlike the pinned IsInf); same honest contract
        // (verified end to end via OpDump).
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.IsNaN));
        var node = Nod(OpType.IsNaN, "", 20,
            new[] { "x" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(20);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { 1f, 2f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("IsNaN", r.Message ?? "");
    }

    [Fact]
    public void SequenceEmptyHonestlyUnsupported_FailsCleanly()
    {
        // C11: no SequenceEmpty schema, provider, kernel, or dispatch arm
        // exists anywhere in src (SplitToSequence/SequenceAt are supported,
        // the construction family is not); same honest contract.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.SequenceEmpty));
        var node = Nod(OpType.SequenceEmpty, "", 11,
            new string[0], new[] { "s" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(11);
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("SequenceEmpty", r.Message ?? "");
    }

    [Fact]
    public void SequenceConstructHonestlyUnsupported_FailsCleanly()
    {
        // C11: no SequenceConstruct schema, provider, kernel, or dispatch
        // arm exists anywhere in src; same honest contract.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.SequenceConstruct));
        var node = Nod(OpType.SequenceConstruct, "", 11,
            new[] { "a", "b" }, new[] { "s" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(11);
        Bind(graph, "a", DenseTensor<float>.OfValues(new float[] { 1f }));
        Bind(graph, "b", DenseTensor<float>.OfValues(new float[] { 2f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("SequenceConstruct", r.Message ?? "");
    }

    [Fact]
    public void SequenceInsertHonestlyUnsupported_FailsCleanly()
    {
        // C11: no SequenceInsert schema, provider, kernel, or dispatch arm
        // exists anywhere in src; same honest contract.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.SequenceInsert));
        var node = Nod(OpType.SequenceInsert, "", 11,
            new[] { "s", "t" }, new[] { "o" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(11);
        Bind(graph, "t", DenseTensor<float>.OfValues(new float[] { 1f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("SequenceInsert", r.Message ?? "");
    }

    [Fact]
    public void SequenceEraseHonestlyUnsupported_FailsCleanly()
    {
        // C11: no SequenceErase schema, provider, kernel, or dispatch arm
        // exists anywhere in src; same honest contract.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.SequenceErase));
        var node = Nod(OpType.SequenceErase, "", 11,
            new[] { "s" }, new[] { "o" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(11);
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("SequenceErase", r.Message ?? "");
    }

    [Fact]
    public void SequenceLengthHonestlyUnsupported_FailsCleanly()
    {
        // C11: no SequenceLength schema, provider, kernel, or dispatch arm
        // exists anywhere in src; same honest contract.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.SequenceLength));
        var node = Nod(OpType.SequenceLength, "", 11,
            new[] { "s" }, new[] { "n" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(11);
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("SequenceLength", r.Message ?? "");
    }

    [Fact]
    public void ConcatFromSequenceHonestlyUnsupported_FailsCleanly()
    {
        // C11: no ConcatFromSequence schema, provider, kernel, or dispatch
        // arm exists anywhere in src; same honest contract.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.ConcatFromSequence));
        var node = Nod(OpType.ConcatFromSequence, "", 11,
            new[] { "s" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(11);
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("ConcatFromSequence", r.Message ?? "");
    }

    [Fact]
    public void EluHonestlyUnsupported_FailsCleanly()
    {
        // C11: no Elu schema, provider, kernel, or dispatch arm exists
        // anywhere in src; same honest contract as Sigmoid/Log.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.Elu));
        var node = Nod(OpType.Elu, "", 13,
            new[] { "x" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { -1f, 0.5f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("Elu", r.Message ?? "");
    }

    [Fact]
    public void SeluHonestlyUnsupported_FailsCleanly()
    {
        // C11: no Selu schema, provider, kernel, or dispatch arm exists
        // anywhere in src; same honest contract.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.Selu));
        var node = Nod(OpType.Selu, "", 13,
            new[] { "x" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { -1f, 0.5f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("Selu", r.Message ?? "");
    }

    [Fact]
    public void CeluHonestlyUnsupported_FailsCleanly()
    {
        // C11: no Celu schema, provider, kernel, or dispatch arm exists
        // anywhere in src; same honest contract.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.Celu));
        var node = Nod(OpType.Celu, "", 13,
            new[] { "x" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { -1f, 0.5f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("Celu", r.Message ?? "");
    }

    [Fact]
    public void LeakyReluHonestlyUnsupported_FailsCleanly()
    {
        // C11: no LeakyRelu schema, provider, kernel, or dispatch arm exists
        // anywhere in src; same honest contract (verified end to end via OpDump).
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.LeakyRelu));
        var node = Nod(OpType.LeakyRelu, "", 13,
            new[] { "x" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { -1f, 0.5f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("LeakyRelu", r.Message ?? "");
    }

    [Fact]
    public void PReluHonestlyUnsupported_FailsCleanly()
    {
        // C11: no PRelu schema, provider, kernel, or dispatch arm exists
        // anywhere in src; same honest contract.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.PRelu));
        var node = Nod(OpType.PRelu, "", 13,
            new[] { "x", "slope" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { -1f, 0.5f }));
        Bind(graph, "slope", DenseTensor<float>.OfValues(new float[] { 0.25f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("PRelu", r.Message ?? "");
    }

    [Fact]
    public void ThresholdedReluHonestlyUnsupported_FailsCleanly()
    {
        // C11: no ThresholdedRelu schema, provider, kernel, or dispatch arm
        // exists anywhere in src; same honest contract.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.ThresholdedRelu));
        var node = Nod(OpType.ThresholdedRelu, "", 13,
            new[] { "x" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { -1f, 0.5f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("ThresholdedRelu", r.Message ?? "");
    }

    [Fact]
    public void HardSigmoidHonestlyUnsupported_FailsCleanly()
    {
        // C11: no HardSigmoid schema, provider, kernel, or dispatch arm exists
        // anywhere in src; same honest contract.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.HardSigmoid));
        var node = Nod(OpType.HardSigmoid, "", 13,
            new[] { "x" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { -1f, 0.5f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("HardSigmoid", r.Message ?? "");
    }

    [Fact]
    public void HardSwishHonestlyUnsupported_FailsCleanly()
    {
        // C11: no HardSwish schema, provider, kernel, or dispatch arm exists
        // anywhere in src; same honest contract.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.HardSwish));
        var node = Nod(OpType.HardSwish, "", 13,
            new[] { "x" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { -1f, 0.5f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("HardSwish", r.Message ?? "");
    }

    [Fact]
    public void SoftplusHonestlyUnsupported_FailsCleanly()
    {
        // C11: no Softplus schema, provider, kernel, or dispatch arm exists
        // anywhere in src; same honest contract.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.Softplus));
        var node = Nod(OpType.Softplus, "", 13,
            new[] { "x" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { -1f, 0.5f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("Softplus", r.Message ?? "");
    }

    [Fact]
    public void SoftsignHonestlyUnsupported_FailsCleanly()
    {
        // C11: no Softsign schema, provider, kernel, or dispatch arm exists
        // anywhere in src; same honest contract.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.Softsign));
        var node = Nod(OpType.Softsign, "", 13,
            new[] { "x" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { -1f, 0.5f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("Softsign", r.Message ?? "");
    }

    [Fact]
    public void MishHonestlyUnsupported_FailsCleanly()
    {
        // C11: no Mish schema, provider, kernel, or dispatch arm exists
        // anywhere in src; same honest contract.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.Mish));
        var node = Nod(OpType.Mish, "", 13,
            new[] { "x" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { -1f, 0.5f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("Mish", r.Message ?? "");
    }

    [Fact]
    public void ReduceMinHonestlyUnsupported_FailsCleanly()
    {
        // C11: no ReduceMin schema, provider, kernel, or dispatch arm exists
        // anywhere in src (unlike ReduceMean/Sum/Max); same honest contract
        // (verified end to end via OpDump).
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.ReduceMin));
        var node = Nod(OpType.ReduceMin, "", 13,
            new[] { "x" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { 1f, 3f, 2f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("ReduceMin", r.Message ?? "");
    }

    [Fact]
    public void ReduceProdHonestlyUnsupported_FailsCleanly()
    {
        // C11: no ReduceProd schema, provider, kernel, or dispatch arm exists
        // anywhere in src (unlike ReduceMean/Sum/Max); same honest contract.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.ReduceProd));
        var node = Nod(OpType.ReduceProd, "", 13,
            new[] { "x" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { 1f, 3f, 2f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("ReduceProd", r.Message ?? "");
    }

    [Fact]
    public void UpsampleHonestlyUnsupported_FailsCleanly()
    {
        // C11: no Upsample schema, provider, kernel, or dispatch arm exists
        // anywhere in src (deprecated Resize alias); same honest contract
        // (verified end to end via OpDump).
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.Upsample));
        var node = Nod(OpType.Upsample, "", 9,
            new[] { "x", "scales" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(9);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { 1f, 2f }));
        Bind(graph, "scales", DenseTensor<float>.OfValues(new float[] { 2f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("Upsample", r.Message ?? "");
    }

    [Fact]
    public void SignHonestlyUnsupported_FailsCleanly()
    {
        // C11: no Sign schema, provider, kernel, or dispatch arm exists
        // anywhere in src; same honest contract (verified end to end via OpDump).
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.Sign));
        var node = Nod(OpType.Sign, "", 13,
            new[] { "x" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { -1f, 0f, 2f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("Sign", r.Message ?? "");
    }

    [Fact]
    public void ShrinkHonestlyUnsupported_FailsCleanly()
    {
        // C11: no Shrink schema, provider, kernel, or dispatch arm exists
        // anywhere in src; same honest contract (verified end to end via OpDump).
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.Shrink));
        var node = Nod(OpType.Shrink, "", 13,
            new[] { "x" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(13);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { -1f, 0f, 2f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("Shrink", r.Message ?? "");
    }

    [Fact]
    public void IsInfHonestlyUnsupported_FailsCleanly()
    {
        // C11: no IsInf schema, provider, kernel, or dispatch arm exists
        // anywhere in src; same honest contract (verified end to end via OpDump).
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.IsInf));
        var node = Nod(OpType.IsInf, "", 20,
            new[] { "x" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(20);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { -1f, 0f, 2f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("IsInf", r.Message ?? "");
    }



    [Fact]
    public void RegistryEntries_AreImmutable()
    {
        // C08: no consumer may rewrite the capability registry after construction.
        var fields = typeof(OperatorSchema).GetFields(BindingFlags.Public | BindingFlags.Instance);
        Assert.Empty(fields);
        var props = typeof(OperatorSchema).GetProperties(BindingFlags.Public | BindingFlags.Instance);
        Assert.NotEmpty(props);
        foreach (var prop in props)
        {
            var set = prop.SetMethod;
            bool initOnly = false;
            if (set is not null)
            {
                foreach (var mod in set.ReturnParameter.GetRequiredCustomModifiers())
                    if (mod.FullName == "System.Runtime.CompilerServices.IsExternalInit") initOnly = true;
            }
            Assert.True(set is null || initOnly, "settable property " + prop.Name);
        }
        Assert.NotEmpty(OperatorSchemas.All);
        Assert.True(CPUExecutionProvider.SupportsOp(OpType.Add));
    }

    [Fact]
    public void CumSumHonestlyUnsupported_FailsCleanly()
    {
        // C11: no CumSum schema, provider, kernel, or dispatch arm exists
        // anywhere in src; same honest contract (verified end to end via OpDump).
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.CumSum));
        var node = Nod(OpType.CumSum, "", 14,
            new[] { "x" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(14);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { 1f, 2f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("CumSum", r.Message ?? "");
    }

    [Fact]
    public void HardmaxHonestlyUnsupported_FailsCleanly()
    {
        // C11: no Hardmax schema, provider, kernel, or dispatch arm exists
        // anywhere in src; same honest contract (verified end to end via OpDump).
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.Hardmax));
        var node = Nod(OpType.Hardmax, "", 14,
            new[] { "x" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(14);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { 1f, 2f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("Hardmax", r.Message ?? "");
    }

    [Fact]
    public void DepthToSpaceHonestlyUnsupported_FailsCleanly()
    {
        // C11: no DepthToSpace schema, provider, kernel, or dispatch arm
        // exists anywhere in src; same honest contract (verified end to end
        // via OpDump).
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.DepthToSpace));
        var node = Nod(OpType.DepthToSpace, "", 14,
            new[] { "x" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(14);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { 1f, 2f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("DepthToSpace", r.Message ?? "");
    }

    [Fact]
    public void IfHonestlyUnsupported_FailsCleanly()
    {
        // C11: If exists only as an enum value; no schema, provider,
        // kernel, or dispatch arm exists anywhere in src (control flow
        // is out of scope); same honest contract.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.If));
        var node = Nod(OpType.If, "", 14,
            new[] { "x" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(14);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { 1f, 2f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("If", r.Message ?? "");
    }

    [Fact]
    public void LoopHonestlyUnsupported_FailsCleanly()
    {
        // C11: Loop exists only as an enum value; same honest contract.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.Loop));
        var node = Nod(OpType.Loop, "", 14,
            new[] { "x" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(14);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { 1f, 2f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("Loop", r.Message ?? "");
    }

    [Fact]
    public void ScanHonestlyUnsupported_FailsCleanly()
    {
        // C11: Scan exists only as an enum value; same honest contract.
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.Scan));
        var node = Nod(OpType.Scan, "", 14,
            new[] { "x" }, new[] { "z" }, false);
        Assert.False(CPUExecutionProvider.SupportsNode(node));
        var graph = Graph(14);
        Bind(graph, "x", DenseTensor<float>.OfValues(new float[] { 1f, 2f }));
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("Scan", r.Message ?? "");
    }
}
