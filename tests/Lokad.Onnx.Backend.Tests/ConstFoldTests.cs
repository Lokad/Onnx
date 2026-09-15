using System.Collections.Generic;
using System.Linq;
using Lokad.Onnx.Optimization;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// G02-M1: pure-constant folding replaces foldable chains with bit-identical Constants,
/// drops unused Constants, and deduplicates bitwise-identical payloads, while graph
/// outputs, fused nodes and oversized results are left alone.
/// </summary>
public class ConstFoldTests
{
    static OnnxModel FoldModel()
    {
        var mp = new OnnxModel { Name = "tiny-constfold" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "z1", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "z2", ElementType = TensorElementType.Float, Dims = new[] { 4, 2 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "z3", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "z4", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        var a = new float[12];
        for (int i = 0; i < 12; i++) a[i] = i + 1f;
        mp.Initializers.Add(new OnnxTensor { Name = "A", ElementType = TensorElementType.Float, Dims = new[] { 2, 6 }, Data = a });
        mp.Initializers.Add(new OnnxTensor { Name = "W", ElementType = TensorElementType.Float, Dims = new[] { 4, 4 }, Data = new float[] { 1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f } });
        mp.Initializers.Add(new OnnxTensor { Name = "S", ElementType = TensorElementType.Int64, Dims = new[] { 2 }, Data = new long[] { 3, 4 } });
        mp.Initializers.Add(new OnnxTensor { Name = "starts", ElementType = TensorElementType.Int64, Dims = new[] { 2 }, Data = new long[] { 0, 1 } });
        mp.Initializers.Add(new OnnxTensor { Name = "ends", ElementType = TensorElementType.Int64, Dims = new[] { 2 }, Data = new long[] { 4, 3 } });
        mp.Initializers.Add(new OnnxTensor { Name = "axes", ElementType = TensorElementType.Int64, Dims = new[] { 2 }, Data = new long[] { 0, 1 } });
        mp.Initializers.Add(new OnnxTensor { Name = "steps", ElementType = TensorElementType.Int64, Dims = new[] { 2 }, Data = new long[] { 1, 1 } });
        Dictionary<string, object> NoAttrs() => new Dictionary<string, object>();
        mp.Nodes.Add(new OnnxNode { OpType = "Reshape", Inputs = new[] { "A", "S" }, Outputs = new[] { "t1" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Transpose", Inputs = new[] { "t1" }, Outputs = new[] { "t2" }, Attributes = new Dictionary<string, object> { ["perm"] = new long[] { 1, 0 } } });
        mp.Nodes.Add(new OnnxNode { OpType = "Slice", Inputs = new[] { "t2", "starts", "ends", "axes", "steps" }, Outputs = new[] { "z2" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "MatMul", Inputs = new[] { "x", "W" }, Outputs = new[] { "z1" }, Attributes = NoAttrs() });
        var seven = Model.ToTensor(new OnnxTensor { ElementType = TensorElementType.Float, Dims = new[] { 1 }, Data = new[] { 7f } });
        mp.Nodes.Add(new OnnxNode { OpType = "Constant", Inputs = new string[0], Outputs = new[] { "cA" }, Attributes = new Dictionary<string, object> { ["value"] = seven } });
        mp.Nodes.Add(new OnnxNode { OpType = "Constant", Inputs = new string[0], Outputs = new[] { "cB" }, Attributes = new Dictionary<string, object> { ["value"] = seven } });
        mp.Nodes.Add(new OnnxNode { OpType = "Mul", Inputs = new[] { "x", "cA" }, Outputs = new[] { "z3" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Mul", Inputs = new[] { "x", "cB" }, Outputs = new[] { "z4" }, Attributes = NoAttrs() });
        var stray = Model.ToTensor(new OnnxTensor { ElementType = TensorElementType.Float, Dims = new[] { 2 }, Data = new[] { 1f, 2f } });
        mp.Nodes.Add(new OnnxNode { OpType = "Constant", Inputs = new string[0], Outputs = new[] { "stray" }, Attributes = new Dictionary<string, object> { ["value"] = stray } });
        return mp;
    }

    static float[] RunOutputs(ComputationalGraph graph)
    {
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f } });
        Assert.True(graph.Execute(new Dictionary<string, ITensor> { { "x", x } }, true), graph.LastErrorMessage);
        var flat = new List<float>();
        foreach (var name in new[] { "z1", "z2", "z3", "z4" }) flat.AddRange(((Tensor<float>)graph.Outputs[name]).ToArray());
        return flat.ToArray();
    }

    [Fact]
    public void FoldMatchesUnfusedTwinBitwise()
    {
        var fused = Model.Load(FoldModel())!;
        var plain = Model.Load(FoldModel(), runOptimizer: false)!;
        var a = RunOutputs(fused);
        var b = RunOutputs(plain);
        Assert.Equal(b.Length, a.Length);
        for (int i = 0; i < a.Length; i++) Assert.Equal(b[i], a[i]);
    }

    [Fact]
    public void InitializerChainKeepsNodesWhileConstantsDedupe()
    {
        // Initializer-rooted chains never fold (overridable inputs, replacement and
        // in-place mutation must keep working; see ConstFoldProvenanceTests), while
        // true Constant nodes still merge and strays still sweep.
        var graph = Model.Load(FoldModel())!;
        Assert.Contains(graph.Nodes, n => n.Op == OpType.Reshape);
        Assert.Contains(graph.Nodes, n => n.Op == OpType.Transpose);
        Assert.Contains(graph.Nodes, n => n.Op == OpType.Slice && n.Outputs.Length == 1 && n.Outputs[0] == "z2");
        Assert.DoesNotContain(graph.Nodes, n => n.Op == OpType.Constant && n.Outputs.Length == 1 && n.Outputs[0] == "z2");
        Assert.DoesNotContain(graph.Nodes, n => n.Outputs.Length == 1 && n.Outputs[0] == "stray");
        int sevenConstants = graph.Nodes.Count(n => n.Op == OpType.Constant && n.Outputs.Length == 1 && !n.Outputs[0].StartsWith("z"));
        Assert.Equal(1, sevenConstants);
    }

    [Fact]
    public void StrayConstantDropsWithNothingToFold()
    {
        var mp = new OnnxModel { Name = "tiny-stray" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { 2 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "z", ElementType = TensorElementType.Float, Dims = new[] { 2 } });
        mp.Nodes.Add(new OnnxNode { OpType = "Add", Inputs = new[] { "x", "x" }, Outputs = new[] { "z" }, Attributes = new Dictionary<string, object>() });
        var stray = Model.ToTensor(new OnnxTensor { ElementType = TensorElementType.Float, Dims = new[] { 2 }, Data = new[] { 1f, 2f } });
        mp.Nodes.Add(new OnnxNode { OpType = "Constant", Inputs = new string[0], Outputs = new[] { "stray" }, Attributes = new Dictionary<string, object> { ["value"] = stray } });
        var graph = Model.Load(mp, runOptimizer: false)!;
        Assert.Equal(2, graph.Nodes.Count);
        var report = GraphOptimizer.Run(graph);
        Assert.Contains(report, c => c.Pass == "constfold");
        Assert.Single(graph.Nodes);
        Assert.Equal(OpType.Add, graph.Nodes[0].Op);
    }

    [Fact]    public void DisabledKeepsChain()
    {
        var graph = Model.Load(FoldModel(), runOptimizer: false)!;
        var report = GraphOptimizer.Run(graph, new[] { "constfold" });
        Assert.Empty(report);
        Assert.Contains(graph.Nodes, n => n.OpTypeName == "Reshape");
    }
}
