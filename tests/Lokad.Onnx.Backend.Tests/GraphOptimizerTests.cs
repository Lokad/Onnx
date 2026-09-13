using System.Collections.Generic;
using System.Linq;
using Lokad.Onnx.Optimization;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// G01-M2: the LayerNorm rewrite runs as a pipeline pass. The pass fires through the
/// runner with identical results, stays off when disabled by name, and is a no-op on
/// an already-fused graph. Model.Load behavior itself is covered by the unmodified
/// GraphFusionTests families going through the public entry.
/// </summary>
public class GraphOptimizerTests
{
    static OnnxModel LayerNormModel()
    {
        var mp = new OnnxModel { Name = "tiny-ln-pass" };
        mp.Opset[""] = 11;
        mp.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "z", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        mp.Initializers.Add(new OnnxTensor { Name = "gamma", ElementType = TensorElementType.Float, Dims = new[] { 4 }, Data = new[] { 1f, 2f, 3f, 4f } });
        mp.Initializers.Add(new OnnxTensor { Name = "beta", ElementType = TensorElementType.Float, Dims = new[] { 4 }, Data = new[] { 0.5f, -0.5f, 1f, 0f } });
        mp.Initializers.Add(new OnnxTensor { Name = "two", ElementType = TensorElementType.Float, Dims = new int[0], Data = new[] { 2f } });
        var eps = Model.ToTensor(new OnnxTensor { ElementType = TensorElementType.Float, Dims = new int[0], Data = new[] { 1e-5f } });
        Dictionary<string, object> NoAttrs() => new Dictionary<string, object>();
        mp.Nodes.Add(new OnnxNode { OpType = "ReduceMean", Inputs = new[] { "x" }, Outputs = new[] { "m" }, Attributes = new Dictionary<string, object> { ["axes"] = new long[] { -1 }, ["keepdims"] = 1L } });
        mp.Nodes.Add(new OnnxNode { OpType = "Sub", Inputs = new[] { "x", "m" }, Outputs = new[] { "d" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Pow", Inputs = new[] { "d", "two" }, Outputs = new[] { "s" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "ReduceMean", Inputs = new[] { "s" }, Outputs = new[] { "v" }, Attributes = new Dictionary<string, object> { ["axes"] = new long[] { -1 }, ["keepdims"] = 1L } });
        mp.Nodes.Add(new OnnxNode { OpType = "Constant", Inputs = new string[0], Outputs = new[] { "e" }, Attributes = new Dictionary<string, object> { ["value"] = eps } });
        mp.Nodes.Add(new OnnxNode { OpType = "Add", Inputs = new[] { "v", "e" }, Outputs = new[] { "ve" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Sqrt", Inputs = new[] { "ve" }, Outputs = new[] { "sd" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Div", Inputs = new[] { "d", "sd" }, Outputs = new[] { "n" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Mul", Inputs = new[] { "n", "gamma" }, Outputs = new[] { "g" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Add", Inputs = new[] { "g", "beta" }, Outputs = new[] { "z" }, Attributes = NoAttrs() });
        return mp;
    }

    [Fact]
    public void LayerNormPassFiresThroughPipeline()
    {
        var graph = Model.Load(LayerNormModel(), runOptimizer: false)!;
        Assert.Equal(10, graph.Nodes.Count);
        Assert.DoesNotContain(graph.Nodes, n => n.Op == OpType.LayerNormalization);
        var report = GraphOptimizer.Run(graph);
        var change = Assert.Single(report.Where(c => c.Pass == "layernorm"));
        Assert.Equal(1, change.Rewritten);
        Assert.Single(graph.Nodes);
        Assert.Equal(OpType.LayerNormalization, graph.Nodes[0].Op);
    }

    [Fact]
    public void DisabledPassKeepsPattern()
    {
        var graph = Model.Load(LayerNormModel(), runOptimizer: false)!;
        var report = GraphOptimizer.Run(graph, new[] { "layernorm" });
        Assert.Empty(report);
        Assert.Equal(10, graph.Nodes.Count);
        Assert.DoesNotContain(graph.Nodes, n => n.Op == OpType.LayerNormalization);
    }

    [Fact]
    public void SecondRunIsIdempotent()
    {
        var graph = Model.Load(LayerNormModel(), runOptimizer: false)!;
        var first = GraphOptimizer.Run(graph);
        Assert.NotEmpty(first);
        int fused = graph.Nodes.Count;
        var second = GraphOptimizer.Run(graph);
        Assert.Empty(second);
        Assert.Equal(fused, graph.Nodes.Count);
    }

    static OnnxModel TinyRopeModel(bool exposeRot)
    {
        var mp = new OnnxModel { Name = "tiny-rope-pass" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "z", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        if (exposeRot) mp.Outputs.Add(new OnnxValueInfo { Name = "rot", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        mp.Initializers.Add(new OnnxTensor { Name = "freq", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 }, Data = new[] { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f } });
        mp.Initializers.Add(new OnnxTensor { Name = "s0", ElementType = TensorElementType.Int64, Dims = new[] { 1 }, Data = new long[] { 0 } });
        mp.Initializers.Add(new OnnxTensor { Name = "e0", ElementType = TensorElementType.Int64, Dims = new[] { 1 }, Data = new long[] { 2 } });
        mp.Initializers.Add(new OnnxTensor { Name = "ax", ElementType = TensorElementType.Int64, Dims = new[] { 1 }, Data = new long[] { -1 } });
        mp.Initializers.Add(new OnnxTensor { Name = "st", ElementType = TensorElementType.Int64, Dims = new[] { 1 }, Data = new long[] { 1 } });
        mp.Initializers.Add(new OnnxTensor { Name = "s1", ElementType = TensorElementType.Int64, Dims = new[] { 1 }, Data = new long[] { 2 } });
        mp.Initializers.Add(new OnnxTensor { Name = "e1", ElementType = TensorElementType.Int64, Dims = new[] { 1 }, Data = new long[] { long.MaxValue } });
        Dictionary<string, object> NoAttrs() => new Dictionary<string, object>();
        mp.Nodes.Add(new OnnxNode { OpType = "Cos", Inputs = new[] { "freq" }, Outputs = new[] { "co" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Sin", Inputs = new[] { "freq" }, Outputs = new[] { "si" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Slice", Inputs = new[] { "x", "s0", "e0", "ax", "st" }, Outputs = new[] { "x0" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Slice", Inputs = new[] { "x", "s1", "e1", "ax", "st" }, Outputs = new[] { "x1" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Neg", Inputs = new[] { "x1" }, Outputs = new[] { "nx1" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Concat", Inputs = new[] { "nx1", "x0" }, Outputs = new[] { "rot" }, Attributes = new Dictionary<string, object> { ["axis"] = -1L } });
        mp.Nodes.Add(new OnnxNode { OpType = "Mul", Inputs = new[] { "x", "co" }, Outputs = new[] { "m0" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Mul", Inputs = new[] { "rot", "si" }, Outputs = new[] { "m1" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Add", Inputs = new[] { "m0", "m1" }, Outputs = new[] { "z" }, Attributes = NoAttrs() });
        return mp;
    }

    static OnnxModel TinyGeluModel()
    {
        var mp = new OnnxModel { Name = "tiny-gelu-pass" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(new OnnxValueInfo { Name = "xg", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "zg", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        mp.Initializers.Add(new OnnxTensor { Name = "c0", ElementType = TensorElementType.Float, Dims = new int[0], Data = new[] { 1.4142135f } });
        mp.Initializers.Add(new OnnxTensor { Name = "c1", ElementType = TensorElementType.Float, Dims = new int[0], Data = new[] { 1f } });
        mp.Initializers.Add(new OnnxTensor { Name = "c2", ElementType = TensorElementType.Float, Dims = new int[0], Data = new[] { 0.5f } });
        Dictionary<string, object> NoAttrs() => new Dictionary<string, object>();
        // Chain tensors carry a g-suffix: tensor names are SSA, so sharing d/e/m with the
        // LayerNorm half would corrupt the producer map (last writer wins) and void the test.
        mp.Nodes.Add(new OnnxNode { OpType = "Div", Inputs = new[] { "xg", "c0" }, Outputs = new[] { "dg" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Erf", Inputs = new[] { "dg" }, Outputs = new[] { "eg" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Add", Inputs = new[] { "eg", "c1" }, Outputs = new[] { "ag" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Mul", Inputs = new[] { "xg", "ag" }, Outputs = new[] { "mg" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Mul", Inputs = new[] { "mg", "c2" }, Outputs = new[] { "zg" }, Attributes = NoAttrs() });
        return mp;
    }

    [Fact]
    public void RopeExtraConsumerBlocksFusion()
    {
        var graph = Model.Load(TinyRopeModel(exposeRot: true), runOptimizer: false)!;
        Assert.Equal(9, graph.Nodes.Count);
        var report = GraphOptimizer.Run(graph);
        Assert.DoesNotContain(report, c => c.Pass == "rope");
        Assert.Equal(9, graph.Nodes.Count);
        Assert.DoesNotContain(graph.Nodes, n => n.Op == OpType.RotaryEmbedding);
    }

    [Fact]
    public void RopePatternFiresThroughPipeline()
    {
        var graph = Model.Load(TinyRopeModel(exposeRot: false), runOptimizer: false)!;
        var report = GraphOptimizer.Run(graph);
        var change = Assert.Single(report.Where(c => c.Pass == "rope"));
        Assert.Equal(1, change.Rewritten);
        Assert.Contains(graph.Nodes, n => n.Op == OpType.RotaryEmbedding);
    }

    [Fact]
    public void GeluPassFiresThroughPipeline()
    {
        var graph = Model.Load(TinyGeluModel(), runOptimizer: false)!;
        Assert.Equal(5, graph.Nodes.Count);
        var report = GraphOptimizer.Run(graph);
        var change = Assert.Single(report.Where(c => c.Pass == "gelu"));
        Assert.Equal(1, change.Rewritten);
        Assert.Single(graph.Nodes);
        Assert.Equal(OpType.Gelu, graph.Nodes[0].Op);
    }

    [Fact]    public void MixedGraphFusesAllPassesThenIdles()
    {
        var ln = LayerNormModel();
        var gelu = TinyGeluModel();
        foreach (var inp in gelu.Inputs) ln.Inputs.Add(inp);
        foreach (var i in gelu.Initializers) ln.Initializers.Add(i);
        foreach (var o in gelu.Outputs) ln.Outputs.Add(o);
        foreach (var n in gelu.Nodes) ln.Nodes.Add(n);
        var graph = Model.Load(ln, runOptimizer: false)!;
        Assert.Equal(15, graph.Nodes.Count);
        var first = GraphOptimizer.Run(graph);
        Assert.Contains(first, c => c.Pass == "layernorm" && c.Rewritten == 1);
        Assert.Contains(first, c => c.Pass == "gelu" && c.Rewritten == 1);
        Assert.Equal(2, graph.Nodes.Count);
        var second = GraphOptimizer.Run(graph);
        Assert.Empty(second);
    }
}
