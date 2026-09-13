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
}
