using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// P02: one ordered pipeline run per load. Sequential loads agree exactly,
/// concurrent loads agree with each other, and the pass set is complete.
/// </summary>
public class OptimizerPipelineTests
{
    static OnnxModel PipelineModel()
    {
        var mp = new OnnxModel { Name = "tiny-pipeline" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "w", ElementType = TensorElementType.Float, Dims = new[] { 4, 2 } });
        mp.Initializers.Add(new OnnxTensor { Name = "A", ElementType = TensorElementType.Float, Dims = new[] { 2, 6 }, Data = new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f } });
        mp.Initializers.Add(new OnnxTensor { Name = "S", ElementType = TensorElementType.Int64, Dims = new[] { 2 }, Data = new long[] { 3, 4 } });
        mp.Initializers.Add(new OnnxTensor { Name = "bias", ElementType = TensorElementType.Float, Dims = new[] { 4 }, Data = new float[] { 1f, 2f, 3f, 4f } });
        Dictionary<string, object> NoAttrs() => new Dictionary<string, object>();
        mp.Nodes.Add(new OnnxNode { OpType = "Reshape", Inputs = new[] { "A", "S" }, Outputs = new[] { "t1" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Transpose", Inputs = new[] { "t1" }, Outputs = new[] { "t2" }, Attributes = new Dictionary<string, object> { ["perm"] = new long[] { 1, 0 } } });
        mp.Nodes.Add(new OnnxNode { OpType = "Add", Inputs = new[] { "x", "bias" }, Outputs = new[] { "s" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Gelu", Inputs = new[] { "s" }, Outputs = new[] { "y" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "MatMul", Inputs = new[] { "t2", "x" }, Outputs = new[] { "w" }, Attributes = NoAttrs() });
        return mp;
    }

    static string Dump(ComputationalGraph graph) =>
        string.Join(";", graph.Nodes.Select(n => n.Op + "(" + string.Join(",", n.Inputs ?? new string[0]) + "->" + string.Join(",", n.Outputs ?? new string[0]) + ")"))
        + "|" + string.Join(",", graph.Initializers.Keys.OrderBy(k => k));

    [Fact]
    public void SequentialLoads_AgreeExactly()
    {
        var a = Model.Load(PipelineModel())!;
        var b = Model.Load(PipelineModel())!;
        Assert.Equal(Dump(a), Dump(b));
        Assert.Contains(a.Nodes, n => n.Op == OpType.BiasGelu);
    }

    [Fact]
    public async Task ConcurrentLoads_AgreeExactly()
    {
        var tasks = new Task<string>[8];
        for (int i = 0; i < tasks.Length; i++)
            tasks[i] = Task.Run(() => Dump(Model.Load(PipelineModel())!));
        var dumps = await Task.WhenAll(tasks);
        foreach (var d in dumps) Assert.Equal(dumps[0], d);
    }

    [Fact]
    public void StandardPassSet_IsComplete()
    {
        Optimization.GraphOptimizer.EnsureStandardPasses();
        foreach (var name in new[] { "layernorm", "rope", "gelu", "gelu-tanh", "convrelu", "addrelu", "biasgelu", "gemmgelu", "constfold", "reshape-zerocopy" })
            Assert.Contains(Optimization.GraphOptimizer.Passes, p => p.Name == name);
    }
}
