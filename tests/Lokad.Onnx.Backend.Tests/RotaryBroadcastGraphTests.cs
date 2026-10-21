using System.Collections.Generic;
using System.Linq;

namespace Lokad.Onnx.Backend.Tests;

public class RotaryBroadcastGraphTests
{
    static OnnxModel Graph(bool blockFusion)
    {
        var model = new OnnxModel { Name = "broadcast-rope" };
        model.Opset[""] = 14;
        model.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        model.Inputs.Add(new OnnxValueInfo { Name = "freq", ElementType = TensorElementType.Float, Dims = new[] { 2, 1 } });
        model.Outputs.Add(new OnnxValueInfo { Name = "z", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        if (blockFusion) model.Outputs.Add(new OnnxValueInfo { Name = "m0", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        foreach (var (name, value) in new[] { ("s0", 0L), ("e0", 2L), ("s1", 2L), ("e1", long.MaxValue), ("axis", 1L), ("step", 1L) })
            model.Initializers.Add(new OnnxTensor { Name = name, ElementType = TensorElementType.Int64, Dims = new[] { 1 }, Data = new[] { value } });
        void Add(string op, string[] inputs, string output, Dictionary<string, object>? attributes = null)
            => model.Nodes.Add(new OnnxNode { OpType = op, Domain = "", Inputs = inputs, Outputs = new[] { output }, Attributes = attributes ?? new() });
        Add("Cos", new[] { "freq" }, "co");
        Add("Sin", new[] { "freq" }, "si");
        Add("Slice", new[] { "x", "s0", "e0", "axis", "step" }, "x0");
        Add("Slice", new[] { "x", "s1", "e1", "axis", "step" }, "x1");
        Add("Neg", new[] { "x1" }, "nx1");
        Add("Concat", new[] { "nx1", "x0" }, "rot", new() { ["axis"] = 1L });
        Add("Mul", new[] { "x", "co" }, "m0");
        Add("Mul", new[] { "rot", "si" }, "m1");
        Add("Add", new[] { "m0", "m1" }, "z");
        return model;
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FusedBroadcast_AgreesWithUnfusedGraph(bool disablePool)
    {
        var results = new List<float[]>();
        foreach (bool blocked in new[] { false, true })
        {
            var graph = Model.Load(Graph(blocked));
            Assert.Equal(blocked ? 0 : 1, graph.Nodes.Count(n => n.Op == OpType.RotaryEmbedding));
            var options = ExecutionOptions.Default with { Tensor = TensorExecutionOptions.Auto with { DisableBufferPool = disablePool } };
            Assert.True(graph.Execute(new Dictionary<string, ITensor>
            {
                ["x"] = new DenseTensor<float>(new float[] { 1, 2, 3, 4, 5, 6, 7, 8 }, new[] { 2, 4 }),
                ["freq"] = new DenseTensor<float>(new float[] { .1f, .2f }, new[] { 2, 1 })
            }, true, ExecutionProvider.CPU, options), graph.LastErrorMessage);
            results.Add(((Tensor<float>)graph.Outputs["z"]).ToArray());
        }
        Assert.Equal(results[1], results[0]);
    }
}
