using System.Reflection;

namespace Lokad.Onnx.Backend.Tests;

[Collection("SequentialLogSink")]
public class ParakeetPositionPreparationTests
{
    static readonly Func<ComputationalGraph, int, int> Prepare = typeof(ParakeetTranscriber).Assembly
        .GetType("Lokad.Onnx.ParakeetPositionPreparation", true)!
        .GetMethod("TryPrepare", BindingFlags.Static | BindingFlags.NonPublic)!
        .CreateDelegate<Func<ComputationalGraph, int, int>>();
    static readonly DenseTensor<float> Positions = Source();
    static readonly DenseTensor<float> Weight = Matrix();
    static DenseTensor<float> Source()
    {
        var values = new float[9999 * 1024];
        for (int row = 4997; row < 5002; row++) for (int c = 0; c < 1024; c++)
            values[row * 1024 + c] = ((row - 4999) * 31 + c % 17) * .03125f;
        return new DenseTensor<float>(values, new[] { 1, 9999, 1024 });
    }
    static DenseTensor<float> Matrix()
    {
        var values = new float[1024 * 1024];
        for (int c = 0; c < 1024; c++) values[c * 1024 + c] = .5f;
        return new DenseTensor<float>(values, new[] { 1024, 1024 });
    }
    static ComputationalGraph Graph()
    {
        var graph = new ComputationalGraph(0);
        graph.Metadata["Name"] = "private-position-preparation";
        graph.Inputs["x"] = new DenseTensor<float>(new[] { 1, 3, 1024 });
        Node Add(OpType op, string name, string[] inputs, string output, Dictionary<string, object>? attributes = null)
        {
            var node = new Node { ID = graph.Nodes.Count + 1, Name = name, Op = op, Inputs = inputs, Outputs = new[] { output }, Attributes = attributes, OpsetVersion = 18 };
            graph.Nodes.Add(node); return node;
        }
        void Constant(string name, ITensor value) => Add(OpType.Constant, name, Array.Empty<string>(), name, new() { ["value"] = value });
        Constant("center", new DenseTensor<long>(new long[] { 5000 }, Array.Empty<int>()));
        Constant("one", new DenseTensor<long>(new long[] { 1 }, Array.Empty<int>()));
        Constant("axis", new DenseTensor<long>(new long[] { 1 }, new[] { 1 }));
        Constant("unsqueeze-axis", new DenseTensor<long>(new long[] { 0 }, new[] { 1 }));
        Constant("positions", Positions);
        Add(OpType.Shape, "shape", new[] { "x" }, "shape");
        Add(OpType.Gather, "frames", new[] { "shape", "one" }, "frames", new() { ["axis"] = 0 });
        Add(OpType.Sub, "begin", new[] { "center", "frames" }, "begin");
        Add(OpType.Add, "plus", new[] { "center", "frames" }, "plus");
        Add(OpType.Sub, "end", new[] { "plus", "one" }, "end");
        Add(OpType.Unsqueeze, "starts", new[] { "begin", "unsqueeze-axis" }, "starts");
        Add(OpType.Unsqueeze, "ends", new[] { "end", "unsqueeze-axis" }, "ends");
        Add(OpType.Slice, "/pos_enc/Slice", new[] { "positions", "starts", "ends", "axis", "axis" }, "selected");
        Add(OpType.Shape, "selected-shape", new[] { "selected" }, "observed");
        graph.Outputs["observed"] = new DenseTensor<long>(new[] { 3 });
        for (int layer = 0; layer < 24; layer++)
        {
            string weight = "w" + layer, output = "y" + layer;
            graph.Initializers[weight] = Weight;
            Add(OpType.MatMul, $"/layers.{layer}/self_attn/linear_pos/MatMul", new[] { "selected", weight }, output);
            graph.Outputs[output] = new DenseTensor<float>(new[] { 1, 5, 1024 });
        }
        return graph;
    }

    [Fact]
    public void PrivatePreparationRetiresSourcesAndPreservesAllSupportedLengthsAndHeldOutputs()
    {
        var graph = Graph();
        Assert.Equal(24, Prepare(graph, 3));
        Assert.Equal(0, Prepare(graph, 3));
        Assert.DoesNotContain(graph.Nodes, n => n.Outputs.Contains("positions"));
        Assert.DoesNotContain(graph.Initializers.Keys, n => n.StartsWith("w", StringComparison.Ordinal));
        Assert.Equal(24L * 5 * 1024, graph.Initializers.Where(k => k.Key.StartsWith("__lokad.parakeet.position./layers.")).Sum(k => k.Value.Length));
        var held = new List<(Tensor<float> Output, float[] Values)>();
        foreach (int time in new[] { 1, 3, 2, 3, 1 })
        {
            var input = new DenseTensor<float>(new[] { 1, time, 1024 });
            graph.Inputs["x"] = input;
            var execution = graph.CreateExecution(ExecutionOptions.Memory);
            try
            {
                Assert.True(execution.Execute(new Dictionary<string, ITensor> { ["x"] = input }, true, ExecutionProvider.CPU, ExecutionOptions.Memory), execution.LastErrorMessage);
                Assert.Equal(new long[] { 1, 2 * time - 1, 1024 }, Assert.IsAssignableFrom<Tensor<long>>(execution.Outputs["observed"]).ToArray());
                for (int layer = 0; layer < 24; layer++)
                {
                    var actual = Assert.IsAssignableFrom<Tensor<float>>(execution.Outputs["y" + layer]);
                    Assert.Equal(new[] { 1, 2 * time - 1, 1024 }, actual.Dimensions.ToArray());
                    for (int row = 0; row < 2 * time - 1; row++) for (int c = 0; c < 1024; c++)
                        Assert.Equal(Positions.GetValue((5000 - time + row) * 1024 + c) * .5f, actual.GetValue(row * 1024 + c));
                    held.Add((actual, actual.ToArray()));
                }
            }
            finally { execution.Reset(); }
            foreach (var prior in held) Assert.Equal(prior.Values, prior.Output.ToArray());
        }
    }

    [Theory]
    [InlineData("bound")]
    [InlineData("axis")]
    [InlineData("visible")]
    [InlineData("extra-values")]
    [InlineData("weight-consumer")]
    [InlineData("foreign-domain")]
    public void UnknownExportsDeclineBeforeAnyGraphOrInitializerChange(string change)
    {
        var graph = Graph();
        int maximum = 3;
        if (change == "bound") maximum = 377;
        if (change == "axis") graph.Nodes.Single(n => n.Name == "axis").Attributes!["value"] = new DenseTensor<long>(new long[] { 2 }, new[] { 1 });
        if (change == "visible") graph.Outputs["selected"] = new DenseTensor<float>(new[] { 1, 5, 1024 });
        if (change is "extra-values" or "weight-consumer") graph.Nodes.Add(new Node { Name = "extra", Op = OpType.Identity,
            Inputs = new[] { change == "extra-values" ? "selected" : "w0" }, Outputs = new[] { "extra" } });
        if (change == "foreign-domain") { var node = graph.Nodes[^1]; node.Domain = "foreign"; graph.Nodes[^1] = node; }
        var nodes = graph.Nodes;
        var initializers = graph.Initializers.ToArray();
        Assert.Equal(0, Prepare(graph, maximum));
        Assert.Same(nodes, graph.Nodes);
        Assert.Equal(initializers, graph.Initializers.ToArray());
    }
}
