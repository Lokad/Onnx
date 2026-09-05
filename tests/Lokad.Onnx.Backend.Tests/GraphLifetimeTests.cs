using System.IO;
using System.Linq;

namespace Lokad.Onnx.Backend.Tests;

public class GraphLifetimeTests
{
    static ComputationalGraph TinyGraph()
    {
        var graph = new ComputationalGraph();
        graph.Inputs["x"] = DenseTensor<float>.OfValues(new float[] { 1f, 2f });
        graph.Initializers["w"] = DenseTensor<float>.OfValues(new float[] { 2f, 3f });
        graph.Outputs["z"] = DenseTensor<float>.OfShape(2);
        graph.Nodes.Add(new Node { Name = "add", Op = OpType.Add, Inputs = new[] { "x", "w" }, Outputs = new[] { "t" } });
        graph.Nodes.Add(new Node { Name = "mul", Op = OpType.Mul, Inputs = new[] { "t", "w" }, Outputs = new[] { "z" } });
        graph.Nodes.Add(new Node { Name = "relu", Op = OpType.Relu, Inputs = new[] { "t" }, Outputs = new[] { "dead" } });
        graph.RefreshLifetimeAnalysis();
        return graph;
    }

    [Fact]
    public void Lifetime_PinsSmallGraph()
    {
        var graph = TinyGraph();
        Assert.Equal(2, graph.LastUseIndex["t"]);
        Assert.Equal(int.MaxValue, graph.LastUseIndex["x"]);
        Assert.Equal(int.MaxValue, graph.LastUseIndex["w"]);
        Assert.Equal(3, graph.LastUseIndex["z"]);
        Assert.Equal(2, graph.LastUseIndex["dead"]);
    }

    [SkippableFact]
    public void Lifetime_PinsFusedE5Graph()
    {
        var modelPath = FindE5ModelPath();
        if (modelPath is null && System.Environment.GetEnvironmentVariable("LOKAD_ONNX_RUN_LOCAL_MODEL_TESTS") == "1")
        {
            Assert.Fail("e5 model requested via LOKAD_ONNX_RUN_LOCAL_MODEL_TESTS=1 but not found at models/multilingual-e5-small/model.onnx.");
        }
        Skip.If(modelPath is null, "e5 model not present; set LOKAD_ONNX_RUN_LOCAL_MODEL_TESTS=1 to require it.");

        var graph = Model.Load(modelPath)!;
        Assert.Equal(25, graph.Nodes.Count(n => n.Op == OpType.LayerNormalization));
        Assert.Equal(96, graph.Nodes.Count(n => n.Op == OpType.MatMul));
        Assert.Equal(graph.Nodes.Count, graph.LastUseIndex["last_hidden_state"]);
        foreach (var name in graph.Inputs.Keys) Assert.Equal(int.MaxValue, graph.LastUseIndex[name]);
        foreach (var name in graph.Initializers.Keys) Assert.Equal(int.MaxValue, graph.LastUseIndex[name]);
        foreach (var node in graph.Nodes)
        {
            foreach (var input in node.Inputs)
            {
                if (string.IsNullOrEmpty(input)) continue;
                Assert.True(graph.LastUseIndex.ContainsKey(input), "Missing lifetime for " + input);
            }
        }
    }

    static string? FindE5ModelPath()
    {
        var dir = new DirectoryInfo(Directory.GetCurrentDirectory());
        while (dir is not null)
        {
            var candidate = Path.Combine(dir.FullName, "models", "multilingual-e5-small", "model.onnx");
            if (File.Exists(candidate))
            {
                return candidate;
            }
            dir = dir.Parent;
        }
        return null;
    }
}
