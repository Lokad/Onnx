using System.IO;
using System.Linq;

namespace Lokad.Onnx.Backend.Tests;

public class GraphFusionRopeTests
{
    static string? FindModel(params string[] parts)
    {
        var dir = new DirectoryInfo(Directory.GetCurrentDirectory());
        while (dir is not null)
        {
            var candidate = Path.Combine(new[] { dir.FullName }.Concat(parts).ToArray());
            if (File.Exists(candidate)) return candidate;
            dir = dir.Parent;
        }
        return null;
    }

    [SkippableFact]
    public void DinoV3FusesRopeBranches()
    {
        var modelPath = FindModel("models", "dinov3-vits16", "onnx", "model.onnx");
        Skip.If(modelPath is null, "DINOv3 model not present.");
        var graph = OnnxImport.Load(modelPath)!;
        Assert.Equal(24, graph.Nodes.Count(n => n.Op == OpType.RotaryEmbedding));
        foreach (var node in graph.Nodes.Where(n => n.Op == OpType.RotaryEmbedding))
        {
            Assert.Equal(3, node.Inputs.Length);
            Assert.Equal(1, node.Outputs.Length);
            Assert.True(node.RequiredInt("half") > 0);
        }
    }

    [SkippableFact]
    public void DinoV2HasNoRopePatterns()
    {
        var modelPath = FindModel("models", "dinov2-small-onnx", "model.onnx");
        Skip.If(modelPath is null, "DINOv2 model not present.");
        var graph = OnnxImport.Load(modelPath)!;
        Assert.DoesNotContain(graph.Nodes, n => n.Op == OpType.RotaryEmbedding);
    }

    [SkippableFact]
    public void E5HasNoRopePatterns()
    {
        var modelPath = FindModel("models", "multilingual-e5-small", "model.onnx");
        Skip.If(modelPath is null, "e5 model not present.");
        var graph = OnnxImport.Load(modelPath)!;
        Assert.DoesNotContain(graph.Nodes, n => n.Op == OpType.RotaryEmbedding);
    }
}
