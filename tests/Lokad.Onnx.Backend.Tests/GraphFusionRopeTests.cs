using System.Linq;

namespace Lokad.Onnx.Backend.Tests;

public class GraphFusionRopeTests
{


    [SkippableFact]
    public void DinoV3FusesRopeBranches()
    {
        var modelPath = ModelFixture.RequireModelOrSkip("DINOv3", "models", "dinov3-vits16", "onnx", "model.onnx");
        var graph = OnnxImport.Load(modelPath)!;
        Assert.Equal(24, graph.Nodes.Count(n => n.Op == OpType.RotaryEmbedding));
        foreach (var node in graph.Nodes.Where(n => n.Op == OpType.RotaryEmbedding))
        {
            Assert.Equal(3, node.Inputs.Length);
            Assert.Single(node.Outputs);
            Assert.True(node.RequiredInt("half") > 0);
        }
    }

    [SkippableFact]
    public void DinoV2HasNoRopePatterns()
    {
        var modelPath = ModelFixture.RequireModelOrSkip("DINOv2", "models", "dinov2-small-onnx", "model.onnx");
        var graph = OnnxImport.Load(modelPath)!;
        Assert.DoesNotContain(graph.Nodes, n => n.Op == OpType.RotaryEmbedding);
    }

    [SkippableFact]
    public void E5HasNoRopePatterns()
    {
        var modelPath = ModelFixture.RequireModelOrSkip("e5", "models", "multilingual-e5-small", "model.onnx");
        var graph = OnnxImport.Load(modelPath)!;
        Assert.DoesNotContain(graph.Nodes, n => n.Op == OpType.RotaryEmbedding);
    }
}
