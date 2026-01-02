using System;
using System.IO;
using System.Linq;

namespace Lokad.Onnx.Backend.Tests;

public class GraphExecutionDinoV2Tests
{
    [Fact]
    public void CanInferWithDinoV2Small()
    {
        var modelPath = FindModelPath();
        if (modelPath is null)
        {
            return;
        }

        var graph = Model.Load(modelPath);
        Assert.NotNull(graph);

        var input = DenseTensor<float>.OfShape(1, 3, 224, 224);
        input.Fill(0.5f);

        var ok = graph!.Execute(new ITensor[] { input }, true);
        if (!ok)
        {
            var details = $"Failure at node '{graph.LastFailedNodeName}' ({graph.LastFailedNodeOp}): {graph.LastErrorMessage}";
            Assert.True(ok, details);
        }
        var output = (Tensor<float>)graph.Outputs.Values.First();
        Assert.Equal(new[] { 1, 257, 384 }, output.Dimensions.ToArray());

        var values = output.ToArray();
        foreach (var v in values)
        {
            Assert.False(float.IsNaN(v) || float.IsInfinity(v));
        }
    }

    static string? FindModelPath()
    {
        var dir = new DirectoryInfo(Directory.GetCurrentDirectory());
        while (dir is not null)
        {
            var candidate = Path.Combine(dir.FullName, "models", "dinov2-small-onnx", "model.onnx");
            if (File.Exists(candidate))
            {
                return candidate;
            }
            dir = dir.Parent;
        }
        return null;
    }
}
