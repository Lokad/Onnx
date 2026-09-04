using System;
using System.IO;
using System.Linq;

namespace Lokad.Onnx.Backend.Tests;

public class GraphExecutionDinoV2Tests
{
    [SkippableFact]
    public void CanInferWithDinoV2Small()
    {
        var modelPath = FindModelPath();
        if (modelPath is null && System.Environment.GetEnvironmentVariable("LOKAD_ONNX_RUN_LOCAL_MODEL_TESTS") == "1")
        {
            Assert.Fail("DINOv2 model requested via LOKAD_ONNX_RUN_LOCAL_MODEL_TESTS=1 but not found at models/dinov2-small-onnx/model.onnx.");
        }
        Skip.If(modelPath is null, "DINOv2 model not present; set LOKAD_ONNX_RUN_LOCAL_MODEL_TESTS=1 to require it.");

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
        double sum = 0;
        foreach (var v in values) sum += v;
        Assert.True(Math.Abs(sum / values.Length - 0.0890405351) < 1e-6, "Mean drift vs ORT reference.");
        int[] spots = new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 384, 385, 1000, 10000, 49344, 98687 };
        float[] expected = new float[] { 2.78317356f, 2.0718112f, 0.99896085f, 0.58540159f, 1.33147788f, -0.90802592f, -0.7181195f, -1.58635163f, 2.34371638f, -2.9496913f, -0.32700467f, -1.35144138f, -0.95092082f, -2.83867621f };
        for (int i = 0; i < spots.Length; i++) Assert.True(Math.Abs(values[spots[i]] - expected[i]) < 1e-3, "Spot " + spots[i] + " drift vs ORT reference.");
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
