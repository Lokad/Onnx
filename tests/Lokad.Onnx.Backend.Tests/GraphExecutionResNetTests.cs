using System;
using System.IO;
using System.Linq;

namespace Lokad.Onnx.Backend.Tests;

public class GraphExecutionResNetTests
{
    [SkippableFact]
    public void CanInferWithResNet50()
    {
        var modelPath = FindModelPath();
        if (modelPath is null && System.Environment.GetEnvironmentVariable("LOKAD_ONNX_RUN_LOCAL_MODEL_TESTS") == "1")
        {
            Assert.Fail("ResNet50 model requested via LOKAD_ONNX_RUN_LOCAL_MODEL_TESTS=1 but not found at models/resnet50-onnx/model.onnx.");
        }
        Skip.If(modelPath is null, "ResNet50 model not present; set LOKAD_ONNX_RUN_LOCAL_MODEL_TESTS=1 to require it.");

        var graph = OnnxImport.Load(modelPath);
        Assert.NotNull(graph);

        var input = DenseTensor<float>.OfShape(1, 3, 224, 224);
        input.Fill(0.5f);

        var ok = graph!.Execute(new ITensor[] { input }, true);
        if (!ok)
        {
            var details = $"Failure at node '{graph.LastFailedNodeName}' ({graph.LastFailedNodeOp}): {graph.LastErrorMessage}";
            Assert.True(ok, details);
        }
        var output = (Tensor<float>)graph.Outputs["output"];
        Assert.Equal(new[] { 1, 2048 }, output.Dimensions.ToArray());

        var values = output.ToArray();
        foreach (var v in values)
        {
            Assert.False(float.IsNaN(v) || float.IsInfinity(v));
        }
        double sum = 0;
        foreach (var v in values) sum += v;
        Assert.True(Math.Abs(sum / values.Length - 0.0070858793) < 1e-6, "Mean drift vs ORT reference.");
        int[] spots = new int[] { 7, 11, 12, 44, 57, 64, 90, 103, 396, 960, 2046 };
        float[] expected = new float[] { 0.05441209f, 1.15632105f, 0.06200309f, 0.00150553f, 0.00077280f, 0.00054965f, 0.26585737f, 0.02300462f, 1.89254141f, 0.05548073f, 0.18742643f };
        for (int i = 0; i < spots.Length; i++) Assert.True(Math.Abs(values[spots[i]] - expected[i]) < 1e-4, "Spot " + spots[i] + " drift vs ORT reference.");
    }

    static string? FindModelPath()
    {
        var dir = new DirectoryInfo(Directory.GetCurrentDirectory());
        while (dir is not null)
        {
            var candidate = Path.Combine(dir.FullName, "models", "resnet50-onnx", "model.onnx");
            if (File.Exists(candidate))
            {
                return candidate;
            }
            dir = dir.Parent;
        }
        return null;
    }
}
