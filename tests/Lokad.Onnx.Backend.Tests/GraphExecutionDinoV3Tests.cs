using System;
using System.IO;
using System.Linq;

namespace Lokad.Onnx.Backend.Tests;

public class GraphExecutionDinoV3Tests
{
    [SkippableFact]
    public void CanInferWithDinoV3Small()
    {
        var modelPath = FindModelPath();
        if (modelPath is null && System.Environment.GetEnvironmentVariable("LOKAD_ONNX_RUN_LOCAL_MODEL_TESTS") == "1")
        {
            Assert.Fail("DINOv3 model requested via LOKAD_ONNX_RUN_LOCAL_MODEL_TESTS=1 but not found at models/dinov3-vits16/onnx/model.onnx.");
        }
        Skip.If(modelPath is null, "DINOv3 model not present; set LOKAD_ONNX_RUN_LOCAL_MODEL_TESTS=1 to require it.");

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
        var output = (Tensor<float>)graph.Outputs["last_hidden_state"];
        Assert.Equal(new[] { 1, 201, 384 }, output.Dimensions.ToArray());

        var values = output.ToArray();
        foreach (var v in values)
        {
            Assert.False(float.IsNaN(v) || float.IsInfinity(v));
        }
        double sum = 0;
        foreach (var v in values) sum += v;
        Assert.True(Math.Abs(sum / values.Length - -0.0071360121) < 1e-6, "Mean drift vs ORT reference.");
        int[] spots = new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 384, 385, 1000, 10000, 38592, 77183 };
        float[] expected = new float[] { -0.26056969f, 0.58470774f, 0.2459141f, -1.24236286f, 0.56767243f, 0.06302299f, 0.60508633f, 0.16997504f, -0.99009454f, 0.38251263f, -0.11045331f, -0.14392422f, 0.36077532f, -0.59126061f };
        for (int i = 0; i < spots.Length; i++) Assert.True(Math.Abs(values[spots[i]] - expected[i]) < 1e-4, "Spot " + spots[i] + " drift vs ORT reference.");
    }

    static string? FindModelPath()
    {
        var dir = new DirectoryInfo(Directory.GetCurrentDirectory());
        while (dir is not null)
        {
            var candidate = Path.Combine(dir.FullName, "models", "dinov3-vits16", "onnx", "model.onnx");
            if (File.Exists(candidate))
            {
                return candidate;
            }
            dir = dir.Parent;
        }
        return null;
    }
}
