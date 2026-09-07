using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Lokad.Onnx.Backend.Tests;

public class GraphExecutionGpt2Tests
{
    [SkippableFact]
    public void CanInferWithGpt2()
    {
        var modelPath = FindModelPath();
        if (modelPath is null && System.Environment.GetEnvironmentVariable("LOKAD_ONNX_RUN_LOCAL_MODEL_TESTS") == "1")
        {
            Assert.Fail("GPT-2 model requested via LOKAD_ONNX_RUN_LOCAL_MODEL_TESTS=1 but not found at models/gpt2-onnx/onnx/model.onnx.");
        }
        Skip.If(modelPath is null, "GPT-2 model not present; set LOKAD_ONNX_RUN_LOCAL_MODEL_TESTS=1 to require it.");

        var graph = OnnxImport.Load(modelPath);
        Assert.NotNull(graph);

        var inputs = new Dictionary<string, ITensor>();
        var ids = DenseTensor<long>.OfShape(1, 4);
        ids[0, 0] = 15496; ids[0, 1] = 11; ids[0, 2] = 314; ids[0, 3] = 716;
        inputs["input_ids"] = ids;
        var mask = DenseTensor<long>.OfShape(1, 4);
        mask.Fill(1);
        inputs["attention_mask"] = mask;
        var pos = DenseTensor<long>.OfShape(1, 4);
        pos[0, 0] = 0; pos[0, 1] = 1; pos[0, 2] = 2; pos[0, 3] = 3;
        inputs["position_ids"] = pos;
        for (int layer = 0; layer < 12; layer++)
        {
            inputs["past_key_values." + layer + ".key"] = DenseTensor<float>.OfShape(1, 12, 0, 64);
            inputs["past_key_values." + layer + ".value"] = DenseTensor<float>.OfShape(1, 12, 0, 64);
        }

        var ok = graph!.Execute(inputs, true);
        if (!ok)
        {
            var details = $"Failure at node '{graph.LastFailedNodeName}' ({graph.LastFailedNodeOp}): {graph.LastErrorMessage}";
            Assert.True(ok, details);
        }
        var output = (Tensor<float>)graph.Outputs["logits"];
        Assert.Equal(new[] { 1, 4, 50257 }, output.Dimensions.ToArray());

        var values = output.ToArray();
        foreach (var v in values)
        {
            Assert.False(float.IsNaN(v) || float.IsInfinity(v));
        }
        double sum = 0;
        foreach (var v in values) sum += v;
        Assert.True(Math.Abs(sum / values.Length - -111.89471436) < 1e-4, "Mean drift vs ORT reference.");
        int[] spots = new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 100, 384, 385, 1000, 10000, 50000 };
        float[] expected = new float[] { -35.23624420f, -35.32659531f, -38.97534943f, -39.39067459f, -37.65318298f, -38.67243576f, -36.02489090f, -36.48415375f, -43.31263733f, -40.13024902f, -38.99939346f, -40.74980927f, -41.98722839f, -44.11378098f };
        for (int i = 0; i < spots.Length; i++) Assert.True(Math.Abs(values[spots[i]] - expected[i]) < 1e-3, "Spot " + spots[i] + " drift vs ORT reference.");
    }

    static string? FindModelPath()
    {
        var dir = new DirectoryInfo(Directory.GetCurrentDirectory());
        while (dir is not null)
        {
            var candidate = Path.Combine(dir.FullName, "models", "gpt2-onnx", "onnx", "model.onnx");
            if (File.Exists(candidate))
            {
                return candidate;
            }
            dir = dir.Parent;
        }
        return null;
    }
}
