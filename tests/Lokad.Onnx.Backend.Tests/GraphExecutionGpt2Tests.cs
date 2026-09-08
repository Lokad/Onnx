using System.Collections.Generic;

namespace Lokad.Onnx.Backend.Tests;

public class GraphExecutionGpt2Tests
{
    static Dictionary<string, ITensor> FirstStepInputs()
    {
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
        return inputs;
    }

    [SkippableFact]
    public void CanInferWithGpt2()
    {
        var graph = ModelFixture.LoadRequiredModel("GPT-2", "models", "gpt2-onnx", "onnx", "model.onnx");

        ModelFixture.AssertExecuted(graph, graph.Execute(FirstStepInputs(), true));
        var output = (Tensor<float>)graph.Outputs["logits"];
        Assert.Equal(new[] { 1, 4, 50257 }, output.Dimensions.ToArray());

        var values = ModelFixture.CheckedOutput(graph, "logits");
        ModelFixture.AssertMean(values, -111.89471436, 1e-4, "gpt2");
        ModelFixture.AssertSpots(values,
            new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 100, 384, 385, 1000, 10000, 50000 },
            new float[] { -35.23624420f, -35.32659531f, -38.97534943f, -39.39067459f, -37.65318298f, -38.67243576f, -36.02489090f, -36.48415375f, -43.31263733f, -40.13024902f, -38.99939346f, -40.74980927f, -41.98722839f, -44.11378098f },
            1e-3, "gpt2");
    }

    [SkippableFact]
    public void Gpt2Continuation_WithNonemptyPast_StateAdvances()
    {
        var graph = ModelFixture.LoadRequiredModel("GPT-2", "models", "gpt2-onnx", "onnx", "model.onnx");

        ModelFixture.AssertExecuted(graph, graph.Execute(FirstStepInputs(), true));
        var first = (Tensor<float>)graph.Outputs["logits"];
        Assert.Equal(new[] { 1, 4, 50257 }, first.Dimensions.ToArray());
        var firstValues = ModelFixture.CheckedOutput(graph, "logits");

        var next = new Dictionary<string, ITensor>();
        var ids = DenseTensor<long>.OfShape(1, 1);
        ids[0, 0] = 317;
        next["input_ids"] = ids;
        var mask = DenseTensor<long>.OfShape(1, 5);
        mask.Fill(1);
        next["attention_mask"] = mask;
        var pos = DenseTensor<long>.OfShape(1, 1);
        pos[0, 0] = 4;
        next["position_ids"] = pos;
        for (int layer = 0; layer < 12; layer++)
        {
            var pastKey = (Tensor<float>)graph.Outputs["present." + layer + ".key"];
            var pastValue = (Tensor<float>)graph.Outputs["present." + layer + ".value"];
            Assert.Equal(new[] { 1, 12, 4, 64 }, pastKey.Dimensions.ToArray());
            Assert.Equal(new[] { 1, 12, 4, 64 }, pastValue.Dimensions.ToArray());
            next["past_key_values." + layer + ".key"] = pastKey;
            next["past_key_values." + layer + ".value"] = pastValue;
        }

        graph.Reset();
        ModelFixture.AssertExecuted(graph, graph.Execute(next, true));
        var logits = (Tensor<float>)graph.Outputs["logits"];
        Assert.Equal(new[] { 1, 1, 50257 }, logits.Dimensions.ToArray());
        var values = ModelFixture.CheckedOutput(graph, "logits");
        for (int layer = 0; layer < 12; layer++)
        {
            var pastKey = (Tensor<float>)graph.Outputs["present." + layer + ".key"];
            var pastValue = (Tensor<float>)graph.Outputs["present." + layer + ".value"];
            Assert.Equal(new[] { 1, 12, 5, 64 }, pastKey.Dimensions.ToArray());
            Assert.Equal(new[] { 1, 12, 5, 64 }, pastValue.Dimensions.ToArray());
            ModelFixture.CheckedOutput(graph, "present." + layer + ".key");
            ModelFixture.CheckedOutput(graph, "present." + layer + ".value");
        }
        // The new position id must move the logits: identical outputs would
        // prove the past state and position were ignored.
        bool moved = false;
        for (int i = 0; i < values.Length; i++)
        {
            if (values[i] != firstValues[3 * 50257 + i]) { moved = true; break; }
        }
        Assert.True(moved, "Continuation logits match the first-step last position exactly; past state had no effect.");
    }
}
