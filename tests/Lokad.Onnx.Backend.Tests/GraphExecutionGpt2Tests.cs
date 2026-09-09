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
        // Reference: python onnxruntime 1.29.0, CPU only, sequential execution,
        // intra-op 1, inter-op 1, ORT_ENABLE_ALL, fed with this exact two-step
        // sequence. The same probe reproduces the frozen prefill oracle above
        // exactly, which proves the harness replicates these inputs faithfully.
        ModelFixture.AssertMean(values, -83.02188873, 1e-4, "gpt2 continuation");
        ModelFixture.AssertSpots(values,
            new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 100, 384, 385, 1000, 10000, 50000 },
            new float[] { -75.79772949f, -76.21729279f, -77.27503967f, -75.84378052f, -78.30615234f, -75.83762360f, -75.81633759f, -76.28662109f, -84.51298523f, -80.63732910f, -76.93731689f, -78.58767700f, -81.93132782f, -86.75286102f },
            1e-3, "gpt2 continuation");
        var expectedKeyMeans = new double[] { 0.05062426, -0.02445106, -0.02396914, 0.06689094, -0.09357750, -0.01892482, -0.00484662, -0.03750284, -0.04111664, -0.01602590, 0.00261750, -0.01420080 };
        var expectedValueMeans = new double[] { 0.00690828, -0.00134631, -0.00664767, 0.02234010, -0.00779858, 0.03035101, 0.02179371, 0.00809788, -0.00381827, 0.00320051, -0.00997224, 0.00539252 };
        for (int layer = 0; layer < 12; layer++)
        {
            var pastKey = (Tensor<float>)graph.Outputs["present." + layer + ".key"];
            var pastValue = (Tensor<float>)graph.Outputs["present." + layer + ".value"];
            Assert.Equal(new[] { 1, 12, 5, 64 }, pastKey.Dimensions.ToArray());
            Assert.Equal(new[] { 1, 12, 5, 64 }, pastValue.Dimensions.ToArray());
            var keyValues = ModelFixture.CheckedOutput(graph, "present." + layer + ".key");
            var valueValues = ModelFixture.CheckedOutput(graph, "present." + layer + ".value");
            ModelFixture.AssertMean(keyValues, expectedKeyMeans[layer], 1e-4, "gpt2 present key " + layer);
            ModelFixture.AssertMean(valueValues, expectedValueMeans[layer], 1e-4, "gpt2 present value " + layer);
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

    static Dictionary<string, ITensor> NextStepInputs(ComputationalGraph graph, long tokenId, int positionId, int maskLength)
    {
        var next = new Dictionary<string, ITensor>();
        var ids = DenseTensor<long>.OfShape(1, 1);
        ids[0, 0] = tokenId;
        next["input_ids"] = ids;
        var mask = DenseTensor<long>.OfShape(1, maskLength);
        mask.Fill(1);
        next["attention_mask"] = mask;
        var pos = DenseTensor<long>.OfShape(1, 1);
        pos[0, 0] = positionId;
        next["position_ids"] = pos;
        for (int layer = 0; layer < 12; layer++)
        {
            next["past_key_values." + layer + ".key"] = (Tensor<float>)graph.Outputs["present." + layer + ".key"];
            next["past_key_values." + layer + ".value"] = (Tensor<float>)graph.Outputs["present." + layer + ".value"];
        }
        return next;
    }

    [SkippableFact]
    public void Gpt2SecondDecode_MatchesReference()
    {
        var graph = ModelFixture.LoadRequiredModel("GPT-2", "models", "gpt2-onnx", "onnx", "model.onnx");

        ModelFixture.AssertExecuted(graph, graph.Execute(FirstStepInputs(), true));
        var second = NextStepInputs(graph, 317, 4, 5);
        graph.Reset();
        ModelFixture.AssertExecuted(graph, graph.Execute(second, true));
        var third = NextStepInputs(graph, 13, 5, 6);
        graph.Reset();
        ModelFixture.AssertExecuted(graph, graph.Execute(third, true));
        var logits = (Tensor<float>)graph.Outputs["logits"];
        Assert.Equal(new[] { 1, 1, 50257 }, logits.Dimensions.ToArray());
        var values = ModelFixture.CheckedOutput(graph, "logits");
        // Reference: python onnxruntime 1.29.0, CPU only, sequential execution,
        // intra-op 1, inter-op 1, ORT_ENABLE_ALL, chaining this exact sequence.
        // Token 13 is the reference first-decode argmax on both sides, and the
        // probe reproduces the frozen second-decode oracle exactly.
        ModelFixture.AssertMean(values, -100.34924316, 1e-4, "gpt2 second decode");
        ModelFixture.AssertSpots(values,
            new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 100, 384, 385, 1000, 10000, 50000 },
            new float[] { -95.06102753f, -96.20240021f, -96.85228729f, -96.95389557f, -96.41825104f, -94.26221466f, -96.64118195f, -95.47057343f, -100.24946594f, -98.96081543f, -95.80052185f, -99.97512054f, -101.98978424f, -104.86552429f },
            1e-3, "gpt2 second decode");
        var expectedKeyMeans = new double[] { 0.04939841, -0.01536002, -0.02439982, 0.07019597, -0.10968190, -0.01138363, -0.00584086, -0.03299505, -0.04796230, -0.01823102, 0.00574899, -0.01341769 };
        var expectedValueMeans = new double[] { 0.00882331, 0.00262872, -0.00606723, 0.01956192, -0.00578579, 0.02334877, 0.01860573, -0.00232594, 0.00356014, 0.00725605, -0.01373047, 0.00984911 };
        for (int layer = 0; layer < 12; layer++)
        {
            var pastKey = (Tensor<float>)graph.Outputs["present." + layer + ".key"];
            var pastValue = (Tensor<float>)graph.Outputs["present." + layer + ".value"];
            Assert.Equal(new[] { 1, 12, 6, 64 }, pastKey.Dimensions.ToArray());
            Assert.Equal(new[] { 1, 12, 6, 64 }, pastValue.Dimensions.ToArray());
            var keyValues = ModelFixture.CheckedOutput(graph, "present." + layer + ".key");
            var valueValues = ModelFixture.CheckedOutput(graph, "present." + layer + ".value");
            ModelFixture.AssertMean(keyValues, expectedKeyMeans[layer], 1e-4, "gpt2 decode2 key " + layer);
            ModelFixture.AssertMean(valueValues, expectedValueMeans[layer], 1e-4, "gpt2 decode2 value " + layer);
        }
    }

    [SkippableFact]
    public void Gpt2LongerPrefill_MatchesReference()
    {
        var graph = ModelFixture.LoadRequiredModel("GPT-2", "models", "gpt2-onnx", "onnx", "model.onnx");

        var inputs = new Dictionary<string, ITensor>();
        var ids = DenseTensor<long>.OfShape(1, 8);
        ids[0, 0] = 15496; ids[0, 1] = 11; ids[0, 2] = 314; ids[0, 3] = 716;
        ids[0, 4] = 317; ids[0, 5] = 13; ids[0, 6] = 198; ids[0, 7] = 220;
        inputs["input_ids"] = ids;
        var mask = DenseTensor<long>.OfShape(1, 8);
        mask.Fill(1);
        inputs["attention_mask"] = mask;
        var pos = DenseTensor<long>.OfShape(1, 8);
        for (int i = 0; i < 8; i++) pos[0, i] = i;
        inputs["position_ids"] = pos;
        for (int layer = 0; layer < 12; layer++)
        {
            inputs["past_key_values." + layer + ".key"] = DenseTensor<float>.OfShape(1, 12, 0, 64);
            inputs["past_key_values." + layer + ".value"] = DenseTensor<float>.OfShape(1, 12, 0, 64);
        }
        ModelFixture.AssertExecuted(graph, graph.Execute(inputs, true));
        var output = (Tensor<float>)graph.Outputs["logits"];
        Assert.Equal(new[] { 1, 8, 50257 }, output.Dimensions.ToArray());
        var values = ModelFixture.CheckedOutput(graph, "logits");
        // Reference: python onnxruntime 1.29.0, CPU only, sequential execution,
        // intra-op 1, inter-op 1, ORT_ENABLE_ALL, on this exact eight-token input.
        ModelFixture.AssertMean(values, -105.76840210, 1e-4, "gpt2 prefill8");
        ModelFixture.AssertSpots(values,
            new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 100, 384, 385, 1000, 10000, 50000, 100000, 150000, 200000, 300000, 400000 },
            new float[] { -35.23624420f, -35.32659531f, -38.97534943f, -39.39067459f, -37.65318298f, -38.67243576f, -36.02489090f, -36.48415375f, -43.31263733f, -40.13024902f, -38.99939346f, -40.74980927f, -41.98722839f, -44.11378098f, -119.70864105f, -159.31016541f, -128.38629150f, -102.50579071f, -69.67440033f },
            1e-3, "gpt2 prefill8");
        var expectedKeyMeans = new double[] { 0.04311457, -0.01195455, -0.02591512, 0.08086473, -0.12117205, -0.02222601, -0.00511020, -0.02485690, -0.03139824, -0.00605219, -0.00398189, -0.01579665 };
        var expectedValueMeans = new double[] { 0.00853814, -0.00001035, -0.00830152, 0.02115173, -0.00504831, 0.01871932, 0.00264782, -0.00883885, -0.00289023, 0.00607003, -0.00831287, 0.01212665 };
        for (int layer = 0; layer < 12; layer++)
        {
            var pastKey = (Tensor<float>)graph.Outputs["present." + layer + ".key"];
            var pastValue = (Tensor<float>)graph.Outputs["present." + layer + ".value"];
            Assert.Equal(new[] { 1, 12, 8, 64 }, pastKey.Dimensions.ToArray());
            Assert.Equal(new[] { 1, 12, 8, 64 }, pastValue.Dimensions.ToArray());
            var keyValues = ModelFixture.CheckedOutput(graph, "present." + layer + ".key");
            var valueValues = ModelFixture.CheckedOutput(graph, "present." + layer + ".value");
            ModelFixture.AssertMean(keyValues, expectedKeyMeans[layer], 1e-4, "gpt2 prefill8 key " + layer);
            ModelFixture.AssertMean(valueValues, expectedValueMeans[layer], 1e-4, "gpt2 prefill8 value " + layer);
        }
    }

    [SkippableFact]
    public void Gpt2Continuation_ZeroedPast_Diverges()
    {
        var graph = ModelFixture.LoadRequiredModel("GPT-2", "models", "gpt2-onnx", "onnx", "model.onnx");

        ModelFixture.AssertExecuted(graph, graph.Execute(FirstStepInputs(), true));
        var real = new Dictionary<string, ITensor>();
        var zeroed = new Dictionary<string, ITensor>();
        var ids = DenseTensor<long>.OfShape(1, 1);
        ids[0, 0] = 317;
        var mask = DenseTensor<long>.OfShape(1, 5);
        mask.Fill(1);
        var pos = DenseTensor<long>.OfShape(1, 1);
        pos[0, 0] = 4;
        for (int layer = 0; layer < 12; layer++)
        {
            var pastKey = (Tensor<float>)graph.Outputs["present." + layer + ".key"];
            var pastValue = (Tensor<float>)graph.Outputs["present." + layer + ".value"];
            real["past_key_values." + layer + ".key"] = pastKey;
            real["past_key_values." + layer + ".value"] = pastValue;
            zeroed["past_key_values." + layer + ".key"] = DenseTensor<float>.OfShape(1, 12, 4, 64);
            zeroed["past_key_values." + layer + ".value"] = DenseTensor<float>.OfShape(1, 12, 4, 64);
        }
        real["input_ids"] = ids;
        real["attention_mask"] = mask;
        real["position_ids"] = pos;
        zeroed["input_ids"] = ids;
        zeroed["attention_mask"] = mask;
        zeroed["position_ids"] = pos;

        graph.Reset();
        ModelFixture.AssertExecuted(graph, graph.Execute(real, true));
        var expected = ((Tensor<float>)graph.Outputs["logits"]).ToArray();
        graph.Reset();
        ModelFixture.AssertExecuted(graph, graph.Execute(zeroed, true));
        var actual = ModelFixture.CheckedOutput(graph, "logits");
        Assert.Equal(expected.Length, actual.Length);
        double maxDiff = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double diff = System.Math.Abs(expected[i] - actual[i]);
            if (diff > maxDiff) maxDiff = diff;
        }
        // ORT moves by 41.7 here; anything near zero would prove the past is ignored.
        Assert.True(maxDiff > 1.0, "Zeroed past barely moved the decode output (max diff " + maxDiff + ").");
    }
}
