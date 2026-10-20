namespace Lokad.Onnx.Backend.Tests;

public class GraphExecutionResNetTests
{
    [SkippableFact]
    public void CanInferWithResNet50()
    {
        var graph = ModelFixture.LoadRequiredModel("ResNet50", "models", "resnet50-onnx", "model.onnx");

        var input = DenseTensor<float>.OfShape(1, 3, 224, 224);
        input.Fill(0.5f);

        ModelFixture.AssertExecuted(graph, graph.Execute(new ITensor[] { input }, true));
        var output = (Tensor<float>)graph.Outputs["output"];
        Assert.Equal(new[] { 1, 2048 }, output.Dimensions.ToArray());

        var values = ModelFixture.CheckedOutput(graph, "output");
        ModelFixture.AssertMean(values, 0.0070858793, 1e-6, "resnet50");
        ModelFixture.AssertSpots(values,
            new int[] { 7, 11, 12, 44, 57, 64, 90, 103, 396, 960, 2046 },
            new float[] { 0.05441209f, 1.15632105f, 0.06200309f, 0.00150553f, 0.00077280f, 0.00054965f, 0.26585737f, 0.02300462f, 1.89254141f, 0.05548073f, 0.18742643f },
            1e-4, "resnet50");
    }

    [SkippableFact]
    public void CanInferWithResNet50RampInput()
    {
        var graph = ModelFixture.LoadRequiredModel("ResNet50", "models", "resnet50-onnx", "model.onnx");

        // Same bit-exact ramp as the DINOv3 varied-input test; reference
        // from python onnxruntime 1.29.0, CPU, sequential, intra-op 1,
        // inter-op 1, ORT_ENABLE_ALL.
        var input = DenseTensor<float>.OfShape(1, 3, 224, 224);
        var span = input.Buffer.Span;
        for (int i = 0; i < span.Length; i++) span[i] = (i % 256) / 256.0f;
        ModelFixture.AssertExecuted(graph, graph.Execute(new ITensor[] { input }, true));
        var output = (Tensor<float>)graph.Outputs["output"];
        Assert.Equal(new[] { 1, 2048 }, output.Dimensions.ToArray());
        var values = ModelFixture.CheckedOutput(graph, "output");
        ModelFixture.AssertMean(values, 0.02602991, 1e-6, "resnet50 ramp");
        var spots = new int[] { 7, 11, 12, 44, 57, 64, 90, 103, 396, 960, 2046 };
        ModelFixture.AssertSpots(values, spots,
            new float[] { 0.0f, 0.30972621f, 0.0f, 0.0f, 0.0f, 0.0f, 0.27978170f, 0.0f, 0.67714292f, 0.0f, 0.12580942f },
            1e-4, "resnet50 ramp");
        var constant = new float[] { 0.05441209f, 1.15632105f, 0.06200309f, 0.00150553f, 0.00077280f, 0.00054965f, 0.26585737f, 0.02300462f, 1.89254141f, 0.05548073f, 0.18742643f };
        double maxDiff = 0;
        for (int i = 0; i < spots.Length; i++)
        {
            double diff = System.Math.Abs(values[spots[i]] - constant[i]);
            if (diff > maxDiff) maxDiff = diff;
        }
        Assert.True(maxDiff > 0.1, "Ramp input barely moved the ResNet output (max diff " + maxDiff + ").");
    }

    [SkippableTheory]
    [InlineData("constant", "default")]
    [InlineData("constant", "scalar")]
    [InlineData("constant", "simd")]
    [InlineData("ramp", "default")]
    [InlineData("ramp", "scalar")]
    [InlineData("ramp", "simd")]
    public void ResNetOutput_MatchesCompleteNativeReference(string sample, string mode)
    {
        var graph = ModelFixture.LoadRequiredModel("ResNet50", "models", "resnet50-onnx", "model.onnx");
        var input = DenseTensor<float>.OfShape(1, 3, 224, 224);
        for (int i = 0; i < input.Buffer.Length; i++)
            input.Buffer.Span[i] = sample == "constant" ? .5f : (i % 256) / 256f;
        var inputSnapshot = input.ToArray();
        var options = mode == "scalar" ? ExecutionOptions.Scalar : mode == "simd" ? ExecutionOptions.Simd : ExecutionOptions.Default;
        ModelFixture.AssertExecuted(graph, graph.Execute(new ITensor[] { input }, true, ExecutionProvider.CPU, options));
        var values = ModelFixture.CheckedOutput(graph, "output");
        using var fixture = System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "fixtures", "resnet-ort.json")));
        var expected = fixture.RootElement.GetProperty("cases").GetProperty(sample).GetProperty("values").EnumerateArray().Select(v => v.GetSingle()).ToArray();
        Assert.Equal(2048, expected.Length);
        Assert.Equal(expected.Length, values.Length);
        for (int i = 0; i < values.Length; i++)
        {
            double error = Math.Abs((double)values[i] - expected[i]);
            double bound = 1e-5 * Math.Max(1, Math.Abs(expected[i]));
            Assert.True(float.IsFinite(values[i]) && error <= bound, $"{sample}/{mode}[{i}]: {values[i]:R} versus {expected[i]:R}, error {error:R}, bound {bound:R}");
        }
        var held = (Tensor<float>)graph.Outputs["output"]!;
        ModelFixture.AssertExecuted(graph, graph.Execute(new ITensor[] { input }, true, ExecutionProvider.CPU, options));
        Assert.Equal(values.Select(BitConverter.SingleToInt32Bits), ModelFixture.CheckedOutput(graph, "output").Select(BitConverter.SingleToInt32Bits));
        Assert.Equal(values, held.ToArray());
        Assert.Equal(inputSnapshot, input.ToArray());
    }
}
