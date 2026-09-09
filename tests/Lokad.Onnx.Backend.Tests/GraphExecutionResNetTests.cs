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

    [SkippableFact]
    public void ResNetOutput_MatchesFrozenBitHash()
    {
        Skip.If(!System.Runtime.Intrinsics.X86.Fma.IsSupported, "Bit-exact hash requires x86 FMA.");
        var graph = ModelFixture.LoadRequiredModel("ResNet50", "models", "resnet50-onnx", "model.onnx");
        var input = DenseTensor<float>.OfShape(1, 3, 224, 224);
        input.Fill(0.5f);
        ModelFixture.AssertExecuted(graph, graph.Execute(new ITensor[] { input }, true));
        // The mean binds this run to the ORT 1.29 reference above; the hash
        // then freezes every output bit, so a defect in any unsampled element fails.
        var values = ModelFixture.CheckedOutput(graph, "output");
        ModelFixture.AssertMean(values, 0.0070858793, 1e-6, "resnet50");
        Assert.Equal(15780008002253672869UL, HashBits(values));
    }

    static ulong HashBits(float[] values)
    {
        ulong h = 1469598103934665603UL;
        foreach (var v in values)
        {
            h ^= (ulong)(uint)System.BitConverter.SingleToInt32Bits(v);
            h *= 1099511628211UL;
        }
        return h;
    }
}
