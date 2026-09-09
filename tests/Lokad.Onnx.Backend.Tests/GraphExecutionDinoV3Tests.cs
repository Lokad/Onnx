namespace Lokad.Onnx.Backend.Tests;

public class GraphExecutionDinoV3Tests
{
    [SkippableFact]
    public void CanInferWithDinoV3Small()
    {
        var graph = ModelFixture.LoadRequiredModel("DINOv3", "models", "dinov3-vits16", "onnx", "model.onnx");

        var input = DenseTensor<float>.OfShape(1, 3, 224, 224);
        input.Fill(0.5f);

        ModelFixture.AssertExecuted(graph, graph.Execute(new ITensor[] { input }, true));
        var output = (Tensor<float>)graph.Outputs["last_hidden_state"];
        Assert.Equal(new[] { 1, 201, 384 }, output.Dimensions.ToArray());

        // Reference for every oracle below: python onnxruntime 1.29.0, CPU
        // only, sequential execution, intra-op 1, inter-op 1, ORT_ENABLE_ALL.
        // The probe reproduces the frozen constant-input first-output oracle
        // exactly, which proves the harness replicates these inputs faithfully.
        var values = ModelFixture.CheckedOutput(graph, "last_hidden_state");
        ModelFixture.AssertMean(values, -0.0071360121, 1e-6, "dinov3");
        ModelFixture.AssertSpots(values,
            new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 384, 385, 1000, 10000, 38592, 77183 },
            new float[] { -0.26056969f, 0.58470774f, 0.2459141f, -1.24236286f, 0.56767243f, 0.06302299f, 0.60508633f, 0.16997504f, -0.99009454f, 0.38251263f, -0.11045331f, -0.14392422f, 0.36077532f, -0.59126061f },
            1e-4, "dinov3");
        var pooler = (Tensor<float>)graph.Outputs["pooler_output"];
        Assert.Equal(new[] { 1, 384 }, pooler.Dimensions.ToArray());
        var poolerValues = ModelFixture.CheckedOutput(graph, "pooler_output");
        ModelFixture.AssertMean(poolerValues, 0.00842788, 1e-6, "dinov3 pooler");
        ModelFixture.AssertSpots(poolerValues,
            new int[] { 0, 1, 2, 3, 7, 100, 200, 300, 383 },
            new float[] { -0.26056969f, 0.58470774f, 0.24591410f, -1.24236286f, 0.16997504f, -0.51840544f, 0.83507681f, -0.03455426f, -0.13659120f },
            1e-4, "dinov3 pooler");
    }

    [SkippableFact]
    public void CanInferWithDinoV3RampInput()
    {
        var graph = ModelFixture.LoadRequiredModel("DINOv3", "models", "dinov3-vits16", "onnx", "model.onnx");

        var input = RampImage();
        ModelFixture.AssertExecuted(graph, graph.Execute(new ITensor[] { input }, true));
        var output = (Tensor<float>)graph.Outputs["last_hidden_state"];
        Assert.Equal(new[] { 1, 201, 384 }, output.Dimensions.ToArray());
        var values = ModelFixture.CheckedOutput(graph, "last_hidden_state");
        ModelFixture.AssertMean(values, -0.00475693, 1e-6, "dinov3 ramp");
        var spots = new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 384, 385, 1000, 10000, 38592, 77183 };
        ModelFixture.AssertSpots(values, spots,
            new float[] { 0.21765247f, -0.53260744f, -0.11590165f, -0.89731288f, 0.18587849f, -0.20586464f, 0.96187025f, 0.35831058f, -0.96956730f, 0.11185741f, 0.07871664f, -0.24873462f, -0.15325923f, -0.45006031f },
            1e-4, "dinov3 ramp");
        var poolerValues = ModelFixture.CheckedOutput(graph, "pooler_output");
        ModelFixture.AssertMean(poolerValues, 0.00601039, 1e-6, "dinov3 ramp pooler");
        ModelFixture.AssertSpots(poolerValues,
            new int[] { 0, 1, 2, 3, 7, 100, 200, 300, 383 },
            new float[] { 0.21765247f, -0.53260744f, -0.11590165f, -0.89731288f, 0.35831058f, 0.37874156f, -0.11354458f, -0.12819622f, -0.07630807f },
            1e-4, "dinov3 ramp pooler");
        // The varied input must produce a visibly different output than the
        // frozen constant-input oracle; identical values would prove the
        // input is ignored.
        var constant = new float[] { -0.26056969f, 0.58470774f, 0.2459141f, -1.24236286f, 0.56767243f, 0.06302299f, 0.60508633f, 0.16997504f, -0.99009454f, 0.38251263f, -0.11045331f, -0.14392422f, 0.36077532f, -0.59126061f };
        double maxDiff = 0;
        for (int i = 0; i < spots.Length; i++)
        {
            double diff = System.Math.Abs(values[spots[i]] - constant[i]);
            if (diff > maxDiff) maxDiff = diff;
        }
        Assert.True(maxDiff > 0.1, "Ramp input barely moved the DINOv3 output (max diff " + maxDiff + ").");
    }

    static DenseTensor<float> RampImage()
    {
        // Bit-exact on every runtime: division by 256 is exact in binary
        // floating point, matching the python reference input bit for bit.
        var input = DenseTensor<float>.OfShape(1, 3, 224, 224);
        var span = input.Buffer.Span;
        for (int i = 0; i < span.Length; i++) span[i] = (i % 256) / 256.0f;
        return input;
    }
}
