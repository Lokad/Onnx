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

        var values = ModelFixture.CheckedOutput(graph, "last_hidden_state");
        ModelFixture.AssertMean(values, -0.0071360121, 1e-6, "dinov3");
        ModelFixture.AssertSpots(values,
            new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 384, 385, 1000, 10000, 38592, 77183 },
            new float[] { -0.26056969f, 0.58470774f, 0.2459141f, -1.24236286f, 0.56767243f, 0.06302299f, 0.60508633f, 0.16997504f, -0.99009454f, 0.38251263f, -0.11045331f, -0.14392422f, 0.36077532f, -0.59126061f },
            1e-4, "dinov3");
    }
}
