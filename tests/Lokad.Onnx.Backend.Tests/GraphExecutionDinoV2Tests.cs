namespace Lokad.Onnx.Backend.Tests;

public class GraphExecutionDinoV2Tests
{
    [SkippableFact]
    public void CanInferWithDinoV2Small()
    {
        var graph = ModelFixture.LoadRequiredModel("DINOv2", "models", "dinov2-small-onnx", "model.onnx");

        var input = DenseTensor<float>.OfShape(1, 3, 224, 224);
        input.Fill(0.5f);

        ModelFixture.AssertExecuted(graph, graph.Execute(new ITensor[] { input }, true));
        var output = (Tensor<float>)graph.Outputs.Values.First();
        Assert.Equal(new[] { 1, 257, 384 }, output.Dimensions.ToArray());

        var values = ModelFixture.CheckedFirstOutput(graph);
        ModelFixture.AssertMean(values, 0.0890405351, 1e-6, "dinov2");
        ModelFixture.AssertSpots(values,
            new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 384, 385, 1000, 10000, 49344, 98687 },
            new float[] { 2.78317356f, 2.0718112f, 0.99896085f, 0.58540159f, 1.33147788f, -0.90802592f, -0.7181195f, -1.58635163f, 2.34371638f, -2.9496913f, -0.32700467f, -1.35144138f, -0.95092082f, -2.83867621f },
            1e-3, "dinov2");
    }
}
