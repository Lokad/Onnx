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
}
