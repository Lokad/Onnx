namespace Lokad.Onnx.Backend.Tests;

using System.Reflection;

public class Community1ModelMetadataTests
{
    const string SegmentationHash = "af62796adfc46ab36fb27c183e7fe6530a745c665b64e949acd73bf01a18a31a";

    static OnnxModel Description()
    {
        OnnxSubgraph Branch() => new()
        {
            Outputs = { new OnnxValueInfo { Name = "122", ElementType = TensorElementType.Float, Dims = [-1, 80, -1], DimParams = ["batch", null, "frames"] } },
            Nodes = { new OnnxNode { Name = "forward", OpType = "Identity", Inputs = ["features"], Outputs = ["122"] } }
        };
        return new OnnxModel
        {
            Opset = { [""] = 13 },
            Inputs = { new OnnxValueInfo { Name = "features", ElementType = TensorElementType.Float, Dims = [1, 80, 7] } },
            Outputs = { new OnnxValueInfo { Name = "scores", ElementType = TensorElementType.Float, Dims = [-1, -1, 7], DimParams = ["batch", "frames", null] } },
            Initializers = { new OnnxTensor { Name = "condition", ElementType = TensorElementType.Bool, Dims = [], Data = new[] { true } } },
            Nodes = {
                new OnnxNode { Name = "If_117", OpType = "If", Inputs = ["condition"], Outputs = ["hidden"],
                    Attributes = new() { ["then_branch"] = Branch(), ["else_branch"] = Branch() } },
                new OnnxNode { Name = "reduce", OpType = "ReduceMean", Inputs = ["hidden"], Outputs = ["scores"],
                    Attributes = new() { ["axes"] = new long[] { 1 }, ["keepdims"] = 1L } }
            }
        };
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void OnlyPinnedSegmentationMetadataIsCorrected(bool pinned)
    {
        var input = new DenseTensor<float>(Enumerable.Repeat(1f, 80 * 7).ToArray(), [1, 80, 7]);
        var feed = new Dictionary<string, ITensor> { ["features"] = input };
        var original = Model.Load(Description());
        Assert.False(original.Execute(feed, true));
        Assert.Contains("Shared symbolic dimension frames", original.LastErrorMessage);

        var model = Description();
        var normalize = typeof(Community1Diarizer).GetMethod("NormalizeSegmentationSymbols", BindingFlags.NonPublic | BindingFlags.Static)!;
        normalize.Invoke(null, [model, pinned ? SegmentationHash : new string('0', 64)]);
        var graph = Model.Load(model);
        Assert.Equal(pinned, graph.Execute(feed, true));
        if (pinned)
        {
            Assert.Equal(new[] { 1, 1, 7 }, graph.Outputs["scores"].Dims);
            Assert.Equal(Enumerable.Repeat(1f, 7), ((Tensor<float>)graph.Outputs["scores"]).ToArray());
        }
        else Assert.Contains("Shared symbolic dimension frames", graph.LastErrorMessage);
        Assert.Equal(Enumerable.Repeat(1f, 80 * 7), input.ToArray());
    }
}
