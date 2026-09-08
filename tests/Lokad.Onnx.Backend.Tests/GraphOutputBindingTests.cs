using System.Collections.Generic;
using System.Linq;

namespace Lokad.Onnx.Backend.Tests;

// R3: every run resolves its outputs from the immutable OutputDescs
// declarations. Failures keep previous values inaccessible, retries work
// without Reset, and dangling outputs fail instead of reporting success.
public class GraphOutputBindingTests
{
    static OnnxValueInfo IO(string name) =>
        new OnnxValueInfo { Name = name, ElementType = TensorElementType.Float, Dims = new[] { 2 } };

    static OnnxModel PassthroughModel()
    {
        var mp = new OnnxModel { Name = "passthrough" };
        mp.Opset[""] = 11;
        mp.Inputs.Add(IO("x"));
        mp.Outputs.Add(IO("x"));
        return mp;
    }

    static Dictionary<string, ITensor> GoodX() =>
        new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[] { 1f, 2f }) } };

    [Fact]
    public void Passthrough_RetryWithoutReset_SucceedsWithOneOutput()
    {
        var graph = Model.Load(PassthroughModel())!;
        Assert.True(graph.Execute(GoodX(), true));
        Assert.Equal(new float[] { 1f, 2f }, ((Tensor<float>)graph.Outputs["x"]).ToArray());

        Assert.False(graph.Execute(new Dictionary<string, ITensor>(), true));
        Assert.NotNull(graph.LastErrorMessage);
        Assert.Empty(graph.Outputs);

        Assert.True(graph.Execute(GoodX(), true));
        Assert.Single(graph.Outputs);
        Assert.Equal(new float[] { 1f, 2f }, ((Tensor<float>)graph.Outputs["x"]).ToArray());
    }

    [Fact]
    public void InitializerPassthrough_RetryWithoutReset()
    {
        var mp = new OnnxModel { Name = "init-passthrough" };
        mp.Opset[""] = 11;
        mp.Initializers.Add(new OnnxTensor
        {
            Name = "w", ElementType = TensorElementType.Float, Dims = new[] { 2 },
            Data = new float[] { 3f, 4f },
        });
        mp.Outputs.Add(IO("w"));
        var graph = Model.Load(mp)!;
        var none = new Dictionary<string, ITensor>();
        Assert.True(graph.Execute(none, true));
        Assert.Equal(new float[] { 3f, 4f }, ((Tensor<float>)graph.Outputs["w"]).ToArray());

        var extra = new Dictionary<string, ITensor> { { "nope", DenseTensor<float>.OfValues(new float[] { 0f }) } };
        Assert.False(graph.Execute(extra, true));
        Assert.Empty(graph.Outputs);

        Assert.True(graph.Execute(none, true));
        Assert.Single(graph.Outputs);
        Assert.Equal(new float[] { 3f, 4f }, ((Tensor<float>)graph.Outputs["w"]).ToArray());
    }

    [Fact]
    public void EmptyGraph_NoOutputs_SucceedsVacuously()
    {
        var mp = new OnnxModel { Name = "empty" };
        mp.Opset[""] = 11;
        var graph = Model.Load(mp)!;
        Assert.True(graph.Execute(new Dictionary<string, ITensor>(), true));
        Assert.Empty(graph.Outputs);
    }

    [Fact]
    public void DanglingOutput_FailsNamingIt()
    {
        var mp = new OnnxModel { Name = "dangling" };
        mp.Opset[""] = 11;
        mp.Inputs.Add(IO("x"));
        mp.Outputs.Add(IO("z"));
        var graph = Model.Load(mp)!;
        Assert.False(graph.Execute(GoodX(), true));
        Assert.Contains("z", graph.LastErrorMessage ?? "");
        Assert.Empty(graph.Outputs);
    }

    [Fact]
    public void NodeProduced_RetryWithoutReset_KeepsWorking()
    {
        var mp = new OnnxModel { Name = "relu" };
        mp.Opset[""] = 11;
        mp.Inputs.Add(IO("x"));
        mp.Outputs.Add(IO("y"));
        mp.Nodes.Add(new OnnxNode
        {
            Name = "r", OpType = "Relu", Inputs = new[] { "x" }, Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object>(),
        });
        var graph = Model.Load(mp)!;
        var good = new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[] { -1f, 2f }) } };
        Assert.True(graph.Execute(good, true));
        Assert.Equal(new float[] { 0f, 2f }, ((Tensor<float>)graph.Outputs["y"]).ToArray());
        Assert.False(graph.Execute(new Dictionary<string, ITensor>(), true));
        Assert.Empty(graph.Outputs);
        Assert.True(graph.Execute(good, true));
        Assert.Equal(new float[] { 0f, 2f }, ((Tensor<float>)graph.Outputs["y"]).ToArray());
    }

    [Fact]
    public void MixedNodeAndPassthrough_Retry()
    {
        var mp = new OnnxModel { Name = "mixed" };
        mp.Opset[""] = 11;
        mp.Inputs.Add(IO("x"));
        mp.Outputs.Add(IO("y"));
        mp.Outputs.Add(IO("x"));
        mp.Nodes.Add(new OnnxNode
        {
            Name = "r", OpType = "Relu", Inputs = new[] { "x" }, Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object>(),
        });
        var graph = Model.Load(mp)!;
        var good = new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[] { -1f, 2f }) } };
        Assert.True(graph.Execute(good, true));
        Assert.Equal(2, graph.Outputs.Count);
        Assert.False(graph.Execute(new Dictionary<string, ITensor>(), true));
        Assert.Empty(graph.Outputs);
        Assert.True(graph.Execute(good, true));
        Assert.Equal(2, graph.Outputs.Count);
        Assert.Equal(new float[] { 0f, 2f }, ((Tensor<float>)graph.Outputs["y"]).ToArray());
        Assert.Equal(new float[] { -1f, 2f }, ((Tensor<float>)graph.Outputs["x"]).ToArray());
    }

    [Fact]
    public void Reset_RestoresDeclaredKeysAsDescriptors()
    {
        var graph = Model.Load(PassthroughModel())!;
        Assert.True(graph.Execute(GoodX(), true));
        Assert.False(graph.Execute(new Dictionary<string, ITensor>(), true));
        Assert.Empty(graph.Outputs);
        graph.Reset();
        Assert.Equal(new[] { "x" }, graph.Outputs.Keys.OrderBy(k => k).ToArray());
        Assert.IsType<TensorDesc>(graph.Outputs["x"]);
        Assert.True(graph.Execute(GoodX(), true));
        Assert.Equal(new float[] { 1f, 2f }, ((Tensor<float>)graph.Outputs["x"]).ToArray());
    }

    [Fact]
    public void FreshContext_AfterFailure_Succeeds()
    {
        var graph = Model.Load(PassthroughModel())!;
        Assert.True(graph.Execute(GoodX(), true));
        Assert.False(graph.Execute(new Dictionary<string, ITensor>(), true));
        var ctx = graph.CreateExecution(null);
        Assert.True(ctx.Execute(GoodX(), true));
        Assert.Single(ctx.Outputs);
        Assert.Equal(new float[] { 1f, 2f }, ((Tensor<float>)ctx.Outputs["x"]).ToArray());
    }

    [Fact]
    public void PreviousValues_InaccessibleAfterFailure()
    {
        var graph = Model.Load(PassthroughModel())!;
        Assert.True(graph.Execute(GoodX(), true));
        var first = ((Tensor<float>)graph.Outputs["x"]).ToArray();
        Assert.False(graph.Execute(new Dictionary<string, ITensor>(), true));
        Assert.Empty(graph.Outputs);
        Assert.NotNull(graph.LastErrorMessage);
        Assert.Equal(new float[] { 1f, 2f }, first);
    }
}
