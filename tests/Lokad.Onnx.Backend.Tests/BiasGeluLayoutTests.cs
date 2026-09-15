namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Guards the BiasGelu fast-path layout gate: a reversed-stride (column-major)
/// X must take the legacy two-step path, never the row-major pointer kernel.
/// Same bug class as the voice-branch SigmoidMul shape-lane fix.
/// </summary>
public class BiasGeluLayoutTests
{
    static DenseTensor<float> ReversedInput()
    {
        var xd = new DenseTensor<float>(new[] { 2, 8 }, true);
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 8; j++)
                xd[i, j] = i * 8 + j + 0.25f;
        return xd;
    }

    static float[] LegacyReference(DenseTensor<float> xd, DenseTensor<float> bias)
    {
        var add = CPUExecutionProvider.Add(xd, bias, null, null);
        Assert.Equal(OpStatus.Success, add.Status);
        var gelu = CPUExecutionProvider.Gelu(add.Outputs![0], null, null, null);
        Assert.Equal(OpStatus.Success, gelu.Status);
        return ((Tensor<float>)gelu.Outputs![0]).ToArray();
    }

    static DenseTensor<float> ScalarInput(float v)
    {
        var t = DenseTensor<float>.OfShape();
        t.SetValue(0, v);
        return t;
    }

    [Fact]
    public void BiasGelu_ScalarInputFallsBackToLegacy()
    {
        // S03: scalar x with rank-one bias must not reach Dimensions[^1].
        var x = ScalarInput(2f);
        var bias = DenseTensor<float>.OfValues(new float[] { 1f });
        var r = CPUExecutionProvider.BiasGelu(x, bias, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var got = ((Tensor<float>)r.Outputs![0]).ToArray();
        Assert.Single(got);
        Assert.Equal(2.9959502f, got[0], 5);
    }

    static OnnxModel ScalarBiasGeluModel()
    {
        var mp = new OnnxModel { Name = "tiny-biasgelu-scalar" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new int[0] });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Float, Dims = new[] { 1 } });
        mp.Initializers.Add(new OnnxTensor { Name = "bias", ElementType = TensorElementType.Float, Dims = new[] { 1 }, Data = new float[] { 1f } });
        var NoAttrs = new Dictionary<string, object>();
        mp.Nodes.Add(new OnnxNode { OpType = "Add", Inputs = new[] { "x", "bias" }, Outputs = new[] { "s" }, Attributes = NoAttrs });
        mp.Nodes.Add(new OnnxNode { OpType = "Gelu", Inputs = new[] { "s" }, Outputs = new[] { "y" }, Attributes = NoAttrs });
        return mp;
    }

    [Fact]
    public void BiasGelu_ScalarGraphExecutes()
    {
        var graph = Model.Load(ScalarBiasGeluModel())!;
        var feed = new Dictionary<string, ITensor> { ["x"] = ScalarInput(2f) };
        Assert.True(graph.Execute(feed, true), graph.LastErrorMessage);
        var got = ((Tensor<float>)graph.Outputs["y"]).ToArray();
        Assert.Single(got);
        Assert.Equal(2.9959502f, got[0], 5);
    }

    [Fact]
    public void BiasGelu_ReversedInputMatchesLegacy()
    {
        var xd = ReversedInput();
        Assert.True(xd.IsReversedStride);
        var bias = DenseTensor<float>.OfValues(new float[] { 0f, 0.5f, -0.5f, 1f, -1f, 2f, -2f, 0.25f });
        var r = CPUExecutionProvider.BiasGelu(xd, bias, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var got = ((Tensor<float>)r.Outputs![0]).ToArray();
        var expected = LegacyReference(xd, bias);
        Assert.Equal(expected.Length, got.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(System.BitConverter.SingleToInt32Bits(expected[i]) == System.BitConverter.SingleToInt32Bits(got[i]),
                "differs at " + i + ": " + expected[i] + " vs " + got[i]);
    }
}
