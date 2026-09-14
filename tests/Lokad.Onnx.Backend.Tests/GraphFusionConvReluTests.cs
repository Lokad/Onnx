using System.Collections.Generic;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Covers the shared Conv+Relu fusion: single-consumer pairs fuse with a
/// bit-identical epilogue, while multi-consumer, graph-output, and rank-3
/// cases keep the unfused nodes.
/// </summary>
public class GraphFusionConvReluTests
{
    static OnnxValueInfo NamedIO(string name, int[] dims) =>
        new OnnxValueInfo { Name = name, ElementType = TensorElementType.Float, Dims = dims };

    static OnnxTensor FloatInit(string name, int[] dims, float[] values) =>
        new OnnxTensor { Name = name, ElementType = TensorElementType.Float, Dims = dims, Data = values };

    static OnnxNode Nod(string op, string[] inputs, string[] outputs) =>
        new OnnxNode { OpType = op, Inputs = inputs, Outputs = outputs, Attributes = new Dictionary<string, object>() };

    static OnnxModel ConvReluIdentityModel(int[] xDims, int[] wDims, float[] wValues, float[]? bValues, string outputName)
    {
        var mp = new OnnxModel { Name = "conv-relu" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(NamedIO("x", xDims));
        mp.Outputs.Add(NamedIO(outputName, xDims));
        mp.Initializers.Add(FloatInit("W", wDims, wValues));
        if (bValues is not null)
        {
            mp.Initializers.Add(FloatInit("B", new[] { wDims[0] }, bValues));
            mp.Nodes.Add(Nod("Conv", new[] { "x", "W", "B" }, new[] { "c" }));
        }
        else
        {
            mp.Nodes.Add(Nod("Conv", new[] { "x", "W" }, new[] { "c" }));
        }
        mp.Nodes.Add(Nod("Relu", new[] { "c" }, new[] { "r" }));
        mp.Nodes.Add(Nod("Identity", new[] { "r" }, new[] { outputName }));
        return mp;
    }

    [Fact]
    public void SingleConsumerPair_FusesWithEpilogue()
    {
        var mp = ConvReluIdentityModel(
            new[] { 1, 1, 2, 2 },
            new[] { 1, 1, 1, 1 },
            new[] { 2f },
            new[] { -3f },
            "z");
        var graph = Model.Load(mp)!;
        Assert.Equal(2, graph.Nodes.Count);
        var conv = graph.Nodes[0];
        Assert.Equal(OpType.Conv, conv.Op);
        Assert.NotNull(conv.Attributes);
        Assert.Equal(1, conv.GetInt("fuse_relu", null));
        Assert.Equal(OpType.Identity, graph.Nodes[1].Op);

        var x = DenseTensor<float>.OfValues(new float[1, 1, 2, 2] { { { { 1f, -1f }, { 0.5f, 3f } } } });
        var inputs = new Dictionary<string, ITensor> { { "x", x } };
        Assert.True(graph.Execute(inputs, true));
        var actual = ((Tensor<float>)graph.Outputs["z"]).ToArray();
        Assert.Equal(new float[] { 0f, 0f, 0f, 3f }, actual);
    }

    [Fact]
    public void MultiConsumerConv_SkipsFusion()
    {
        var mp = new OnnxModel { Name = "conv-multi" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(NamedIO("x", new[] { 1, 1, 2, 2 }));
        mp.Outputs.Add(NamedIO("z", new[] { 1, 1, 2, 2 }));
        mp.Initializers.Add(FloatInit("W", new[] { 1, 1, 1, 1 }, new[] { 1f }));
        mp.Initializers.Add(FloatInit("B", new[] { 1 }, new[] { 0f }));
        mp.Nodes.Add(Nod("Conv", new[] { "x", "W", "B" }, new[] { "c" }));
        mp.Nodes.Add(Nod("Relu", new[] { "c" }, new[] { "r" }));
        mp.Nodes.Add(Nod("Add", new[] { "r", "c" }, new[] { "z" }));
        var graph = Model.Load(mp)!;
        Assert.Equal(3, graph.Nodes.Count);
        Assert.Equal(OpType.Conv, graph.Nodes[0].Op);
        Assert.Null(graph.Nodes[0].GetInt("fuse_relu", null));
    }
    [Fact]
    public void ReluOutputAsGraphOutput_SkipsFusion()
    {
        var mp = new OnnxModel { Name = "conv-relu-out" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(NamedIO("x", new[] { 1, 1, 2, 2 }));
        mp.Outputs.Add(NamedIO("r", new[] { 1, 1, 2, 2 }));
        mp.Initializers.Add(FloatInit("W", new[] { 1, 1, 1, 1 }, new[] { 1f }));
        mp.Nodes.Add(Nod("Conv", new[] { "x", "W" }, new[] { "c" }));
        mp.Nodes.Add(Nod("Relu", new[] { "c" }, new[] { "r" }));
        var graph = Model.Load(mp)!;
        Assert.Equal(2, graph.Nodes.Count);
        Assert.Null(graph.Nodes[0].GetInt("fuse_relu", null));
    }

    [Fact]
    public void ConvOutputAsGraphOutput_SkipsFusion()
    {
        var mp = new OnnxModel { Name = "conv-out-kept" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(NamedIO("x", new[] { 1, 1, 2, 2 }));
        mp.Outputs.Add(NamedIO("c", new[] { 1, 1, 2, 2 }));
        mp.Outputs.Add(NamedIO("z", new[] { 1, 1, 2, 2 }));
        mp.Initializers.Add(FloatInit("W", new[] { 1, 1, 1, 1 }, new[] { 1f }));
        mp.Nodes.Add(Nod("Conv", new[] { "x", "W" }, new[] { "c" }));
        mp.Nodes.Add(Nod("Relu", new[] { "c" }, new[] { "r" }));
        mp.Nodes.Add(Nod("Identity", new[] { "r" }, new[] { "z" }));
        var graph = Model.Load(mp)!;
        Assert.Equal(3, graph.Nodes.Count);
        Assert.Null(graph.Nodes[0].GetInt("fuse_relu", null));
    }

    [Fact]
    public void Rank3Weights_SkipFusion()
    {
        var mp = new OnnxModel { Name = "conv1d-relu" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(NamedIO("x", new[] { 1, 1, 4 }));
        mp.Outputs.Add(NamedIO("z", new[] { 1, 1, 2 }));
        mp.Initializers.Add(FloatInit("W", new[] { 1, 1, 3 }, new[] { 1f, 0f, -1f }));
        mp.Nodes.Add(Nod("Conv", new[] { "x", "W" }, new[] { "c" }));
        mp.Nodes.Add(Nod("Relu", new[] { "c" }, new[] { "r" }));
        mp.Nodes.Add(Nod("Identity", new[] { "r" }, new[] { "z" }));
        var graph = Model.Load(mp)!;
        Assert.Equal(OpType.Conv, graph.Nodes[0].Op);
        Assert.Null(graph.Nodes[0].GetInt("fuse_relu", null));
    }

    static void AssertBitsEqual(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(
                System.BitConverter.SingleToInt32Bits(expected[i]),
                System.BitConverter.SingleToInt32Bits(actual[i]));
        }
    }

    static void AssertBitsEqualDouble(double[] expected, double[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(
                System.BitConverter.DoubleToInt64Bits(expected[i]),
                System.BitConverter.DoubleToInt64Bits(actual[i]));
        }
    }
    [Fact]
    public void FusedEpilogue_BitMatchesUnfused_GenericPath()
    {
        var x = DenseTensor<float>.OfValues(new float[1, 1, 2, 2] { { { { 1f, -2f }, { 0.5f, -0.0f } } } });
        var w = DenseTensor<float>.OfValues(new float[1, 1, 2, 2] { { { { 1f, 0f }, { 0f, -1f } } } });
        var b = DenseTensor<float>.OfValues(new float[] { 0.25f });
        var pads = new int[] { 0, 0, 0, 0 };
        var plain = Tensor<float>.Conv2D(x, w, 1, pads, b, null, new int[] { 1, 1 }, null, TensorExecutionOptions.Auto, false);
        var fused = Tensor<float>.Conv2D(x, w, 1, pads, b, null, new int[] { 1, 1 }, null, TensorExecutionOptions.Auto, true);
        var relu = Tensor<float>.Relu(plain, TensorExecutionOptions.Auto);
        AssertBitsEqual(relu.ToArray(), fused.ToArray());
    }

    [Fact]
    public void FusedEpilogue_BitMatchesUnfused_PointwiseNoBias()
    {
        var x = DenseTensor<float>.OfValues(new float[1, 2, 1, 2] { { { { -1f, 2f } }, { { 0f, -0.0f } } } });
        var w = DenseTensor<float>.OfValues(new float[2, 2, 1, 1] { { { { 1f } }, { { -2f } } }, { { { 0.5f } }, { { 1f } } } });
        var pads = new int[] { 0, 0, 0, 0 };
        var plain = Tensor<float>.Conv2D(x, w, 1, pads, null, null, new int[] { 1, 1 }, null, TensorExecutionOptions.Auto, false);
        var fused = Tensor<float>.Conv2D(x, w, 1, pads, null, null, new int[] { 1, 1 }, null, TensorExecutionOptions.Auto, true);
        var relu = Tensor<float>.Relu(plain, TensorExecutionOptions.Auto);
        AssertBitsEqual(relu.ToArray(), fused.ToArray());
    }

    [Fact]
    public void FusedEpilogue_PreservesSignedZeroAndNaN()
    {
        var x = DenseTensor<float>.OfValues(new float[1, 1, 2, 2] { { { { -0.0f, float.NaN }, { -1f, 0f } } } });
        var w = DenseTensor<float>.OfValues(new float[1, 1, 1, 1] { { { { 1f } } } });
        var pads = new int[] { 0, 0, 0, 0 };
        var fused = Tensor<float>.Conv2D(x, w, 1, pads, null, null, new int[] { 1, 1 }, null, TensorExecutionOptions.Auto, true);
        var plain = Tensor<float>.Conv2D(x, w, 1, pads, null, null, new int[] { 1, 1 }, null, TensorExecutionOptions.Auto, false);
        var relu = Tensor<float>.Relu(plain, TensorExecutionOptions.Auto);
        AssertBitsEqual(relu.ToArray(), fused.ToArray());
        // The 1x1 GEMM accumulates onto a zero-cleared tile, so a -0 input
        // normalizes to +0 before either epilogue sees it; both paths agree
        // bit for bit and NaN still propagates as NaN.
        var bits = fused.ToArray();
        Assert.Equal(System.BitConverter.SingleToInt32Bits(0.0f), System.BitConverter.SingleToInt32Bits(bits[0]));
        Assert.True(float.IsNaN(bits[1]));
    }

    [Fact]
    public void FusedEpilogue_BitMatchesUnfused_Double()
    {
        var x = DenseTensor<double>.OfValues(new double[1, 1, 2, 2] { { { { 1.0, -2.0 }, { 0.5, -0.0 } } } });
        var w = DenseTensor<double>.OfValues(new double[1, 1, 2, 2] { { { { 1.0, 0.0 }, { 0.0, -1.0 } } } });
        var b = DenseTensor<double>.OfValues(new double[] { 0.25 });
        var pads = new int[] { 0, 0, 0, 0 };
        var plain = Tensor<double>.Conv2D(x, w, 1, pads, b, null, new int[] { 1, 1 }, null, TensorExecutionOptions.Auto, false);
        var fused = Tensor<double>.Conv2D(x, w, 1, pads, b, null, new int[] { 1, 1 }, null, TensorExecutionOptions.Auto, true);
        var relu = Tensor<double>.Relu(plain, TensorExecutionOptions.Auto);
        AssertBitsEqualDouble(relu.ToArray(), fused.ToArray());
    }

    [Fact]
    public void UnfusedOverload_MatchesLegacyPath()
    {
        var x = DenseTensor<float>.OfValues(new float[1, 1, 2, 2] { { { { 1f, 2f }, { 3f, 4f } } } });
        var w = DenseTensor<float>.OfValues(new float[1, 1, 1, 1] { { { { 2f } } } });
        var b = DenseTensor<float>.OfValues(new float[] { 1f });
        var pads = new int[] { 0, 0, 0, 0 };
        var legacy = Tensor<float>.Conv2D(x, w, 1, pads, b, null, new int[] { 1, 1 }, null, TensorExecutionOptions.Auto);
        var explicitFalse = Tensor<float>.Conv2D(x, w, 1, pads, b, null, new int[] { 1, 1 }, null, TensorExecutionOptions.Auto, false);
        AssertBitsEqual(legacy.ToArray(), explicitFalse.ToArray());
    }

    [Fact]
    public void ProviderFusedConv_MatchesReluOfPlain()
    {
        var x = DenseTensor<float>.OfValues(new float[1, 1, 2, 2] { { { { 1f, -4f }, { 2f, -0.5f } } } });
        var w = DenseTensor<float>.OfValues(new float[1, 1, 1, 1] { { { { 3f } } } });
        var b = DenseTensor<float>.OfValues(new float[] { -2f });
        var plain = CPUExecutionProvider.Conv(x, w, b, null, null, null, null, null, null, null, false);
        var fused = CPUExecutionProvider.Conv(x, w, b, null, null, null, null, null, null, null, true);
        Assert.Equal(OpStatus.Success, plain.Status);
        Assert.Equal(OpStatus.Success, fused.Status);
        var relu = CPUExecutionProvider.Relu(plain.Outputs[0], null);
        Assert.Equal(OpStatus.Success, relu.Status);
        AssertBitsEqual(
            ((Tensor<float>)relu.Outputs[0]).ToArray(),
            ((Tensor<float>)fused.Outputs[0]).ToArray());
    }
}
