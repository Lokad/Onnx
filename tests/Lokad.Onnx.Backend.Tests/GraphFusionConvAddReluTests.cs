using System.Collections.Generic;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Covers the Conv+Add+Relu residual fusion: triples fuse with the skip tensor as a
/// fourth Conv input and a bit-identical in-place epilogue, while multi-consumer,
/// graph-output, and rank-3 cases keep the unfused nodes.
/// </summary>
public class GraphFusionConvAddReluTests
{
    static OnnxValueInfo NamedIO(string name, int[] dims) =>
        new OnnxValueInfo { Name = name, ElementType = TensorElementType.Float, Dims = dims };

    static OnnxTensor FloatInit(string name, int[] dims, float[] values) =>
        new OnnxTensor { Name = name, ElementType = TensorElementType.Float, Dims = dims, Data = values };

    static OnnxNode Nod(string op, string[] inputs, string[] outputs) =>
        new OnnxNode { OpType = op, Inputs = inputs, Outputs = outputs, Attributes = new Dictionary<string, object>() };

    static float[] Range(float start, float step, int n)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = start + step * i;
        return v;
    }

    static OnnxModel TripleModel(int[] xDims, int[] wDims, float[] wValues, float[] bValues, bool withBias, bool swapAdd, string outputName, int[] pads)
    {
        var mp = new OnnxModel { Name = "conv-add-relu" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(NamedIO("x", xDims));
        mp.Inputs.Add(NamedIO("s", xDims));
        mp.Outputs.Add(NamedIO(outputName, xDims));
        mp.Initializers.Add(FloatInit("W", wDims, wValues));
        var convNode = withBias ? Nod("Conv", new[] { "x", "W", "B" }, new[] { "c" }) : Nod("Conv", new[] { "x", "W" }, new[] { "c" });
        convNode.Attributes["pads"] = pads;
        if (withBias)
        {
            mp.Initializers.Add(FloatInit("B", new[] { wDims[0] }, bValues));
        }
        mp.Nodes.Add(convNode);
        mp.Nodes.Add(swapAdd ? Nod("Add", new[] { "s", "c" }, new[] { "a" }) : Nod("Add", new[] { "c", "s" }, new[] { "a" }));
        mp.Nodes.Add(Nod("Relu", new[] { "a" }, new[] { "r" }));
        mp.Nodes.Add(Nod("Identity", new[] { "r" }, new[] { outputName }));
        return mp;
    }

    static float[] ExecuteZ(OnnxModel mp, float[] xValues, int[] xDims, float[] sValues, string outputName)
    {
        var graph = Model.Load(mp)!;
        var x = new DenseTensor<float>(xValues, xDims);
        var s = new DenseTensor<float>(sValues, xDims);
        var inputs = new Dictionary<string, ITensor> { { "x", x }, { "s", s } };
        Assert.True(graph.Execute(inputs, true));
        return ((Tensor<float>)graph.Outputs[outputName]).ToArray();
    }

    [Fact]
    public void TripleWithBias_FusesWithResidualInput()
    {
        var mp = TripleModel(new[] { 1, 1, 2, 2 }, new[] { 1, 1, 1, 1 }, new[] { 2f }, new[] { -3f }, true, false, "z", new[] { 0, 0, 0, 0 });
        var graph = Model.Load(mp)!;
        Assert.Equal(2, graph.Nodes.Count);
        var conv = graph.Nodes[0];
        Assert.Equal(OpType.Conv, conv.Op);
        Assert.True(conv.IsFused);
        Assert.Equal(4, conv.Inputs.Length);
        Assert.Equal("s", conv.Inputs[3]);
        Assert.Equal(OpType.Identity, graph.Nodes[1].Op);
        var actual = ExecuteZ(mp, new[] { 1f, -1f, 0.5f, 3f }, new[] { 1, 1, 2, 2 }, new[] { 0.5f, 1f, -2f, 4f }, "z");
        Assert.Equal(new float[] { 0f, 0f, 0f, 7f }, actual);
    }

    [Fact]
    public void TripleWithoutBias_FusesWithEmptyBiasSlot()
    {
        var mp = TripleModel(new[] { 1, 1, 2, 2 }, new[] { 1, 1, 1, 1 }, new[] { 2f }, new float[0], false, false, "z", new[] { 0, 0, 0, 0 });
        var graph = Model.Load(mp)!;
        Assert.Equal(2, graph.Nodes.Count);
        var conv = graph.Nodes[0];
        Assert.Equal(4, conv.Inputs.Length);
        Assert.Equal("s", conv.Inputs[3]);
        var actual = ExecuteZ(mp, new[] { 1f, -1f, 0.5f, 3f }, new[] { 1, 1, 2, 2 }, new[] { 0.5f, 1f, -2f, 4f }, "z");
        Assert.Equal(new float[] { 2.5f, 0f, 0f, 10f }, actual);
    }

    [Fact]
    public void SwappedAddOrder_FusesAndAgrees()
    {
        var mp = TripleModel(new[] { 1, 1, 2, 2 }, new[] { 1, 1, 1, 1 }, new[] { 2f }, new[] { -3f }, true, true, "z", new[] { 0, 0, 0, 0 });
        var graph = Model.Load(mp)!;
        Assert.Equal(2, graph.Nodes.Count);
        Assert.Equal(4, graph.Nodes[0].Inputs.Length);
        var actual = ExecuteZ(mp, new[] { 1f, -1f, 0.5f, 3f }, new[] { 1, 1, 2, 2 }, new[] { 0.5f, 1f, -2f, 4f }, "z");
        Assert.Equal(new float[] { 0f, 0f, 0f, 7f }, actual);
    }

    [Fact]
    public void MultiConsumerAdd_SkipsFusion()
    {
        var mp = new OnnxModel { Name = "conv-add-multi" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(NamedIO("x", new[] { 1, 1, 2, 2 }));
        mp.Inputs.Add(NamedIO("s", new[] { 1, 1, 2, 2 }));
        mp.Outputs.Add(NamedIO("z", new[] { 1, 1, 2, 2 }));
        mp.Outputs.Add(NamedIO("z2", new[] { 1, 1, 2, 2 }));
        mp.Initializers.Add(FloatInit("W", new[] { 1, 1, 1, 1 }, new[] { 1f }));
        mp.Nodes.Add(Nod("Conv", new[] { "x", "W" }, new[] { "c" }));
        mp.Nodes.Add(Nod("Add", new[] { "c", "s" }, new[] { "a" }));
        mp.Nodes.Add(Nod("Relu", new[] { "a" }, new[] { "r" }));
        mp.Nodes.Add(Nod("Identity", new[] { "r" }, new[] { "z" }));
        mp.Nodes.Add(Nod("Identity", new[] { "a" }, new[] { "z2" }));
        var graph = Model.Load(mp)!;
        Assert.Equal(5, graph.Nodes.Count);
        Assert.Equal(2, graph.Nodes[0].Inputs.Length);
    }

    [Fact]
    public void ReluOutputAsGraphOutput_SkipsFusion()
    {
        var mp = new OnnxModel { Name = "conv-add-relu-out" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(NamedIO("x", new[] { 1, 1, 2, 2 }));
        mp.Inputs.Add(NamedIO("s", new[] { 1, 1, 2, 2 }));
        mp.Outputs.Add(NamedIO("r", new[] { 1, 1, 2, 2 }));
        mp.Initializers.Add(FloatInit("W", new[] { 1, 1, 1, 1 }, new[] { 1f }));
        mp.Nodes.Add(Nod("Conv", new[] { "x", "W" }, new[] { "c" }));
        mp.Nodes.Add(Nod("Add", new[] { "c", "s" }, new[] { "a" }));
        mp.Nodes.Add(Nod("Relu", new[] { "a" }, new[] { "r" }));
        var graph = Model.Load(mp)!;
        Assert.Equal(3, graph.Nodes.Count);
        Assert.Equal(2, graph.Nodes[0].Inputs.Length);
    }

    [Fact]
    public void ConvOutputAsGraphOutput_SkipsFusion()
    {
        var mp = new OnnxModel { Name = "conv-out-kept" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(NamedIO("x", new[] { 1, 1, 2, 2 }));
        mp.Inputs.Add(NamedIO("s", new[] { 1, 1, 2, 2 }));
        mp.Outputs.Add(NamedIO("c", new[] { 1, 1, 2, 2 }));
        mp.Outputs.Add(NamedIO("z", new[] { 1, 1, 2, 2 }));
        mp.Initializers.Add(FloatInit("W", new[] { 1, 1, 1, 1 }, new[] { 1f }));
        mp.Nodes.Add(Nod("Conv", new[] { "x", "W" }, new[] { "c" }));
        mp.Nodes.Add(Nod("Add", new[] { "c", "s" }, new[] { "a" }));
        mp.Nodes.Add(Nod("Relu", new[] { "a" }, new[] { "r" }));
        mp.Nodes.Add(Nod("Identity", new[] { "r" }, new[] { "z" }));
        var graph = Model.Load(mp)!;
        Assert.Equal(4, graph.Nodes.Count);
        Assert.Equal(2, graph.Nodes[0].Inputs.Length);
    }

    [Fact]
    public void ChainedTriples_BothFuseWithRemappedSkip()
    {
        var mp = new OnnxModel { Name = "conv-chain" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(NamedIO("x", new[] { 1, 1, 1, 2 }));
        mp.Inputs.Add(NamedIO("s", new[] { 1, 1, 1, 2 }));
        mp.Outputs.Add(NamedIO("z", new[] { 1, 1, 1, 2 }));
        mp.Initializers.Add(FloatInit("W1", new[] { 1, 1, 1, 1 }, new[] { 1f }));
        mp.Initializers.Add(FloatInit("W2", new[] { 1, 1, 1, 1 }, new[] { 1f }));
        mp.Nodes.Add(Nod("Conv", new[] { "x", "W1" }, new[] { "c1" }));
        mp.Nodes.Add(Nod("Add", new[] { "c1", "s" }, new[] { "a1" }));
        mp.Nodes.Add(Nod("Relu", new[] { "a1" }, new[] { "r1" }));
        mp.Nodes.Add(Nod("Conv", new[] { "r1", "W2" }, new[] { "c2" }));
        mp.Nodes.Add(Nod("Add", new[] { "c2", "r1" }, new[] { "a2" }));
        mp.Nodes.Add(Nod("Relu", new[] { "a2" }, new[] { "r2" }));
        mp.Nodes.Add(Nod("Identity", new[] { "r2" }, new[] { "z" }));
        var graph = Model.Load(mp)!;
        Assert.Equal(3, graph.Nodes.Count);
        Assert.Equal(new[] { "x", "W1", "", "s" }, graph.Nodes[0].Inputs);
        Assert.Equal(new[] { "c1", "W2", "", "c1" }, graph.Nodes[1].Inputs);
        var x = new DenseTensor<float>(new[] { 1f, -2f }, new[] { 1, 1, 1, 2 });
        var s = new DenseTensor<float>(new[] { 3f, -1f }, new[] { 1, 1, 1, 2 });
        var inputs = new Dictionary<string, ITensor> { { "x", x }, { "s", s } };
        Assert.True(graph.Execute(inputs, true));
        Assert.Equal(new float[] { 8f, 0f }, ((Tensor<float>)graph.Outputs["z"]).ToArray());
    }

    [Fact]
    public void FusedMatchesUnfused_Bitwise()
    {
        int[] dims = new[] { 1, 2, 4, 4 };
        var mp = TripleModel(dims, new[] { 2, 2, 3, 3 }, Range(-1f, 0.25f, 36), new[] { 0.5f, -0.5f }, true, false, "z", new[] { 1, 1, 1, 1 });
        var veto = TripleModel(dims, new[] { 2, 2, 3, 3 }, Range(-1f, 0.25f, 36), new[] { 0.5f, -0.5f }, true, false, "z", new[] { 1, 1, 1, 1 });
        veto.Outputs.Add(NamedIO("c", dims));
        var fused = Model.Load(mp)!;
        Assert.Equal(2, fused.Nodes.Count);
        var unfused = Model.Load(veto)!;
        Assert.Equal(4, unfused.Nodes.Count);
        var xv = Range(-2f, 0.125f, 32);
        var sv = Range(1f, -0.0625f, 32);
        Assert.Equal(ExecuteZ(veto, xv, dims, sv, "z"), ExecuteZ(mp, xv, dims, sv, "z"));
    }

    [Fact]
    public void TiledShape_FusedMatchesUnfused()
    {
        int[] dims = new[] { 1, 32, 32, 32 };
        var wv = Range(-1f, 0.001f, 32 * 32 * 9);
        var mp = TripleModel(dims, new[] { 32, 32, 3, 3 }, wv, Range(0.25f, -0.01f, 32), true, false, "z", new[] { 1, 1, 1, 1 });
        var veto = TripleModel(dims, new[] { 32, 32, 3, 3 }, wv, Range(0.25f, -0.01f, 32), true, false, "z", new[] { 1, 1, 1, 1 });
        veto.Outputs.Add(NamedIO("c", dims));
        var fused = Model.Load(mp)!;
        Assert.Equal(2, fused.Nodes.Count);
        var unfused = Model.Load(veto)!;
        Assert.Equal(4, unfused.Nodes.Count);
        var xv = Range(-2f, 0.0005f, 32 * 32 * 32);
        var sv = Range(1f, -0.00025f, 32 * 32 * 32);
        var expected = ExecuteZ(veto, xv, dims, sv, "z");
        var actual = ExecuteZ(mp, xv, dims, sv, "z");
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++) Assert.True(Math.Abs(expected[i] - actual[i]) < 1e-4f, "drift at " + i);
    }

    [Fact]
    public void EmbeddingModel_FusesAllSixteenResidualTriples()
    {
        var graph = ModelFixture.LoadRequiredModel("PyannoteEmbedding", "models", "speaker-diarization-community-1", "onnx", "embedding", "embedding_encoder.onnx");
        int fusedConv = 0;
        int add = 0;
        foreach (var n in graph.Nodes)
        {
            if (n.Op == OpType.Conv && n.Inputs.Length == 4)
            {
                fusedConv++;
                Assert.False(string.IsNullOrEmpty(n.Inputs[3]), "Fused residual input must name the skip tensor.");
            }
            if (n.Op == OpType.Add) add++;
        }
        Assert.Equal(16, fusedConv);
        Assert.Equal(0, add);
    }

    [Fact]
    public void Rank3Weights_SkipFusion()
    {
        var mp = new OnnxModel { Name = "conv1d-add-relu" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(NamedIO("x", new[] { 1, 1, 4 }));
        mp.Inputs.Add(NamedIO("s", new[] { 1, 1, 4 }));
        mp.Outputs.Add(NamedIO("z", new[] { 1, 1, 4 }));
        mp.Initializers.Add(FloatInit("W", new[] { 1, 1, 3 }, new[] { 1f, 0f, -1f }));
        mp.Nodes.Add(Nod("Conv", new[] { "x", "W" }, new[] { "c" }));
        mp.Nodes.Add(Nod("Add", new[] { "c", "s" }, new[] { "a" }));
        mp.Nodes.Add(Nod("Relu", new[] { "a" }, new[] { "r" }));
        mp.Nodes.Add(Nod("Identity", new[] { "r" }, new[] { "z" }));
        var graph = Model.Load(mp)!;
        Assert.Equal(4, graph.Nodes.Count);
        Assert.Equal(2, graph.Nodes[0].Inputs.Length);
    }
}

