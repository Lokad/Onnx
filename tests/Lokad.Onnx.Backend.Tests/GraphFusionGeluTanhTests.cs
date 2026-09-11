using System.Collections.Generic;

namespace Lokad.Onnx.Backend.Tests;

// The tanh-approximate GELU chain (x times one half; cube; times 0.044715;
// plus x; times sqrt(2/pi); Tanh; plus one; times) must fuse into one native
// Gelu carrying approximate=tanh; wrong constants and exposed intermediates
// must block fusion while leaving unfused execution intact.
public class GraphFusionGeluTanhTests
{
    static OnnxValueInfo NamedIO(string name, int d0, int d1) =>
        new OnnxValueInfo { Name = name, ElementType = TensorElementType.Float, Dims = new[] { d0, d1 } };

    static OnnxTensor FloatInit(string name, int[] dims, float[] values) =>
        new OnnxTensor { Name = name, ElementType = TensorElementType.Float, Dims = dims, Data = values };

    static OnnxNode Nod(string op, string[] inputs, string[] outputs) =>
        new OnnxNode { OpType = op, Domain = "", Inputs = inputs, Outputs = outputs, Attributes = new Dictionary<string, object>() };

    static OnnxModel TinyGeluTanhModel(float half, float three, float c, float s, float one)
    {
        var mp = new OnnxModel { Name = "tiny-gelu-tanh" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(NamedIO("x", 2, 4));
        mp.Outputs.Add(NamedIO("z", 2, 4));
        mp.Initializers.Add(FloatInit("kh", new int[0], new[] { half }));
        mp.Initializers.Add(FloatInit("ke", new int[0], new[] { three }));
        mp.Initializers.Add(FloatInit("kq", new int[0], new[] { c }));
        mp.Initializers.Add(FloatInit("ks", new int[0], new[] { s }));
        mp.Initializers.Add(FloatInit("ko", new int[0], new[] { one }));
        mp.Nodes.Add(Nod("Mul", new[] { "x", "kh" }, new[] { "t1" }));
        mp.Nodes.Add(Nod("Pow", new[] { "x", "ke" }, new[] { "t2" }));
        mp.Nodes.Add(Nod("Mul", new[] { "t2", "kq" }, new[] { "t3" }));
        mp.Nodes.Add(Nod("Add", new[] { "x", "t3" }, new[] { "t4" }));
        mp.Nodes.Add(Nod("Mul", new[] { "t4", "ks" }, new[] { "t5" }));
        mp.Nodes.Add(Nod("Tanh", new[] { "t5" }, new[] { "t6" }));
        mp.Nodes.Add(Nod("Add", new[] { "t6", "ko" }, new[] { "t7" }));
        mp.Nodes.Add(Nod("Mul", new[] { "t1", "t7" }, new[] { "z" }));
        return mp;
    }

    static ITensor XInput() =>
        DenseTensor<float>.OfValues(new float[,] { { 0f, 1f, 2f, 3f }, { 4f, 5f, 6f, 7f } });

    [Fact]
    public void TanhChain_FusesToSingleNativeGelu()
    {
        var graph = Model.Load(TinyGeluTanhModel(0.5f, 3f, 0.044715f, 0.7978846f, 1f))!;
        Assert.Single(graph.Nodes);
        var fused = graph.Nodes[0];
        Assert.Equal(OpType.Gelu, fused.Op);
        Assert.True(fused.IsFused);
        Assert.Equal(new[] { "x" }, fused.Inputs);
        Assert.True(fused.Attributes is not null && fused.Attributes.TryGetValue("approximate", out var av) && (string)av == "tanh");
        Assert.True(CPUExecutionProvider.SupportsNode(fused));
        Assert.True(graph.Execute(new Dictionary<string, ITensor> { { "x", XInput() } }, true));
        var actual = ((Tensor<float>)graph.Outputs["z"]).ToArray();
        Assert.Equal(8, actual.Length);
        Assert.Equal(0f, actual[0]);
        for (int i = 1; i < actual.Length; i++) Assert.True(actual[i] > 0f);
    }

    [Fact]
    public void FusedMatchesUnfusedBlockedTwin()
    {
        var fused = Model.Load(TinyGeluTanhModel(0.5f, 3f, 0.044715f, 0.7978846f, 1f))!;
        Assert.True(fused.Execute(new Dictionary<string, ITensor> { { "x", XInput() } }, true));
        var fusedOut = ((Tensor<float>)fused.Outputs["z"]).ToArray();
        var mp2 = TinyGeluTanhModel(0.5f, 3f, 0.044715f, 0.7978846f, 1f);
        mp2.Outputs.Add(NamedIO("t6", 2, 4));
        var unfused = Model.Load(mp2)!;
        Assert.DoesNotContain(unfused.Nodes, n => n.Op == OpType.Gelu);
        Assert.True(unfused.Execute(new Dictionary<string, ITensor> { { "x", XInput() } }, true));
        var unfusedOut = ((Tensor<float>)unfused.Outputs["z"]).ToArray();
        Assert.Equal(fusedOut.Length, unfusedOut.Length);
        for (int i = 0; i < fusedOut.Length; i++) Assert.Equal(fusedOut[i], unfusedOut[i], 5);
    }

    [Fact]
    public void WrongConstant_BlocksFusion()
    {
        var graph = Model.Load(TinyGeluTanhModel(0.5f, 3f, 0.044716f, 0.7978846f, 1f))!;
        Assert.DoesNotContain(graph.Nodes, n => n.Op == OpType.Gelu);
        Assert.Equal(8, graph.Nodes.Count);
        Assert.True(graph.Execute(new Dictionary<string, ITensor> { { "x", XInput() } }, true));
    }

    [Fact]
    public void SwappedPowOrder_BlocksFusion()
    {
        var mp = TinyGeluTanhModel(0.5f, 3f, 0.044715f, 0.7978846f, 1f);
        mp.Nodes[1] = new OnnxNode { OpType = "Pow", Domain = "", Inputs = new[] { "ke", "x" }, Outputs = new[] { "t2" }, Attributes = new Dictionary<string, object>() };
        var graph = Model.Load(mp)!;
        Assert.DoesNotContain(graph.Nodes, n => n.Op == OpType.Gelu);
        Assert.True(graph.Execute(new Dictionary<string, ITensor> { { "x", XInput() } }, true));
    }
    [SkippableFact]
    public void Gpt2_FusesTwelveTanhGelu()
    {
        var graph = ModelFixture.LoadRequiredModel("GPT-2", "models", "gpt2-onnx", "onnx", "model.onnx");
        int fused = 0;
        foreach (var node in graph.Nodes)
            if (node.Op == OpType.Gelu && node.IsFused && node.Attributes is not null && node.Attributes.TryGetValue("approximate", out var av) && (string)av == "tanh") fused++;
        Assert.Equal(12, fused);
    }
}
