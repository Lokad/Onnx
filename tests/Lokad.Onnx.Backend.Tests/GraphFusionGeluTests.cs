using System.Collections.Generic;
using System.Linq;

namespace Lokad.Onnx.Backend.Tests;

// C01: the exact-GELU erf chain (Div by sqrt(2), Erf, plus one, times x,
// times one half) must fuse into one native exact Gelu; wrong constants,
// swapped division order, foreign multiplicands, and exposed intermediates
// must block fusion while leaving unfused execution intact.
public class GraphFusionGeluTests
{
    const float Sqrt2 = 1.4142135f;

    static OnnxValueInfo NamedIO(string name, int d0, int d1) =>
        new OnnxValueInfo { Name = name, ElementType = TensorElementType.Float, Dims = new[] { d0, d1 } };

    static OnnxTensor FloatInit(string name, int[] dims, float[] values) =>
        new OnnxTensor { Name = name, ElementType = TensorElementType.Float, Dims = dims, Data = values };

    static OnnxNode Nod(string op, string[] inputs, string[] outputs) =>
        new OnnxNode { OpType = op, Domain = "", Inputs = inputs, Outputs = outputs, Attributes = new Dictionary<string, object>() };

    static OnnxModel TinyGeluModel(float c0, float c1, float c2)
    {
        var mp = new OnnxModel { Name = "tiny-gelu" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(NamedIO("x", 2, 4));
        mp.Outputs.Add(NamedIO("z", 2, 4));
        mp.Initializers.Add(FloatInit("c0", new int[0], new[] { c0 }));
        mp.Initializers.Add(FloatInit("c1", new int[0], new[] { c1 }));
        mp.Initializers.Add(FloatInit("c2", new int[0], new[] { c2 }));
        mp.Nodes.Add(Nod("Div", new[] { "x", "c0" }, new[] { "d" }));
        mp.Nodes.Add(Nod("Erf", new[] { "d" }, new[] { "e" }));
        mp.Nodes.Add(Nod("Add", new[] { "e", "c1" }, new[] { "a" }));
        mp.Nodes.Add(Nod("Mul", new[] { "x", "a" }, new[] { "m" }));
        mp.Nodes.Add(Nod("Mul", new[] { "m", "c2" }, new[] { "z" }));
        return mp;
    }

    static ITensor XInput() =>
        DenseTensor<float>.OfValues(new float[,] { { 0f, 1f, 2f, 3f }, { 4f, 5f, 6f, 7f } });

    [Fact]
    public void ExactChain_FusesToSingleNativeGelu()
    {
        var graph = Model.Load(TinyGeluModel(Sqrt2, 1f, 0.5f))!;
        Assert.Single(graph.Nodes);
        var fused = graph.Nodes[0];
        Assert.Equal(OpType.Gelu, fused.Op);
        Assert.True(fused.IsFused);
        Assert.True(string.IsNullOrEmpty(fused.Domain));
        Assert.Equal(14, fused.OpsetVersion);
        Assert.Equal(new[] { "x" }, fused.Inputs);
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
        var fused = Model.Load(TinyGeluModel(Sqrt2, 1f, 0.5f))!;
        Assert.True(fused.Execute(new Dictionary<string, ITensor> { { "x", XInput() } }, true));
        var fusedOut = ((Tensor<float>)fused.Outputs["z"]).ToArray();

        // Exposing the erf value blocks fusion but keeps every original
        // operator executing on the unfused path.
        var mp2 = TinyGeluModel(Sqrt2, 1f, 0.5f);
        mp2.Outputs.Add(NamedIO("e", 2, 4));
        var unfused = Model.Load(mp2)!;
        Assert.DoesNotContain(unfused.Nodes, n => n.Op == OpType.Gelu);
        Assert.True(unfused.Execute(new Dictionary<string, ITensor> { { "x", XInput() } }, true));
        var unfusedOut = ((Tensor<float>)unfused.Outputs["z"]).ToArray();
        Assert.Equal(fusedOut.Length, unfusedOut.Length);
        for (int i = 0; i < fusedOut.Length; i++) Assert.Equal(fusedOut[i], unfusedOut[i], 5);
    }

    [Fact]
    public void FusedMatchesUnfused_ExceptionalInputs()
    {
        // Same twin construction as BlockedTwin, but the feed carries NaN
        // and infinities: every fused model routes all values through the
        // fused kernel, so one-sided exceptional handling would diverge.
        float[,] vals = new float[,] { { float.NaN, float.PositiveInfinity, float.NegativeInfinity, 1f }, { 2f, 3f, 4f, 5f } };
        var fused = Model.Load(TinyGeluModel(Sqrt2, 1f, 0.5f))!;
        Assert.True(fused.Execute(new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(vals) } }, true), fused.LastErrorMessage);
        var fz = ((Tensor<float>)fused.Outputs["z"]).ToArray();
        var mp2 = TinyGeluModel(Sqrt2, 1f, 0.5f);
        mp2.Outputs.Add(NamedIO("e", 2, 4));
        var unfused = Model.Load(mp2)!;
        Assert.True(unfused.Execute(new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(vals) } }, true), unfused.LastErrorMessage);
        var uz = ((Tensor<float>)unfused.Outputs["z"]).ToArray();
        Assert.Equal(fz.Length, uz.Length);
        foreach (int i in new int[] { 0, 2 }) Assert.True(float.IsNaN(fz[i]) && float.IsNaN(uz[i]), "index " + i);
        Assert.Equal(fz[1], uz[1], 5);
        Assert.True(float.IsPositiveInfinity(fz[1]) && float.IsPositiveInfinity(uz[1]));
        for (int i = 3; i < fz.Length; i++) Assert.Equal(fz[i], uz[i], 5);
    }

    [Fact]
    public void TwoChainedBlocks_BothFuse()
    {
        // dinov2 miniature: two GELU blocks in series sharing one constant
        // set, like encoder layers sharing layer-0 constants.
        var mp = new OnnxModel { Name = "twin-gelu" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(NamedIO("x", 2, 4));
        mp.Outputs.Add(NamedIO("z", 2, 4));
        mp.Initializers.Add(FloatInit("c0", new int[0], new[] { Sqrt2 }));
        mp.Initializers.Add(FloatInit("c1", new int[0], new[] { 1f }));
        mp.Initializers.Add(FloatInit("c2", new int[0], new[] { 0.5f }));
        mp.Nodes.Add(Nod("Div", new[] { "x", "c0" }, new[] { "d0" }));
        mp.Nodes.Add(Nod("Erf", new[] { "d0" }, new[] { "e0" }));
        mp.Nodes.Add(Nod("Add", new[] { "e0", "c1" }, new[] { "a0" }));
        mp.Nodes.Add(Nod("Mul", new[] { "x", "a0" }, new[] { "m0" }));
        mp.Nodes.Add(Nod("Mul", new[] { "m0", "c2" }, new[] { "h" }));
        mp.Nodes.Add(Nod("Div", new[] { "h", "c0" }, new[] { "d1" }));
        mp.Nodes.Add(Nod("Erf", new[] { "d1" }, new[] { "e1" }));
        mp.Nodes.Add(Nod("Add", new[] { "e1", "c1" }, new[] { "a1" }));
        mp.Nodes.Add(Nod("Mul", new[] { "h", "a1" }, new[] { "m1" }));
        mp.Nodes.Add(Nod("Mul", new[] { "m1", "c2" }, new[] { "z" }));
        var graph = Model.Load(mp)!;
        Assert.Equal(2, graph.Nodes.Count(n => n.Op == OpType.Gelu && n.IsFused));
    }

    [Fact]
    public void WrongDivisor_BlocksFusion()
    {
        var graph = Model.Load(TinyGeluModel(1.5f, 1f, 0.5f))!;
        Assert.DoesNotContain(graph.Nodes, n => n.Op == OpType.Gelu);
        Assert.Equal(5, graph.Nodes.Count);
        Assert.True(graph.Execute(new Dictionary<string, ITensor> { { "x", XInput() } }, true));
    }

    [Fact]
    public void SwappedDivisionOrder_BlocksFusion()
    {
        var mp = TinyGeluModel(Sqrt2, 1f, 0.5f);
        mp.Nodes[0] = Nod("Div", new[] { "c0", "x" }, new[] { "d" });
        var graph = Model.Load(mp)!;
        Assert.DoesNotContain(graph.Nodes, n => n.Op == OpType.Gelu);
        Assert.True(graph.Execute(new Dictionary<string, ITensor> { { "x", XInput() } }, true));
    }

    [Fact]
    public void ForeignMultiplicand_BlocksFusion()
    {
        var mp = TinyGeluModel(Sqrt2, 1f, 0.5f);
        mp.Nodes[3] = Nod("Mul", new[] { "a", "a" }, new[] { "m" });
        var graph = Model.Load(mp)!;
        Assert.DoesNotContain(graph.Nodes, n => n.Op == OpType.Gelu);
        Assert.True(graph.Execute(new Dictionary<string, ITensor> { { "x", XInput() } }, true));
    }

    [Fact]
    public void ConstantNodes_AreDroppedWhenDead()
    {
        var mp = TinyGeluModel(Sqrt2, 1f, 0.5f);
        mp.Initializers.RemoveAll(t => t.Name == "c0" || t.Name == "c1" || t.Name == "c2");
        var chain = mp.Nodes.ToList();
        mp.Nodes.Clear();
        void ConstNode(string name, float v)
        {
            var t = Model.ToTensor(new OnnxTensor { ElementType = TensorElementType.Float, Dims = new int[0], Data = new[] { v } });
            mp.Nodes.Add(new OnnxNode { OpType = "Constant", Domain = "", Inputs = new string[0], Outputs = new[] { name }, Attributes = new Dictionary<string, object> { { "value", t } } });
        }
        ConstNode("c0", Sqrt2);
        ConstNode("c1", 1f);
        ConstNode("c2", 0.5f);
        mp.Nodes.AddRange(chain);
        var graph = Model.Load(mp)!;
        Assert.Single(graph.Nodes);
        Assert.Equal(OpType.Gelu, graph.Nodes[0].Op);
        Assert.DoesNotContain(graph.Nodes, n => n.Op == OpType.Constant);
        Assert.True(graph.Execute(new Dictionary<string, ITensor> { { "x", XInput() } }, true));
    }
}
