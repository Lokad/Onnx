using System.Collections.Generic;
using System.Linq;

namespace Lokad.Onnx.Backend.Tests;

// R1: the rotary-embedding fusion must preserve operator domains. Every
// participant (Slice/Neg/Concat/Cos/Sin/Mul nodes, bound Constant sources and
// the surviving Add) has to be standard-domain with a supported schema.
public class GraphFusionRopeDomainTests
{
    static OnnxValueInfo NamedIO(string name, int d0, int d1) =>
        new OnnxValueInfo { Name = name, ElementType = TensorElementType.Float, Dims = new[] { d0, d1 } };

    static OnnxTensor FloatInit(string name, int[] dims, float[] values) =>
        new OnnxTensor { Name = name, ElementType = TensorElementType.Float, Dims = dims, Data = values };

    static OnnxTensor IntInit(string name, int[] dims, long[] values) =>
        new OnnxTensor { Name = name, ElementType = TensorElementType.Int64, Dims = dims, Data = values };

    static Dictionary<string, object> Attrs(params (string Name, object Value)[] attrs)
    {
        var d = new Dictionary<string, object>();
        foreach (var a in attrs) d[a.Name] = a.Value;
        return d;
    }

    static OnnxNode Nod(string op, string[] inputs, string[] outputs, Dictionary<string, object>? attrs) =>
        Nod(op, inputs, outputs, attrs, "");

    static OnnxNode Nod(string op, string[] inputs, string[] outputs, Dictionary<string, object>? attrs, string domain) =>
        new OnnxNode { OpType = op, Domain = domain, Inputs = inputs, Outputs = outputs, Attributes = attrs ?? new Dictionary<string, object>() };

    // Compact rotary cluster with half=2 on the last axis: out = x*cos +
    // concat(-x[2:], x[:2])*sin. Slice bounds come from initializers.
    static (OnnxModel Model, Dictionary<string, int> Pos) TinyRopeModel()
    {
        var mp = new OnnxModel { Name = "tiny-rope" };
        mp.Opset[""] = 14;
        mp.Opset["custom"] = 1;
        mp.Inputs.Add(NamedIO("x", 2, 4));
        mp.Outputs.Add(NamedIO("z", 2, 4));
        mp.Initializers.Add(FloatInit("freq", new[] { 2, 4 },
            new[] { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f }));
        mp.Initializers.Add(IntInit("s0", new[] { 1 }, new long[] { 0 }));
        mp.Initializers.Add(IntInit("e0", new[] { 1 }, new long[] { 2 }));
        mp.Initializers.Add(IntInit("ax", new[] { 1 }, new long[] { -1 }));
        mp.Initializers.Add(IntInit("st", new[] { 1 }, new long[] { 1 }));
        mp.Initializers.Add(IntInit("s1", new[] { 1 }, new long[] { 2 }));
        mp.Initializers.Add(IntInit("e1", new[] { 1 }, new long[] { long.MaxValue }));
        var pos = new Dictionary<string, int>();
        void Add(string label, OnnxNode n) { pos[label] = mp.Nodes.Count; mp.Nodes.Add(n); }
        Add("cos", Nod("Cos", new[] { "freq" }, new[] { "co" }, null));
        Add("sin", Nod("Sin", new[] { "freq" }, new[] { "si" }, null));
        Add("sliceFirst", Nod("Slice", new[] { "x", "s0", "e0", "ax", "st" }, new[] { "x0" }, null));
        Add("sliceSecond", Nod("Slice", new[] { "x", "s1", "e1", "ax", "st" }, new[] { "x1" }, null));
        Add("neg", Nod("Neg", new[] { "x1" }, new[] { "nx1" }, null));
        Add("concat", Nod("Concat", new[] { "nx1", "x0" }, new[] { "rot" }, Attrs(("axis", -1L))));
        Add("mx", Nod("Mul", new[] { "x", "co" }, new[] { "m0" }, null));
        Add("mr", Nod("Mul", new[] { "rot", "si" }, new[] { "m1" }, null));
        Add("add", Nod("Add", new[] { "m0", "m1" }, new[] { "z" }, null));
        return (mp, pos);
    }

    static float[] IndependentRope(float[,] x, float[,] freq)
    {
        int rows = x.GetLength(0), cols = x.GetLength(1), half = cols / 2;
        var y = new float[rows * cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
            {
                float co = (float)System.Math.Cos(freq[i, j]);
                float si = (float)System.Math.Sin(freq[i, j]);
                float rot = j < half ? -x[i, j + half] : x[i, j - half];
                y[i * cols + j] = x[i, j] * co + rot * si;
            }
        return y;
    }

    static float[,] XValues() => new float[,] { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f } };
    static float[,] FreqValues() => new float[,] { { 0.1f, 0.2f, 0.3f, 0.4f }, { 0.5f, 0.6f, 0.7f, 0.8f } };

    [Fact]
    public void AllStandard_FusesWithStandardIdentityAndParity()
    {
        var (mp, _) = TinyRopeModel();
        var graph = Model.Load(mp)!;
        Assert.Single(graph.Nodes.Where(n => n.Op == OpType.RotaryEmbedding));
        var fused = graph.Nodes.First(n => n.Op == OpType.RotaryEmbedding);
        Assert.True(fused.IsFused);
        Assert.True(string.IsNullOrEmpty(fused.Domain));
        Assert.Equal(14, fused.OpsetVersion);
        Assert.Equal(2, fused.RequiredInt("half"));
        Assert.True(CPUExecutionProvider.SupportsNode(fused));
        var x = DenseTensor<float>.OfValues(XValues());
        Assert.True(graph.Execute(new Dictionary<string, ITensor> { { "x", x } }, true));
        var actual = ((Tensor<float>)graph.Outputs["z"]).ToArray();
        var expected = IndependentRope(XValues(), FreqValues());
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], 5);
    }

    [Fact]
    public void FusedMatchesUnfusedBlockedTwin()
    {
        var (mp, _) = TinyRopeModel();
        var fused = Model.Load(mp)!;
        Assert.True(fused.Execute(new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(XValues()) } }, true));
        var fusedOut = ((Tensor<float>)fused.Outputs["z"]).ToArray();

        var (mp2, _) = TinyRopeModel();
        mp2.Outputs.Add(NamedIO("m0", 2, 4));
        var unfused = Model.Load(mp2)!;
        Assert.DoesNotContain(unfused.Nodes, n => n.Op == OpType.RotaryEmbedding);
        Assert.True(unfused.Execute(new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(XValues()) } }, true));
        var unfusedOut = ((Tensor<float>)unfused.Outputs["z"]).ToArray();
        Assert.Equal(fusedOut.Length, unfusedOut.Length);
        for (int i = 0; i < fusedOut.Length; i++) Assert.Equal(fusedOut[i], unfusedOut[i], 5);
    }

    [Theory]
    [InlineData("sliceFirst")]
    [InlineData("sliceSecond")]
    [InlineData("neg")]
    [InlineData("concat")]
    [InlineData("cos")]
    [InlineData("sin")]
    [InlineData("mx")]
    [InlineData("mr")]
    [InlineData("add")]
    public void CustomDomainAtEachPosition_BlocksFusion(string position)
    {
        var (mp, pos) = TinyRopeModel();
        mp.Nodes[pos[position]].Domain = "custom";
        var graph = Model.Load(mp)!;
        Assert.DoesNotContain(graph.Nodes, n => n.Op == OpType.RotaryEmbedding);
        Assert.False(graph.Execute(new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(XValues()) } }, true));
        Assert.Contains("custom", graph.LastErrorMessage ?? "");
    }

    static (OnnxModel Model, int ConstBound) RopeConstantBoundModel(string domain)
    {
        var (mp, _) = TinyRopeModel();
        mp.Initializers.RemoveAll(t => t.Name == "s0");
        var s0 = Model.ToTensor(new OnnxTensor { ElementType = TensorElementType.Int64, Dims = new[] { 1 }, Data = new long[] { 0 } });
        var c = Nod("Constant", new string[0], new[] { "s0" }, Attrs(("value", s0)), domain);
        int idx = mp.Nodes.FindIndex(n => n.OpType == "Slice" && n.Outputs[0] == "x0");
        mp.Nodes.Insert(idx, c);
        return (mp, idx);
    }

    [Fact]
    public void ConstantBound_StandardDomain_Fuses()
    {
        var (mp, _) = RopeConstantBoundModel("");
        var graph = Model.Load(mp)!;
        Assert.Single(graph.Nodes.Where(n => n.Op == OpType.RotaryEmbedding));
        Assert.True(graph.Execute(new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(XValues()) } }, true));
        var actual = ((Tensor<float>)graph.Outputs["z"]).ToArray();
        var expected = IndependentRope(XValues(), FreqValues());
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], 5);
    }

    [Fact]
    public void ConstantBound_CustomDomain_BlocksFusion()
    {
        var (mp, _) = RopeConstantBoundModel("custom");
        var graph = Model.Load(mp)!;
        Assert.DoesNotContain(graph.Nodes, n => n.Op == OpType.RotaryEmbedding);
        Assert.False(graph.Execute(new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(XValues()) } }, true));
    }
}
