using System.Collections.Generic;
using System.Linq;

namespace Lokad.Onnx.Backend.Tests;

// R1: fusion must preserve operator domains. Every participant in a fusion
// pattern (including Constant sources and the surviving node) has to live in
// the standard domain with a supported schema; a single custom-domain
// substitution must block fusion while leaving unfused execution intact.
public class GraphFusionDomainTests
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

    // Tiny LayerNorm cluster: epsilon comes from a Constant node, the Pow
    // exponent from a scalar initializer. Returns the model plus a map from
    // position label to the node index carrying that role.
    static (OnnxModel Model, Dictionary<string, int> Pos) TinyLayerNormModel()
    {
        var mp = new OnnxModel { Name = "tiny-ln" };
        mp.Opset[""] = 11;
        mp.Opset["custom"] = 1;
        mp.Inputs.Add(NamedIO("x", 2, 4));
        mp.Outputs.Add(NamedIO("z", 2, 4));
        mp.Initializers.Add(FloatInit("gamma", new[] { 4 }, new[] { 1f, 2f, 3f, 4f }));
        mp.Initializers.Add(FloatInit("beta", new[] { 4 }, new[] { 0.5f, -0.5f, 1f, 0f }));
        mp.Initializers.Add(FloatInit("two", new int[0], new[] { 2f }));
        var eps = Model.ToTensor(new OnnxTensor { ElementType = TensorElementType.Float, Dims = new int[0], Data = new[] { 1e-5f } });
        var pos = new Dictionary<string, int>();
        void Add(string label, OnnxNode n) { pos[label] = mp.Nodes.Count; mp.Nodes.Add(n); }
        Add("rm1", Nod("ReduceMean", new[] { "x" }, new[] { "m" }, Attrs(("axes", new long[] { -1 }), ("keepdims", 1L))));
        Add("sub", Nod("Sub", new[] { "x", "m" }, new[] { "d" }, null));
        Add("pow", Nod("Pow", new[] { "d", "two" }, new[] { "s" }, null));
        Add("rm2", Nod("ReduceMean", new[] { "s" }, new[] { "v" }, Attrs(("axes", new long[] { -1 }), ("keepdims", 1L))));
        Add("constEps", Nod("Constant", new string[0], new[] { "e" }, Attrs(("value", eps))));
        Add("addEps", Nod("Add", new[] { "v", "e" }, new[] { "ve" }, null));
        Add("sqrt", Nod("Sqrt", new[] { "ve" }, new[] { "sd" }, null));
        Add("div", Nod("Div", new[] { "d", "sd" }, new[] { "n" }, null));
        Add("mul", Nod("Mul", new[] { "n", "gamma" }, new[] { "g" }, null));
        Add("finalAdd", Nod("Add", new[] { "g", "beta" }, new[] { "z" }, null));
        return (mp, pos);
    }

    static float[] IndependentLayerNorm(float[,] x, float[] gamma, float[] beta, double eps)
    {
        int rows = x.GetLength(0), cols = x.GetLength(1);
        var y = new float[rows * cols];
        for (int i = 0; i < rows; i++)
        {
            double mean = 0;
            for (int j = 0; j < cols; j++) mean += x[i, j];
            mean /= cols;
            double variance = 0;
            for (int j = 0; j < cols; j++) variance += (x[i, j] - mean) * (x[i, j] - mean);
            variance /= cols;
            double denom = System.Math.Sqrt(variance + eps);
            for (int j = 0; j < cols; j++) y[i * cols + j] = (float)((x[i, j] - mean) / denom * gamma[j] + beta[j]);
        }
        return y;
    }

    static ITensor XInput() =>
        DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f } });

    [Fact]
    public void CustomSub_BlocksFusionAndFailsExecution()
    {
        // Exact R1 repro: standard opset 11 plus custom opset 1, Sub in domain custom.
        var (mp, _) = TinyLayerNormModel();
        mp.Nodes.First(n => n.OpType == "Sub").Domain = "custom";
        var graph = Model.Load(mp)!;
        Assert.DoesNotContain(graph.Nodes, n => n.Op == OpType.LayerNormalization);
        Assert.Contains(graph.Nodes, n => n.Op == OpType.Div && (n.Domain ?? "") == "");
        var ok = graph.Execute(new Dictionary<string, ITensor> { { "x", XInput() } }, true);
        Assert.False(ok);
        Assert.Contains("custom", graph.LastErrorMessage ?? "");
    }

    [Theory]
    [InlineData("rm1")]
    [InlineData("sub")]
    [InlineData("pow")]
    [InlineData("rm2")]
    [InlineData("constEps")]
    [InlineData("addEps")]
    [InlineData("sqrt")]
    [InlineData("div")]
    [InlineData("mul")]
    [InlineData("finalAdd")]
    public void CustomDomainAtEachPosition_BlocksFusion(string position)
    {
        var (mp, pos) = TinyLayerNormModel();
        mp.Nodes[pos[position]].Domain = "custom";
        var graph = Model.Load(mp)!;
        Assert.DoesNotContain(graph.Nodes, n => n.Op == OpType.LayerNormalization);
        // The unfused cluster cannot dispatch unknown custom semantics.
        Assert.False(graph.Execute(new Dictionary<string, ITensor> { { "x", XInput() } }, true));
    }

    [Fact]
    public void AllStandard_FusesWithStandardIdentityAndParity()
    {
        var (mp, _) = TinyLayerNormModel();
        var graph = Model.Load(mp)!;
        Assert.Single(graph.Nodes);
        var fused = graph.Nodes[0];
        Assert.Equal(OpType.LayerNormalization, fused.Op);
        Assert.True(fused.IsFused);
        Assert.True(string.IsNullOrEmpty(fused.Domain));
        Assert.Equal(11, fused.OpsetVersion);
        Assert.True(CPUExecutionProvider.SupportsNode(fused));
        Assert.True(graph.Execute(new Dictionary<string, ITensor> { { "x", XInput() } }, true));
        var actual = ((Tensor<float>)graph.Outputs["z"]).ToArray();
        var expected = IndependentLayerNorm(
            new float[,] { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f } },
            new[] { 1f, 2f, 3f, 4f }, new[] { 0.5f, -0.5f, 1f, 0f }, 1e-5);
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], 5);
    }

    [Fact]
    public void FusedMatchesUnfusedBlockedTwin()
    {
        var (mp, _) = TinyLayerNormModel();
        var fused = Model.Load(mp)!;
        Assert.True(fused.Execute(new Dictionary<string, ITensor> { { "x", XInput() } }, true));
        var fusedOut = ((Tensor<float>)fused.Outputs["z"]).ToArray();

        // Exposing the normalized values blocks fusion but keeps every
        // original operator executing on the unfused path.
        var (mp2, _) = TinyLayerNormModel();
        mp2.Outputs.Add(NamedIO("n", 2, 4));
        var unfused = Model.Load(mp2)!;
        Assert.DoesNotContain(unfused.Nodes, n => n.Op == OpType.LayerNormalization);
        Assert.True(unfused.Execute(new Dictionary<string, ITensor> { { "x", XInput() } }, true));
        var unfusedOut = ((Tensor<float>)unfused.Outputs["z"]).ToArray();
        var exposed = ((Tensor<float>)unfused.Outputs["n"]).ToArray();
        Assert.Equal(fusedOut.Length, unfusedOut.Length);
        for (int i = 0; i < fusedOut.Length; i++) Assert.Equal(fusedOut[i], unfusedOut[i], 5);
        Assert.Equal(8, exposed.Length);
    }

    static (OnnxModel Model, int ConstExp) LayerNormConstantExponentModel(string domain)
    {
        var (mp, _) = TinyLayerNormModel();
        mp.Initializers.RemoveAll(t => t.Name == "two");
        var two = Model.ToTensor(new OnnxTensor { ElementType = TensorElementType.Float, Dims = new int[0], Data = new[] { 2f } });
        var c = Nod("Constant", new string[0], new[] { "two" }, Attrs(("value", two)), domain);
        int idx = mp.Nodes.FindIndex(n => n.OpType == "Pow");
        mp.Nodes.Insert(idx, c);
        return (mp, idx);
    }

    [Fact]
    public void ConstantExponent_StandardDomain_Fuses()
    {
        var (mp, _) = LayerNormConstantExponentModel("");
        var graph = Model.Load(mp)!;
        Assert.Contains(graph.Nodes, n => n.Op == OpType.LayerNormalization && n.IsFused);
        Assert.True(graph.Execute(new Dictionary<string, ITensor> { { "x", XInput() } }, true));
        var actual = ((Tensor<float>)graph.Outputs["z"]).ToArray();
        var expected = IndependentLayerNorm(
            new float[,] { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f } },
            new[] { 1f, 2f, 3f, 4f }, new[] { 0.5f, -0.5f, 1f, 0f }, 1e-5);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], 5);
    }

    [Fact]
    public void ConstantExponent_CustomDomain_BlocksFusion()
    {
        var (mp, _) = LayerNormConstantExponentModel("custom");
        var graph = Model.Load(mp)!;
        Assert.DoesNotContain(graph.Nodes, n => n.Op == OpType.LayerNormalization);
        Assert.False(graph.Execute(new Dictionary<string, ITensor> { { "x", XInput() } }, true));
    }
}
