using System.Collections.Generic;
using System.Linq;
using Lokad.Onnx.Optimization;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// G02-M3: self-shape Reshape chains (Shape of the Reshape own data input through
/// constant-indexed Gathers and Unsqueezes into a Concat) lower to shared
/// zero-copy constants, while non-self shapes, allowzero shapes, misaligned
/// indices and initializer literals keep their chains. Rewired graphs execute
/// bit-identically across alternating input lengths in one process.
/// </summary>
public class ReshapeZeroCopyTests
{
    static OnnxNode ConstVec1(string name, long value)
    {
        var t = Model.ToTensor(new OnnxTensor { ElementType = TensorElementType.Int64, Dims = new[] { 1 }, Data = new long[] { value } });
        return new OnnxNode { OpType = "Constant", Inputs = new string[0], Outputs = new[] { name }, Attributes = new Dictionary<string, object> { ["value"] = t } };
    }

    static OnnxNode Const64(string name, long value)
    {
        var t = Model.ToTensor(new OnnxTensor { ElementType = TensorElementType.Int64, Dims = new int[0], Data = new long[] { value } });
        return new OnnxNode { OpType = "Constant", Inputs = new string[0], Outputs = new[] { name }, Attributes = new Dictionary<string, object> { ["value"] = t } };
    }

    static void AddSelfChain(OnnxModel mp, string tag, string data, string shapeOut, bool swapIdx, bool initLit, string litInitName)
    {
        // Chain tensors carry a tag suffix: tensor names are SSA, so sharing them
        // across chains would corrupt the producer map (last writer wins).
        string s = "s" + tag, g0 = "g" + tag + "0", g1 = "g" + tag + "1";
        string i0 = "i" + tag + "0", i1 = "i" + tag + "1";
        string u0 = "u" + tag + "0", u1 = "u" + tag + "1";
        string l0 = "l" + tag + "0", l1 = "l" + tag + "1";
        Dictionary<string, object> NoAttrs() => new Dictionary<string, object>();
        mp.Nodes.Add(new OnnxNode { OpType = "Shape", Inputs = new[] { data }, Outputs = new[] { s }, Attributes = NoAttrs() });
        if (swapIdx)
        {
            mp.Nodes.Add(Const64(i0, 1));
            mp.Nodes.Add(Const64(i1, 0));
        }
        else
        {
            mp.Nodes.Add(Const64(i0, 0));
            mp.Nodes.Add(Const64(i1, 1));
        }
        mp.Nodes.Add(new OnnxNode { OpType = "Gather", Inputs = new[] { s, i0 }, Outputs = new[] { g0 }, Attributes = new Dictionary<string, object> { ["axis"] = 0L } });
        mp.Nodes.Add(new OnnxNode { OpType = "Gather", Inputs = new[] { s, i1 }, Outputs = new[] { g1 }, Attributes = new Dictionary<string, object> { ["axis"] = 0L } });
        mp.Nodes.Add(new OnnxNode { OpType = "Unsqueeze", Inputs = new[] { g0 }, Outputs = new[] { u0 }, Attributes = new Dictionary<string, object> { ["axes"] = new long[] { 0 } } });
        mp.Nodes.Add(new OnnxNode { OpType = "Unsqueeze", Inputs = new[] { g1 }, Outputs = new[] { u1 }, Attributes = new Dictionary<string, object> { ["axes"] = new long[] { 0 } } });
        var concatInputs = new List<string> { u0, u1 };
        if (initLit)
        {
            mp.Nodes.Add(ConstVec1(l0, 2));
            concatInputs.Add(l0);
            concatInputs.Add(litInitName);
        }
        else
        {
            mp.Nodes.Add(ConstVec1(l0, 2));
            mp.Nodes.Add(ConstVec1(l1, 2));
            concatInputs.Add(l0);
            concatInputs.Add(l1);
        }
        mp.Nodes.Add(new OnnxNode { OpType = "Concat", Inputs = concatInputs.ToArray(), Outputs = new[] { shapeOut }, Attributes = new Dictionary<string, object> { ["axis"] = 0L } });
    }

    static OnnxModel ZeroCopyModel()
    {
        var mp = new OnnxModel { Name = "tiny-zero-copy" };
        mp.Opset[""] = 11;
        mp.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { -1, -1, 4 } });
        mp.Inputs.Add(new OnnxValueInfo { Name = "z", ElementType = TensorElementType.Float, Dims = new[] { -1, -1, 4 } });
        foreach (var y in new[] { "y1", "y2", "yd", "ya", "yv", "ye" })
            mp.Outputs.Add(new OnnxValueInfo { Name = y, ElementType = TensorElementType.Float, Dims = new[] { -1, -1, -1, -1 } });
        mp.Initializers.Add(new OnnxTensor { Name = "litE", ElementType = TensorElementType.Int64, Dims = new[] { 1 }, Data = new long[] { 2 } });
        Dictionary<string, object> NoAttrs() => new Dictionary<string, object>();
        // Two independent self-shape chains with the same vector: they must share
        // one new Constant and both scaffolds must be swept.
        AddSelfChain(mp, "A", "x", "shpA", false, false, "litE");
        mp.Nodes.Add(new OnnxNode { OpType = "Reshape", Inputs = new[] { "x", "shpA" }, Outputs = new[] { "y1" }, Attributes = NoAttrs() });
        AddSelfChain(mp, "B", "x", "shpB", false, false, "litE");
        mp.Nodes.Add(new OnnxNode { OpType = "Reshape", Inputs = new[] { "x", "shpB" }, Outputs = new[] { "y2" }, Attributes = NoAttrs() });
        // Non-self shape (dims of x applied to z): must keep its chain.
        AddSelfChain(mp, "D", "x", "shpD", false, false, "litE");
        mp.Nodes.Add(new OnnxNode { OpType = "Reshape", Inputs = new[] { "z", "shpD" }, Outputs = new[] { "yd" }, Attributes = NoAttrs() });
        // allowzero reshape: 0 would be a literal zero, must keep its chain.
        AddSelfChain(mp, "Z", "x", "shpZ", false, false, "litE");
        mp.Nodes.Add(new OnnxNode { OpType = "Reshape", Inputs = new[] { "x", "shpZ" }, Outputs = new[] { "ya" }, Attributes = new Dictionary<string, object> { ["allowzero"] = 1L } });
        // Swapped indices ([S,B] instead of [B,S]): valid but misaligned, stays.
        AddSelfChain(mp, "S", "x", "shpS", true, false, "litE");
        mp.Nodes.Add(new OnnxNode { OpType = "Reshape", Inputs = new[] { "x", "shpS" }, Outputs = new[] { "yv" }, Attributes = NoAttrs() });
        // Literal from an initializer (overridable per run): must stay.
        AddSelfChain(mp, "E", "x", "shpE", false, true, "litE");
        mp.Nodes.Add(new OnnxNode { OpType = "Reshape", Inputs = new[] { "x", "shpE" }, Outputs = new[] { "ye" }, Attributes = NoAttrs() });
        return mp;
    }

    static Node ReshapeOf(ComputationalGraph graph, string output)
    {
        return graph.Nodes.First(n => n.Op == OpType.Reshape && n.Outputs.Length == 1 && n.Outputs[0] == output);
    }

    static long[] ConstantValues(ComputationalGraph graph, string name)
    {
        var node = graph.Nodes.First(n => n.Op == OpType.Constant && n.Outputs.Length == 1 && n.Outputs[0] == name);
        var t = (ITensor)node.Attributes!["value"];
        Assert.Equal(TensorElementType.Int64, t.ElementType);
        var values = new long[t.Dims.Length == 0 ? 1 : t.Dims[0]];
        for (int i = 0; i < values.Length; i++) values[i] = System.Convert.ToInt64(t.GetValue(i));
        return values;
    }

    [Fact]
    public void SelfChainsShareOneZeroConstantDecoysKeepTheirs()
    {
        var graph = Model.Load(ZeroCopyModel(), runOptimizer: false)!;
        var report = GraphOptimizer.Run(graph);
        var change = Assert.Single(report.Where(c => c.Pass == "reshape-zerocopy"));
        Assert.Contains("rewrote 2 reshape shape inputs", change.Notes[0]);
        string shared = ReshapeOf(graph, "y1").Inputs[1];
        Assert.Equal(shared, ReshapeOf(graph, "y2").Inputs[1]);
        Assert.Equal(new long[] { 0, 0, 2, 2 }, ConstantValues(graph, shared));
        // The two lowered scaffolds are gone; the four decoy chains stay whole.
        Assert.Equal(4, graph.Nodes.Count(n => n.Op == OpType.Shape));
        Assert.Equal(8, graph.Nodes.Count(n => n.Op == OpType.Gather));
        Assert.Equal(8, graph.Nodes.Count(n => n.Op == OpType.Unsqueeze));
        Assert.Equal(4, graph.Nodes.Count(n => n.Op == OpType.Concat));
        Assert.DoesNotContain(graph.Nodes, n => n.Outputs.Length == 1 && (n.Outputs[0] == "shpA" || n.Outputs[0] == "shpB"));
        Assert.Equal("shpD", ReshapeOf(graph, "yd").Inputs[1]);
        Assert.Equal("shpZ", ReshapeOf(graph, "ya").Inputs[1]);
        Assert.Equal("shpS", ReshapeOf(graph, "yv").Inputs[1]);
        Assert.Equal("shpE", ReshapeOf(graph, "ye").Inputs[1]);
    }

    static DenseTensor<float> SeqTensor(int b, int s, float start)
    {
        var data = new float[b, s, 4];
        float v = start;
        for (int i = 0; i < b; i++) for (int j = 0; j < s; j++) for (int k = 0; k < 4; k++) data[i, j, k] = v++;
        return DenseTensor<float>.OfValues(data);
    }

    static Dictionary<string, float[]> RunAll(ComputationalGraph graph, DenseTensor<float> x, DenseTensor<float> z)
    {
        var feed = new Dictionary<string, ITensor> { { "x", x }, { "z", z } };
        Assert.True(graph.Execute(feed, true), graph.LastErrorMessage);
        var out_ = new Dictionary<string, float[]>();
        foreach (var y in new[] { "y1", "y2", "yd", "ya", "yv", "ye" })
            out_[y] = ((Tensor<float>)graph.Outputs[y]).ToArray();
        return out_;
    }

    static void CheckDims(ComputationalGraph graph, int b, int s)
    {
        Assert.Equal(new[] { b, s, 2, 2 }, graph.Outputs["y1"].Dims);
        Assert.Equal(new[] { b, s, 2, 2 }, graph.Outputs["y2"].Dims);
        Assert.Equal(new[] { b, s, 2, 2 }, graph.Outputs["yd"].Dims);
        Assert.Equal(new[] { b, s, 2, 2 }, graph.Outputs["ya"].Dims);
        Assert.Equal(new[] { s, b, 2, 2 }, graph.Outputs["yv"].Dims);
        Assert.Equal(new[] { b, s, 2, 2 }, graph.Outputs["ye"].Dims);
    }

    [Fact]
    public void RewiredMatchesUnfusedTwinBitwiseAcrossLengths()
    {
        var fused = Model.Load(ZeroCopyModel())!;
        var plain = Model.Load(ZeroCopyModel(), runOptimizer: false)!;
        // x and z carry different batch/sequence extents at equal volume, so the
        // non-self decoy proves its dims still come from x, not from its data z.
        var a = RunAll(fused, SeqTensor(2, 3, 1f), SeqTensor(1, 6, 100f));
        var b = RunAll(plain, SeqTensor(2, 3, 1f), SeqTensor(1, 6, 100f));
        foreach (var y in a.Keys) Assert.Equal(b[y], a[y]);
        CheckDims(fused, 2, 3);
        CheckDims(plain, 2, 3);
        // Alternating lengths in the same processes: nothing was frozen.
        var c = RunAll(fused, SeqTensor(6, 1, 7f), SeqTensor(2, 3, 200f));
        var d = RunAll(plain, SeqTensor(6, 1, 7f), SeqTensor(2, 3, 200f));
        foreach (var y in c.Keys) Assert.Equal(d[y], c[y]);
        CheckDims(fused, 6, 1);
        CheckDims(plain, 6, 1);
    }

    [Fact]
    public void DisabledKeepsChains()
    {
        var graph = Model.Load(ZeroCopyModel(), runOptimizer: false)!;
        var report = GraphOptimizer.Run(graph, new[] { "reshape-zerocopy" });
        Assert.DoesNotContain(report, c => c.Pass == "reshape-zerocopy");
        Assert.Equal(6, graph.Nodes.Count(n => n.Op == OpType.Shape));
        Assert.Equal("shpA", ReshapeOf(graph, "y1").Inputs[1]);
    }

    [Fact]
    public void SecondRunIsIdempotent()
    {
        var graph = Model.Load(ZeroCopyModel(), runOptimizer: false)!;
        var first = GraphOptimizer.Run(graph);
        Assert.Contains(first, c => c.Pass == "reshape-zerocopy");
        int nodes = graph.Nodes.Count;
        var second = GraphOptimizer.Run(graph);
        Assert.DoesNotContain(second, c => c.Pass == "reshape-zerocopy");
        Assert.Equal(nodes, graph.Nodes.Count);
    }
}