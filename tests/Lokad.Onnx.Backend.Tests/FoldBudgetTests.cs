using System.Collections.Generic;
using System.Linq;
using Lokad.Onnx.Optimization;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Q03 bounds: pack budget in bytes, fold pre-estimates with an aggregate cap,
/// and dedupe subgroups. Estimates are sound upper bounds; the post-execution
/// byte check stays as backstop.
/// </summary>
public class FoldBudgetTests
{
    [Theory]
    [InlineData(1, 1, true)]
    [InlineData(0, 5, false)]
    [InlineData(5, 0, false)]
    [InlineData(-1, 5, false)]
    [InlineData(4096, 1, false)]
    [InlineData(3072, 43690, true)]
    [InlineData(3072, 43691, false)]
    [InlineData(4095, 32768, true)]
    [InlineData(4095, 32776, true)]
    [InlineData(4095, 32777, false)]
    [InlineData(2048, 131072, false)]
    public void PackBudget_EnforcesByteUnits(int n, int k, bool expected)
    {
        Assert.Equal(expected, GraphPacking.FitsPackBudget(n, k));
    }

    static OnnxModel CappedModel(int chains, int floatsPerChain)
    {
        var mp = new OnnxModel { Name = "tiny-fold-cap" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { floatsPerChain } });
        for (int c = 0; c < chains; c++)
        {
            var data = new float[floatsPerChain];
            for (int i = 0; i < data.Length; i++) data[i] = c + i * 0.001f;
            var t = Model.ToTensor(new OnnxTensor { ElementType = TensorElementType.Float, Dims = new[] { floatsPerChain }, Data = data });
            string cn = "c" + c, tn = "t" + c, zn = "z" + c;
            mp.Outputs.Add(new OnnxValueInfo { Name = zn, ElementType = TensorElementType.Float, Dims = new[] { floatsPerChain } });
            mp.Nodes.Add(new OnnxNode { OpType = "Constant", Domain = "", Inputs = new string[0], Outputs = new[] { cn }, Attributes = new Dictionary<string, object> { ["value"] = t } });
            mp.Nodes.Add(new OnnxNode { OpType = "Transpose", Domain = "", Inputs = new[] { cn }, Outputs = new[] { tn }, Attributes = new Dictionary<string, object> { ["perm"] = new long[] { 0 } } });
            mp.Nodes.Add(new OnnxNode { OpType = "Mul", Domain = "", Inputs = new[] { "x", tn }, Outputs = new[] { zn }, Attributes = new Dictionary<string, object>() });
        }
        return mp;
    }

    [Fact]
    public void AggregateCap_StopsAfterEightMegabytes()
    {
        // 10 chains near 0.9MB each in one pass invocation: the first 9 fold
        // (8.29MB), the tenth stays. Later fixed-point rounds may fold leftovers;
        // the load-wide bound is rounds times the per-invocation cap.
        var graph = Model.Load(CappedModel(10, 230400), runOptimizer: false)!;
        var facts = GraphFacts.Build(graph);
        int folded = ConstFold.FoldConstants(graph, facts, new List<int>());
        Assert.True(folded > 0);
        Assert.Single(graph.Nodes, n => n.Op == OpType.Transpose && n.Outputs.Length == 1 && n.Outputs[0] == "t9");
        var feed = new Dictionary<string, ITensor> { ["x"] = DenseTensor<float>.OfValues(new float[230400]) };
        Assert.True(graph.Execute(feed, true), graph.LastErrorMessage);
        var got = ((Tensor<float>)graph.Outputs["z9"]).ToArray();
        Assert.Equal(230400, got.Length);
    }

    [Fact]
    public void Dedupe_MergesNonFirstSubgroup()
    {
        var mp = new OnnxModel { Name = "tiny-dedupe-sub" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { 2 } });
        var mk = new System.Func<string, float[], OnnxNode>((name, v) =>
        {
            var t = Model.ToTensor(new OnnxTensor { ElementType = TensorElementType.Float, Dims = new[] { 2 }, Data = v });
            return new OnnxNode { OpType = "Constant", Domain = "", Inputs = new string[0], Outputs = new[] { name }, Attributes = new Dictionary<string, object> { ["value"] = t } };
        });
        mp.Nodes.Add(mk("cA", new float[] { 1f, 2f }));
        mp.Nodes.Add(mk("cB", new float[] { 3f, 4f }));
        mp.Nodes.Add(mk("cC", new float[] { 3f, 4f }));
        int z = 0;
        foreach (var c in new[] { "cA", "cB", "cC" })
        {
            string zn = "z" + (z++);
            mp.Outputs.Add(new OnnxValueInfo { Name = zn, ElementType = TensorElementType.Float, Dims = new[] { 2 } });
            mp.Nodes.Add(new OnnxNode { OpType = "Mul", Domain = "", Inputs = new[] { "x", c }, Outputs = new[] { zn }, Attributes = new Dictionary<string, object>() });
        }
        var graph = Model.Load(mp)!;
        Assert.Equal(2, graph.Nodes.Count(n => n.Op == OpType.Constant));
        var feed = new Dictionary<string, ITensor> { ["x"] = DenseTensor<float>.OfValues(new float[] { 1f, 1f }) };
        Assert.True(graph.Execute(feed, true), graph.LastErrorMessage);
    }

    [Fact]
    public void GatherBlowupEstimate_SkipsBeforeAllocating()
    {
        var mp = new OnnxModel { Name = "tiny-gather-blowup" };
        mp.Opset[""] = 14;
        mp.Outputs.Add(new OnnxValueInfo { Name = "z", ElementType = TensorElementType.Float, Dims = new[] { 300000 } });
        var data = Model.ToTensor(new OnnxTensor { ElementType = TensorElementType.Float, Dims = new[] { 2 }, Data = new float[] { 1f, 2f } });
        var idx = new long[300000];
        var idxt = Model.ToTensor(new OnnxTensor { ElementType = TensorElementType.Int64, Dims = new[] { 300000 }, Data = idx });
        mp.Nodes.Add(new OnnxNode { OpType = "Constant", Domain = "", Inputs = new string[0], Outputs = new[] { "d" }, Attributes = new Dictionary<string, object> { ["value"] = data } });
        mp.Nodes.Add(new OnnxNode { OpType = "Constant", Domain = "", Inputs = new string[0], Outputs = new[] { "i" }, Attributes = new Dictionary<string, object> { ["value"] = idxt } });
        mp.Nodes.Add(new OnnxNode { OpType = "Gather", Domain = "", Inputs = new[] { "d", "i" }, Outputs = new[] { "z" }, Attributes = new Dictionary<string, object>() });
        var graph = Model.Load(mp)!;
        Assert.Contains(graph.Nodes, n => n.Op == OpType.Gather);
        Assert.True(graph.Execute(new Dictionary<string, ITensor>(), true), graph.LastErrorMessage);
        var got = ((Tensor<float>)graph.Outputs["z"]).ToArray();
        Assert.Equal(300000, got.Length);
        Assert.Equal(1f, got[0]);
        Assert.Equal(1f, got[299999]);
    }
}
