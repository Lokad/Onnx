namespace Lokad.Onnx.Backend.Tests;

// E71 bias-epilogue fusion: MatMul feeding Add(data, rank-one-bias-initializer)
// rewrites to the MatMulBias fused op. Composite-first: the provider sequences
// the exact legacy MatMul then Add on non-conforming shapes, so these gates
// prove plumbing with identical values before any model-level timing.
public class GraphFusionMatMulBiasTests
{
    static OnnxModel BiasFusionModel()
    {
        var mp = new OnnxModel { Name = "tiny-matmulbias" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        mp.Inputs.Add(new OnnxValueInfo { Name = "t", ElementType = TensorElementType.Float, Dims = new[] { 2, 3 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y1", ElementType = TensorElementType.Float, Dims = new[] { 2, 3 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y2a", ElementType = TensorElementType.Float, Dims = new[] { 2, 3 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y2b", ElementType = TensorElementType.Float, Dims = new[] { 2, 3 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y3", ElementType = TensorElementType.Float, Dims = new[] { 2, 3 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "m3", ElementType = TensorElementType.Float, Dims = new[] { 2, 3 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y4", ElementType = TensorElementType.Float, Dims = new[] { 2, 3 } });
        mp.Initializers.Add(new OnnxTensor { Name = "w", ElementType = TensorElementType.Float, Dims = new[] { 4, 3 }, Data = new float[] { 1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f, 1f, 1f, 1f } });
        mp.Initializers.Add(new OnnxTensor { Name = "b", ElementType = TensorElementType.Float, Dims = new[] { 3 }, Data = new float[] { 0.5f, -1f, 2f } });
        Dictionary<string, object> NoAttrs() => new Dictionary<string, object>();
        mp.Nodes.Add(new OnnxNode { OpType = "MatMul", Inputs = new[] { "x", "w" }, Outputs = new[] { "m1" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Add", Inputs = new[] { "m1", "b" }, Outputs = new[] { "y1" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "MatMul", Inputs = new[] { "x", "w" }, Outputs = new[] { "m2" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Add", Inputs = new[] { "m2", "b" }, Outputs = new[] { "y2a" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Add", Inputs = new[] { "m2", "b" }, Outputs = new[] { "y2b" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "MatMul", Inputs = new[] { "x", "w" }, Outputs = new[] { "m3" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Add", Inputs = new[] { "m3", "b" }, Outputs = new[] { "y3" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "MatMul", Inputs = new[] { "x", "w" }, Outputs = new[] { "m4" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Add", Inputs = new[] { "m4", "t" }, Outputs = new[] { "y4" }, Attributes = NoAttrs() });
        return mp;
    }

    [Fact]
    public void BiasRegionsFuseDecoysStay()
    {
        var graph = Model.Load(BiasFusionModel(), runOptimizer: false)!;
        var report = Optimization.GraphOptimizer.Run(graph);
        Assert.Contains(report, c => c.Pass == "matmulbias");
        Assert.Contains(graph.Nodes, n => n.Op == OpType.MatMulBias && n.Outputs.Length == 1 && n.Outputs[0] == "y1");
        var fused = graph.Nodes.First(n => n.Op == OpType.MatMulBias && n.Outputs[0] == "y1");
        Assert.Equal(new[] { "x", "w", "b" }, fused.Inputs);
        // Two live consumers keep the two-node form.
        Assert.Contains(graph.Nodes, n => n.Op == OpType.MatMul && n.Outputs.Length == 1 && n.Outputs[0] == "m2");
        // Graph-output exposure on the link keeps the two-node form.
        Assert.Contains(graph.Nodes, n => n.Op == OpType.MatMul && n.Outputs.Length == 1 && n.Outputs[0] == "m3");
        // Dynamic-dynamic residual Add keeps the two-node form.
        Assert.Contains(graph.Nodes, n => n.Op == OpType.Add && n.Outputs.Length == 1 && n.Outputs[0] == "y4");
        Assert.DoesNotContain(graph.Nodes, n => n.Op == OpType.MatMulBias && n.Outputs.Length == 1 && n.Outputs[0] == "y4");
    }

    static System.Collections.Generic.Dictionary<string, ITensor> BiasFeed(int seed)
    {
        var rnd = new System.Random(seed);
        var x = new float[8];
        var t = new float[6];
        for (int i = 0; i < 8; i++) x[i] = (float)rnd.NextDouble() * 8f - 4f;
        for (int i = 0; i < 6; i++) t[i] = (float)rnd.NextDouble() * 8f - 4f;
        x[0] = System.BitConverter.Int32BitsToSingle(0x7FC00001);
        x[1] = float.PositiveInfinity;
        x[2] = -0f;
        return new System.Collections.Generic.Dictionary<string, ITensor>
        {
            ["x"] = new DenseTensor<float>(new System.Memory<float>(x), new int[] { 2, 4 }),
            ["t"] = new DenseTensor<float>(new System.Memory<float>(t), new int[] { 2, 3 }),
        };
    }

    [Fact]
    public void BiasFusion_BitwiseVsLegacy()
    {
        var fused = Model.Load(BiasFusionModel())!;
        var plain = Model.Load(BiasFusionModel(), runOptimizer: false)!;
        var feed = BiasFeed(99);
        Assert.True(fused.Execute(feed, true), fused.LastErrorMessage);
        Assert.True(plain.Execute(feed, true), plain.LastErrorMessage);
        foreach (var name in new[] { "y1", "y2a", "y2b", "y3", "y4" })
        {
            var f = ((Tensor<float>)fused.Outputs[name]!).ToDenseTensor().Buffer.ToArray();
            var p = ((Tensor<float>)plain.Outputs[name]!).ToDenseTensor().Buffer.ToArray();
            Assert.Equal(p.Length, f.Length);
            Assert.True(p.AsSpan().SequenceEqual(f.AsSpan()), name + " diverged");
        }
    }
}
