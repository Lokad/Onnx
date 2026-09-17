namespace Lokad.Onnx.Backend.Tests;

// E69 mask fusion: Add(scores, mask) feeding Softmax rewrites to the
// MaskedSoftmax fused op. Composite-first: the provider sequences the exact
// legacy Add then Softmax on non-conforming shapes, so these gates prove
// plumbing with identical values before any model-level timing.
public class GraphFusionMaskedSoftmaxTests
{
    static OnnxModel MaskFusionModel()
    {
        var mp = new OnnxModel { Name = "tiny-maskedsoftmax" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        mp.Inputs.Add(new OnnxValueInfo { Name = "m", ElementType = TensorElementType.Float, Dims = new[] { 4 } });
        mp.Inputs.Add(new OnnxValueInfo { Name = "xi", ElementType = TensorElementType.Double, Dims = new[] { 2, 4 } });
        mp.Inputs.Add(new OnnxValueInfo { Name = "mi", ElementType = TensorElementType.Double, Dims = new[] { 4 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y1", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y2a", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y2b", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y3", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "a3", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "yi", ElementType = TensorElementType.Double, Dims = new[] { 2, 4 } });
        Dictionary<string, object> NoAttrs() => new Dictionary<string, object>();
        Dictionary<string, object> AxisNeg1() => new Dictionary<string, object> { ["axis"] = -1 };
        mp.Nodes.Add(new OnnxNode { OpType = "Add", Inputs = new[] { "x", "m" }, Outputs = new[] { "a1" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Softmax", Inputs = new[] { "a1" }, Outputs = new[] { "y1" }, Attributes = AxisNeg1() });
        mp.Nodes.Add(new OnnxNode { OpType = "Add", Inputs = new[] { "x", "m" }, Outputs = new[] { "a2" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Softmax", Inputs = new[] { "a2" }, Outputs = new[] { "y2a" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Softmax", Inputs = new[] { "a2" }, Outputs = new[] { "y2b" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Add", Inputs = new[] { "x", "m" }, Outputs = new[] { "a3" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Softmax", Inputs = new[] { "a3" }, Outputs = new[] { "y3" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Add", Inputs = new[] { "xi", "mi" }, Outputs = new[] { "ai" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Softmax", Inputs = new[] { "ai" }, Outputs = new[] { "yi" }, Attributes = NoAttrs() });
        return mp;
    }

    [Fact]
    public void MaskRegionsFuseDecoysStay()
    {
        var graph = Model.Load(MaskFusionModel(), runOptimizer: false)!;
        var report = Optimization.GraphOptimizer.Run(graph);
        Assert.Contains(report, c => c.Pass == "maskedsoftmax");
        Assert.Contains(graph.Nodes, n => n.Op == OpType.MaskedSoftmax && n.Outputs.Length == 1 && n.Outputs[0] == "y1");
        var fused = graph.Nodes.First(n => n.Op == OpType.MaskedSoftmax && n.Outputs[0] == "y1");
        Assert.Equal(new[] { "x", "m" }, fused.Inputs);
        // Two live consumers keep the two-node form.
        Assert.Contains(graph.Nodes, n => n.Op == OpType.Add && n.Outputs.Length == 1 && n.Outputs[0] == "a2");
        // Graph-output exposure on the link keeps the two-node form.
        Assert.Contains(graph.Nodes, n => n.Op == OpType.Add && n.Outputs.Length == 1 && n.Outputs[0] == "a3");
        // Non-float link keeps the two-node form.
        Assert.Contains(graph.Nodes, n => n.Op == OpType.Add && n.Outputs.Length == 1 && n.Outputs[0] == "ai");
        Assert.DoesNotContain(graph.Nodes, n => n.Op == OpType.Softmax && n.Outputs.Length == 1 && n.Outputs[0] == "y1");
    }

    static System.Collections.Generic.Dictionary<string, ITensor> MaskFeed(int seed)
    {
        var rnd = new System.Random(seed);
        var x = new float[8];
        var m = new float[4];
        for (int i = 0; i < 8; i++) x[i] = (float)rnd.NextDouble() * 8f - 4f;
        for (int i = 0; i < 4; i++) m[i] = (rnd.NextDouble() < 0.5) ? float.NegativeInfinity : 0f;
        x[0] = System.BitConverter.Int32BitsToSingle(0x7FC00001);
        x[1] = float.PositiveInfinity;
        x[2] = -0f;
        m[0] = 0f;
        return new System.Collections.Generic.Dictionary<string, ITensor>
        {
            ["x"] = new DenseTensor<float>(new System.Memory<float>(x), new int[] { 2, 4 }),
            ["m"] = new DenseTensor<float>(new System.Memory<float>(m), new int[] { 4 }),
            ["xi"] = new DenseTensor<double>(new System.Memory<double>(new double[8]), new int[] { 2, 4 }),
            ["mi"] = new DenseTensor<double>(new System.Memory<double>(new double[4]), new int[] { 4 }),
        };
    }

    [Fact]
    public void MaskFusion_BitwiseVsLegacy()
    {
        var fused = Model.Load(MaskFusionModel())!;
        var plain = Model.Load(MaskFusionModel(), runOptimizer: false)!;
        var feed = MaskFeed(99);
        Assert.True(fused.Execute(feed, true), fused.LastErrorMessage);
        Assert.True(plain.Execute(feed, true), plain.LastErrorMessage);
        foreach (var name in new[] { "y1", "y2a", "y2b", "y3" })
        {
            var f = ((Tensor<float>)fused.Outputs[name]!).ToDenseTensor().Buffer.ToArray();
            var p = ((Tensor<float>)plain.Outputs[name]!).ToDenseTensor().Buffer.ToArray();
            Assert.Equal(p.Length, f.Length);
            Assert.True(p.AsSpan().SequenceEqual(f.AsSpan()), name + " diverged");
        }
    }
}


