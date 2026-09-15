namespace Lokad.Onnx.Backend.Tests;
/// <summary>
/// Pins the Gemm+GELU epilogue fusion: direct, view-Reshape, and tanh-attribute
/// chains fuse with bitwise twins of fused versus pass-disabled execution, while
/// multi-consumer links and graph-output exposure decline.
/// </summary>
public class GraphFusionGemmGeluTests
{
    static OnnxModel GemmGeluModel()
    {
        var mp = new OnnxModel { Name = "tiny-gemmgelu" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(new OnnxValueInfo { Name = "a1", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        mp.Inputs.Add(new OnnxValueInfo { Name = "a2", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        mp.Inputs.Add(new OnnxValueInfo { Name = "a3", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        mp.Inputs.Add(new OnnxValueInfo { Name = "a4", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        mp.Inputs.Add(new OnnxValueInfo { Name = "a5", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        mp.Inputs.Add(new OnnxValueInfo { Name = "a6", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        mp.Inputs.Add(new OnnxValueInfo { Name = "a7", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y1", ElementType = TensorElementType.Float, Dims = new[] { 2, 3 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y2", ElementType = TensorElementType.Float, Dims = new[] { 2, 3 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y3", ElementType = TensorElementType.Float, Dims = new[] { 1, 2, 3 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y4", ElementType = TensorElementType.Float, Dims = new[] { 2, 3 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y5", ElementType = TensorElementType.Float, Dims = new[] { 2, 3 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "yb2", ElementType = TensorElementType.Float, Dims = new[] { 2, 3 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "g6", ElementType = TensorElementType.Float, Dims = new[] { 2, 3 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y6", ElementType = TensorElementType.Float, Dims = new[] { 2, 3 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y7", ElementType = TensorElementType.Float, Dims = new[] { 1, 2, 3 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "yb7", ElementType = TensorElementType.Float, Dims = new[] { 1, 2, 3 } });
        var rnd = new System.Random(71);
        float[] RF(int n, float s)
        {
            var a = new float[n];
            for (int i = 0; i < n; i++) a[i] = (float)rnd.NextDouble() * s;
            return a;
        }
        mp.Initializers.Add(new OnnxTensor { Name = "W1", ElementType = TensorElementType.Float, Dims = new[] { 4, 3 }, Data = RF(12, 0.5f) });
        mp.Initializers.Add(new OnnxTensor { Name = "C1", ElementType = TensorElementType.Float, Dims = new[] { 3 }, Data = RF(3, 0.5f) });
        mp.Initializers.Add(new OnnxTensor { Name = "W2", ElementType = TensorElementType.Float, Dims = new[] { 4, 3 }, Data = RF(12, 0.5f) });
        mp.Initializers.Add(new OnnxTensor { Name = "W3", ElementType = TensorElementType.Float, Dims = new[] { 4, 3 }, Data = RF(12, 0.5f) });
        mp.Initializers.Add(new OnnxTensor { Name = "C3", ElementType = TensorElementType.Float, Dims = new[] { 3 }, Data = RF(3, 0.5f) });
        mp.Initializers.Add(new OnnxTensor { Name = "S3", ElementType = TensorElementType.Int64, Dims = new[] { 3 }, Data = new long[] { 1, 2, 3 } });
        mp.Initializers.Add(new OnnxTensor { Name = "ones3", ElementType = TensorElementType.Float, Dims = new[] { 1, 2, 3 }, Data = new float[] { 1f, 1f, 1f, 1f, 1f, 1f } });
        mp.Initializers.Add(new OnnxTensor { Name = "W4", ElementType = TensorElementType.Float, Dims = new[] { 4, 3 }, Data = RF(12, 0.5f) });
        mp.Initializers.Add(new OnnxTensor { Name = "W5", ElementType = TensorElementType.Float, Dims = new[] { 4, 3 }, Data = RF(12, 0.5f) });
        mp.Initializers.Add(new OnnxTensor { Name = "W6", ElementType = TensorElementType.Float, Dims = new[] { 4, 3 }, Data = RF(12, 0.5f) });
        mp.Initializers.Add(new OnnxTensor { Name = "W7", ElementType = TensorElementType.Float, Dims = new[] { 4, 3 }, Data = RF(12, 0.5f) });
        mp.Initializers.Add(new OnnxTensor { Name = "C7", ElementType = TensorElementType.Float, Dims = new[] { 3 }, Data = RF(3, 0.5f) });
        mp.Initializers.Add(new OnnxTensor { Name = "S7", ElementType = TensorElementType.Int64, Dims = new[] { 3 }, Data = new long[] { 1, 2, 3 } });
        System.Collections.Generic.Dictionary<string, object> NoAttrs()
        {
            return new System.Collections.Generic.Dictionary<string, object>();
        }
        mp.Nodes.Add(new OnnxNode { OpType = "Gemm", Inputs = new[] { "a1", "W1", "C1" }, Outputs = new[] { "g1" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Gelu", Inputs = new[] { "g1" }, Outputs = new[] { "y1" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Gemm", Inputs = new[] { "a2", "W2" }, Outputs = new[] { "g2" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Gelu", Inputs = new[] { "g2" }, Outputs = new[] { "y2" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Gemm", Inputs = new[] { "a3", "W3", "C3" }, Outputs = new[] { "g3" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Reshape", Inputs = new[] { "g3", "S3" }, Outputs = new[] { "r3" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Gelu", Inputs = new[] { "r3" }, Outputs = new[] { "h3" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Add", Inputs = new[] { "h3", "ones3" }, Outputs = new[] { "y3" }, Attributes = NoAttrs() });
        var tanhAttrs = new System.Collections.Generic.Dictionary<string, object>();
        tanhAttrs["approximate"] = "tanh";
        mp.Nodes.Add(new OnnxNode { OpType = "Gemm", Inputs = new[] { "a4", "W4" }, Outputs = new[] { "g4" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Gelu", Inputs = new[] { "g4" }, Outputs = new[] { "y4" }, Attributes = tanhAttrs });
        mp.Nodes.Add(new OnnxNode { OpType = "Gemm", Inputs = new[] { "a5", "W5" }, Outputs = new[] { "g5" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Gelu", Inputs = new[] { "g5" }, Outputs = new[] { "y5" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Add", Inputs = new[] { "g5", "g5" }, Outputs = new[] { "yb2" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Gemm", Inputs = new[] { "a6", "W6" }, Outputs = new[] { "g6" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Gelu", Inputs = new[] { "g6" }, Outputs = new[] { "y6" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Gemm", Inputs = new[] { "a7", "W7", "C7" }, Outputs = new[] { "g7" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Reshape", Inputs = new[] { "g7", "S7" }, Outputs = new[] { "r7" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Gelu", Inputs = new[] { "r7" }, Outputs = new[] { "y7" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Add", Inputs = new[] { "r7", "r7" }, Outputs = new[] { "yb7" }, Attributes = NoAttrs() });
        return mp;
    }
    [Fact]
    public void RegionsFuseDecoysStay()
    {
        var graph = Model.Load(GemmGeluModel(), runOptimizer: false)!;
        var report = Optimization.GraphOptimizer.Run(graph);
        Assert.Contains(report, c => c.Pass == "gemmgelu");
        Assert.Contains(graph.Nodes, n => n.Op == OpType.GemmGelu && n.Outputs.Length == 1 && n.Outputs[0] == "y1");
        Assert.Contains(graph.Nodes, n => n.Op == OpType.GemmGelu && n.Outputs.Length == 1 && n.Outputs[0] == "y2");
        Assert.DoesNotContain(graph.Nodes, n => n.Op == OpType.Gelu && n.Outputs.Length == 1 && (n.Outputs[0] == "y1" || n.Outputs[0] == "y2"));
        Assert.Contains(graph.Nodes, n => n.Op == OpType.Reshape && n.Outputs.Length == 1 && n.Outputs[0] == "r3");
        Assert.DoesNotContain(graph.Nodes, n => n.Op == OpType.Gelu && n.Outputs.Length == 1 && n.Outputs[0] == "h3");
        Node add3 = default;
        bool found3 = false;
        foreach (var n in graph.Nodes)
            if (n.Op == OpType.Add && n.Outputs.Length == 1 && n.Outputs[0] == "y3") { add3 = n; found3 = true; }
        Assert.True(found3, "y3 Add missing");
        bool rewired = false;
        if (add3.Inputs is not null)
            foreach (var inp in add3.Inputs)
                if (inp == "r3") rewired = true;
        Assert.True(rewired, "y3 Add still consumes the dropped Gelu output");
        Assert.Contains(graph.Nodes, n => n.Op == OpType.GemmGelu && n.Outputs.Length == 1 && n.Outputs[0] == "y4");
        Assert.DoesNotContain(graph.Nodes, n => n.Op == OpType.Gelu && n.Outputs.Length == 1 && n.Outputs[0] == "y4");
        Assert.Contains(graph.Nodes, n => n.Op == OpType.Gelu && n.Outputs.Length == 1 && n.Outputs[0] == "y5");
        Assert.Contains(graph.Nodes, n => n.Op == OpType.Gelu && n.Outputs.Length == 1 && n.Outputs[0] == "y6");
        Assert.Contains(graph.Nodes, n => n.Op == OpType.Gelu && n.Outputs.Length == 1 && n.Outputs[0] == "y7");
        Assert.DoesNotContain(graph.Nodes, n => n.Op == OpType.GemmGelu && n.Outputs.Length == 1 && (n.Outputs[0] == "y5" || n.Outputs[0] == "y6" || n.Outputs[0] == "y7" || n.Outputs[0] == "g7"));
    }
    static System.Collections.Generic.Dictionary<string, ITensor> Feed(int seed)
    {
        var rnd = new System.Random(seed);
        var feed = new System.Collections.Generic.Dictionary<string, ITensor>();
        foreach (var name in new[] { "a1", "a2", "a3", "a4", "a5", "a6", "a7" })
        {
            var v = new float[8];
            for (int i = 0; i < 8; i++) v[i] = (float)rnd.NextDouble() * 4f - 2f;
            feed[name] = new DenseTensor<float>(new System.Memory<float>(v), new int[] { 2, 4 });
        }
        feed["a1"].SetValue(0, System.BitConverter.Int32BitsToSingle(0x7FC00001));
        feed["a3"].SetValue(1, System.BitConverter.Int32BitsToSingle(0x7FC00002));
        feed["a3"].SetValue(2, float.PositiveInfinity);
        return feed;
    }
    static void AssertOutputsBitwise(ComputationalGraph fused, ComputationalGraph plain, System.Collections.Generic.Dictionary<string, ITensor> feed)
    {
        Assert.True(fused.Execute(feed, true), fused.LastErrorMessage);
        Assert.True(plain.Execute(feed, true), plain.LastErrorMessage);
        foreach (var name in new[] { "y1", "y2", "y3", "y4", "y5", "yb2", "g6", "y6", "y7", "yb7" })
        {
            var fa = ((Tensor<float>)fused.Outputs[name]).ToArray();
            var pa = ((Tensor<float>)plain.Outputs[name]).ToArray();
            Assert.Equal(pa.Length, fa.Length);
            for (int i = 0; i < pa.Length; i++)
                Assert.True(System.BitConverter.SingleToInt32Bits(pa[i]) == System.BitConverter.SingleToInt32Bits(fa[i]), name + " differs at " + i);
        }
    }
    [Fact]
    public void FusedMatchesUnfusedTwinBitwise()
    {
        var fused = Model.Load(GemmGeluModel())!;
        var plain = Model.Load(GemmGeluModel(), runOptimizer: false)!;
        AssertOutputsBitwise(fused, plain, Feed(81));
        AssertOutputsBitwise(fused, plain, Feed(82));
    }
    [Fact]
    public void DisabledKeepsPairs()
    {
        var graph = Model.Load(GemmGeluModel(), runOptimizer: false)!;
        var report = Optimization.GraphOptimizer.Run(graph, new[] { "gemmgelu" });
        Assert.DoesNotContain(report, c => c.Pass == "gemmgelu");
        Assert.Contains(graph.Nodes, n => n.Op == OpType.Gelu && n.Outputs.Length == 1 && n.Outputs[0] == "y1");
    }
    [Fact]
    public void SecondRunIsIdempotent()
    {
        var graph = Model.Load(GemmGeluModel(), runOptimizer: false)!;
        var first = Optimization.GraphOptimizer.Run(graph);
        Assert.Contains(first, c => c.Pass == "gemmgelu");
        int nodes = graph.Nodes.Count;
        var second = Optimization.GraphOptimizer.Run(graph);
        Assert.DoesNotContain(second, c => c.Pass == "gemmgelu");
        Assert.Equal(nodes, graph.Nodes.Count);
    }
}
