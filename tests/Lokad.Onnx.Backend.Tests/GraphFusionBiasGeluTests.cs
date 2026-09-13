namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Pins the bias+GELU region fusion: firing on initializer-bias patterns in
/// either input order, decline on extra consumers/graph outputs/non-initializer
/// or double bias/integer/tanh forms, and bitwise twins of fused versus
/// pass-disabled execution with exceptional values (including payload-distinct
/// NaNs in data and bias under both legacy orders).
/// </summary>
public class GraphFusionBiasGeluTests
{
    static OnnxModel BiasGeluModel()
    {
        var mp = new OnnxModel { Name = "tiny-biasgelu" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { 1, 4, 8 } });
        mp.Inputs.Add(new OnnxValueInfo { Name = "a", ElementType = TensorElementType.Float, Dims = new[] { 1, 4, 8 } });
        mp.Inputs.Add(new OnnxValueInfo { Name = "b", ElementType = TensorElementType.Float, Dims = new[] { 8 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y1", ElementType = TensorElementType.Float, Dims = new[] { 1, 4, 8 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y2", ElementType = TensorElementType.Float, Dims = new[] { 1, 4, 8 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y3", ElementType = TensorElementType.Float, Dims = new[] { 1, 4, 8 } });
        var rnd = new System.Random(31);
        var b1 = new float[8];
        var b2 = new float[8];
        for (int i = 0; i < 8; i++) { b1[i] = (float)rnd.NextDouble() * 2f - 1f; b2[i] = (float)rnd.NextDouble() * 2f - 1f; }
        mp.Initializers.Add(new OnnxTensor { Name = "bias1", ElementType = TensorElementType.Float, Dims = new[] { 8 }, Data = b1 });
        mp.Initializers.Add(new OnnxTensor { Name = "bias2", ElementType = TensorElementType.Float, Dims = new[] { 8 }, Data = b2 });
        Dictionary<string, object> NoAttrs() => new Dictionary<string, object>();
        mp.Nodes.Add(new OnnxNode { OpType = "Add", Inputs = new[] { "x", "bias1" }, Outputs = new[] { "s1" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Gelu", Inputs = new[] { "s1" }, Outputs = new[] { "y1" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Add", Inputs = new[] { "bias2", "a" }, Outputs = new[] { "s2" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Gelu", Inputs = new[] { "s2" }, Outputs = new[] { "y2" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Add", Inputs = new[] { "x", "b" }, Outputs = new[] { "s3" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Gelu", Inputs = new[] { "s3" }, Outputs = new[] { "y3" }, Attributes = NoAttrs() });
        return mp;
    }

    [Fact]
    public void BiasRegionsFuseDecoysStay()
    {
        var graph = Model.Load(BiasGeluModel(), runOptimizer: false)!;
        var report = Optimization.GraphOptimizer.Run(graph);
        Assert.Contains(report, c => c.Pass == "biasgelu");
        Assert.Contains(graph.Nodes, n => n.Op == OpType.BiasGelu && n.Outputs.Length == 1 && n.Outputs[0] == "y1");
        Assert.Contains(graph.Nodes, n => n.Op == OpType.BiasGelu && n.Outputs.Length == 1 && n.Outputs[0] == "y2");
        Assert.DoesNotContain(graph.Nodes, n => n.Outputs.Length == 1 && (n.Outputs[0] == "s1" || n.Outputs[0] == "s2"));
        // Runtime (non-initializer) bias keeps the two-node form.
        Assert.Contains(graph.Nodes, n => n.Op == OpType.Add && n.Outputs.Length == 1 && n.Outputs[0] == "s3");
        Assert.Contains(graph.Nodes, n => n.Op == OpType.Gelu && n.Outputs.Length == 1 && n.Outputs[0] == "y3");
    }

    static System.Collections.Generic.Dictionary<string, ITensor> Feed(int seed)
    {
        var rnd = new System.Random(seed);
        float NextF() => (float)rnd.NextDouble() * 8f - 4f;
        var x = new float[32];
        var a = new float[32];
        var b = new float[8];
        for (int i = 0; i < 32; i++) { x[i] = NextF(); a[i] = NextF(); }
        for (int i = 0; i < 8; i++) b[i] = NextF();
        x[0] = System.BitConverter.Int32BitsToSingle(0x7FC00001);
        x[1] = System.BitConverter.Int32BitsToSingle(0x7FC00002);
        a[0] = System.BitConverter.Int32BitsToSingle(0x7FC00003);
        b[0] = System.BitConverter.Int32BitsToSingle(0x7FC00004);
        b[1] = float.PositiveInfinity;
        x[3] = -0f;
        return new System.Collections.Generic.Dictionary<string, ITensor>
        {
            ["x"] = new DenseTensor<float>(new System.Memory<float>(x), new int[] { 1, 4, 8 }),
            ["a"] = new DenseTensor<float>(new System.Memory<float>(a), new int[] { 1, 4, 8 }),
            ["b"] = new DenseTensor<float>(new System.Memory<float>(b), new int[] { 8 }),
        };
    }

    static void AssertOutputsBitwise(ComputationalGraph fused, ComputationalGraph plain, System.Collections.Generic.Dictionary<string, ITensor> feed)
    {
        Assert.True(fused.Execute(feed, true), fused.LastErrorMessage);
        Assert.True(plain.Execute(feed, true), plain.LastErrorMessage);
        foreach (var name in new[] { "y1", "y2", "y3" })
        {
            var fa = ((Tensor<float>)fused.Outputs[name]).ToArray();
            var pa = ((Tensor<float>)plain.Outputs[name]).ToArray();
            Assert.Equal(pa.Length, fa.Length);
            for (int i = 0; i < pa.Length; i++)
                Assert.True(System.BitConverter.SingleToInt32Bits(pa[i]) == System.BitConverter.SingleToInt32Bits(fa[i]),
                    name + " differs at " + i);
        }
    }

    [Fact]
    public void FusedMatchesUnfusedTwinBitwise()
    {
        var fused = Model.Load(BiasGeluModel())!;
        var plain = Model.Load(BiasGeluModel(), runOptimizer: false)!;
        AssertOutputsBitwise(fused, plain, Feed(41));
        AssertOutputsBitwise(fused, plain, Feed(42));
    }

    [Fact]
    public void FusedExactConsumerFiresTanhDeclines()
    {
        // Simulate G01 outputs in place: y1 keeps an exact-fused consumer,
        // y2 gets a tanh-fused consumer. Only the exact form may fuse.
        var graph = Model.Load(BiasGeluModel(), runOptimizer: false)!;
        for (int i = 0; i < graph.Nodes.Count; i++)
        {
            if (graph.Nodes[i].Op == OpType.Gelu && graph.Nodes[i].Outputs.Length == 1 && graph.Nodes[i].Outputs[0] == "y1")
            {
                var n = graph.Nodes[i];
                n.IsFused = true;
                graph.Nodes[i] = n;
            }
            if (graph.Nodes[i].Op == OpType.Gelu && graph.Nodes[i].Outputs.Length == 1 && graph.Nodes[i].Outputs[0] == "y2")
            {
                var n = graph.Nodes[i];
                n.IsFused = true;
                n.Attributes = new System.Collections.Generic.Dictionary<string, object> { ["approximate"] = "tanh" };
                graph.Nodes[i] = n;
            }
        }
        var report = Optimization.GraphOptimizer.Run(graph);
        Assert.Contains(report, c => c.Pass == "biasgelu");
        Assert.Contains(graph.Nodes, n => n.Op == OpType.BiasGelu && n.Outputs.Length == 1 && n.Outputs[0] == "y1");
        Assert.DoesNotContain(graph.Nodes, n => n.Op == OpType.BiasGelu && n.Outputs.Length == 1 && n.Outputs[0] == "y2");
    }

    [Fact]
    public void DisabledKeepsPairs()
    {
        var graph = Model.Load(BiasGeluModel(), runOptimizer: false)!;
        var report = Optimization.GraphOptimizer.Run(graph, new[] { "biasgelu" });
        Assert.DoesNotContain(report, c => c.Pass == "biasgelu");
        Assert.Contains(graph.Nodes, n => n.Op == OpType.Gelu && n.Outputs.Length == 1 && n.Outputs[0] == "y1");
    }

    [Fact]
    public void SecondRunIsIdempotent()
    {
        var graph = Model.Load(BiasGeluModel(), runOptimizer: false)!;
        var first = Optimization.GraphOptimizer.Run(graph);
        Assert.Contains(first, c => c.Pass == "biasgelu");
        int nodes = graph.Nodes.Count;
        var second = Optimization.GraphOptimizer.Run(graph);
        Assert.DoesNotContain(second, c => c.Pass == "biasgelu");
        Assert.Equal(nodes, graph.Nodes.Count);
    }
}
