namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Pins the Conv+Relu and Add+Relu epilogue fusions: firing on minimal
/// patterns, decline on extra consumers/graph outputs/non-relu consumers/
/// integer dtypes, double-precision firing, and bitwise twins of fused
/// versus pass-disabled execution (same cores by construction, proven here).
/// </summary>
public class GraphFusionEpilogueTests
{
    static OnnxModel EpilogueModel()
    {
        var mp = new OnnxModel { Name = "tiny-epilogue" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { 1, 2, 4, 4 } });
        mp.Inputs.Add(new OnnxValueInfo { Name = "a", ElementType = TensorElementType.Float, Dims = new[] { 1, 3, 4, 4 } });
        mp.Inputs.Add(new OnnxValueInfo { Name = "b", ElementType = TensorElementType.Float, Dims = new[] { 1, 3, 4, 4 } });
        mp.Inputs.Add(new OnnxValueInfo { Name = "ai", ElementType = TensorElementType.Int32, Dims = new[] { 1, 2 } });
        mp.Inputs.Add(new OnnxValueInfo { Name = "bi", ElementType = TensorElementType.Int32, Dims = new[] { 1, 2 } });
        mp.Inputs.Add(new OnnxValueInfo { Name = "xd", ElementType = TensorElementType.Double, Dims = new[] { 1, 1, 3, 3 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y1", ElementType = TensorElementType.Float, Dims = new[] { 1, 3, 4, 4 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y2", ElementType = TensorElementType.Float, Dims = new[] { 1, 3, 4, 4 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "yd", ElementType = TensorElementType.Float, Dims = new[] { 1, 3, 4, 4 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "cD", ElementType = TensorElementType.Float, Dims = new[] { 1, 3, 4, 4 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "yi", ElementType = TensorElementType.Int32, Dims = new[] { 1, 2 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "yw", ElementType = TensorElementType.Double, Dims = new[] { 1, 2, 3, 3 } });
        var wf = new float[3 * 2 * 3 * 3];
        var bf = new float[3];
        var rnd = new System.Random(11);
        for (int i = 0; i < wf.Length; i++) wf[i] = (float)rnd.NextDouble() * 2f - 1f;
        for (int i = 0; i < bf.Length; i++) bf[i] = (float)rnd.NextDouble() * 2f - 1f;
        mp.Initializers.Add(new OnnxTensor { Name = "wf", ElementType = TensorElementType.Float, Dims = new[] { 3, 2, 3, 3 }, Data = wf });
        mp.Initializers.Add(new OnnxTensor { Name = "bf", ElementType = TensorElementType.Float, Dims = new[] { 3 }, Data = bf });
        var wd = new double[2 * 1 * 3 * 3];
        for (int i = 0; i < wd.Length; i++) wd[i] = rnd.NextDouble() * 2 - 1;
        mp.Initializers.Add(new OnnxTensor { Name = "wd", ElementType = TensorElementType.Double, Dims = new[] { 2, 1, 3, 3 }, Data = wd });
        var bd = new double[2];
        for (int i = 0; i < bd.Length; i++) bd[i] = rnd.NextDouble() * 2 - 1;
        mp.Initializers.Add(new OnnxTensor { Name = "bd", ElementType = TensorElementType.Double, Dims = new[] { 2 }, Data = bd });
        Dictionary<string, object> ConvAttrs() => new Dictionary<string, object>
        {
            ["kernel_shape"] = new long[] { 3, 3 },
            ["pads"] = new long[] { 1, 1, 1, 1 },
            ["strides"] = new long[] { 1, 1 },
        };
        Dictionary<string, object> NoAttrs() => new Dictionary<string, object>();
        mp.Nodes.Add(new OnnxNode { OpType = "Conv", Inputs = new[] { "x", "wf", "bf" }, Outputs = new[] { "cA" }, Attributes = ConvAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Relu", Inputs = new[] { "cA" }, Outputs = new[] { "y1" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Add", Inputs = new[] { "a", "b" }, Outputs = new[] { "aB" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Relu", Inputs = new[] { "aB" }, Outputs = new[] { "y2" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Conv", Inputs = new[] { "x", "wf", "bf" }, Outputs = new[] { "cD" }, Attributes = ConvAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Relu", Inputs = new[] { "cD" }, Outputs = new[] { "yd" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Add", Inputs = new[] { "ai", "bi" }, Outputs = new[] { "aI" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Relu", Inputs = new[] { "aI" }, Outputs = new[] { "yi" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Conv", Inputs = new[] { "xd", "wd", "bd" }, Outputs = new[] { "cW" }, Attributes = ConvAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Relu", Inputs = new[] { "cW" }, Outputs = new[] { "yw" }, Attributes = NoAttrs() });
        return mp;
    }

    [Fact]
    public void AddReluFiresThroughFusedPredecessors()
    {
        // ResNet bottleneck tail shape: the Add inputs arrive through
        // already-fused Conv+Relu nodes, so dtype facts must see through
        // fused natives or the epilogue starves (regression coverage).
        var mp = new OnnxModel { Name = "tiny-chain" };
        mp.Opset[""] = 14;
        mp.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { 1, 2, 4, 4 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Float, Dims = new[] { 1, 2, 4, 4 } });
        var wf = new float[2 * 2 * 3 * 3];
        var rnd = new System.Random(5);
        for (int i = 0; i < wf.Length; i++) wf[i] = (float)rnd.NextDouble() * 2f - 1f;
        mp.Initializers.Add(new OnnxTensor { Name = "w", ElementType = TensorElementType.Float, Dims = new[] { 2, 2, 3, 3 }, Data = wf });
        Dictionary<string, object> ConvAttrs() => new Dictionary<string, object>
        {
            ["kernel_shape"] = new long[] { 3, 3 },
            ["pads"] = new long[] { 1, 1, 1, 1 },
            ["strides"] = new long[] { 1, 1 },
        };
        Dictionary<string, object> NoAttrs() => new Dictionary<string, object>();
        mp.Nodes.Add(new OnnxNode { OpType = "Conv", Inputs = new[] { "x", "w" }, Outputs = new[] { "c1" }, Attributes = ConvAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Relu", Inputs = new[] { "c1" }, Outputs = new[] { "r1" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Conv", Inputs = new[] { "r1", "w" }, Outputs = new[] { "c2" }, Attributes = ConvAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Add", Inputs = new[] { "c2", "x" }, Outputs = new[] { "s" }, Attributes = NoAttrs() });
        mp.Nodes.Add(new OnnxNode { OpType = "Relu", Inputs = new[] { "s" }, Outputs = new[] { "y" }, Attributes = NoAttrs() });
        var graph = Model.Load(mp, runOptimizer: false)!;
        var report = Optimization.GraphOptimizer.Run(graph);
        Assert.Contains(report, c => c.Pass == "convrelu");
        Assert.Contains(report, c => c.Pass == "addrelu");
        Assert.Contains(graph.Nodes, n => n.Op == OpType.AddRelu && n.Outputs.Length == 1 && n.Outputs[0] == "y");
    }
    [Fact]
    public void EpiloguesFuseDecoysStay()
    {
        var graph = Model.Load(EpilogueModel(), runOptimizer: false)!;
        var report = Optimization.GraphOptimizer.Run(graph);
        Assert.Contains(report, c => c.Pass == "convrelu");
        Assert.Contains(report, c => c.Pass == "addrelu");
        // cA and aB collapse into fused nodes producing the Relu outputs.
        Assert.Contains(graph.Nodes, n => n.Op == OpType.ConvRelu && n.Outputs.Length == 1 && n.Outputs[0] == "y1");
        Assert.Contains(graph.Nodes, n => n.Op == OpType.AddRelu && n.Outputs.Length == 1 && n.Outputs[0] == "y2");
        Assert.Contains(graph.Nodes, n => n.Op == OpType.ConvRelu && n.Outputs.Length == 1 && n.Outputs[0] == "yw");
        // cD is a graph output (raw) so its Relu stays unfused.
        Assert.Contains(graph.Nodes, n => n.Op == OpType.Conv && n.Outputs.Length == 1 && n.Outputs[0] == "cD");
        Assert.Contains(graph.Nodes, n => n.Op == OpType.Relu && n.Outputs.Length == 1 && n.Outputs[0] == "yd");
        // Integer Add+Relu keeps the two-node form.
        Assert.Contains(graph.Nodes, n => n.Op == OpType.Add && n.Outputs.Length == 1 && n.Outputs[0] == "aI");
        Assert.Contains(graph.Nodes, n => n.Op == OpType.Relu && n.Outputs.Length == 1 && n.Outputs[0] == "yi");
        Assert.DoesNotContain(graph.Nodes, n => n.Outputs.Length == 1 && (n.Outputs[0] == "cA" || n.Outputs[0] == "aB" || n.Outputs[0] == "cW"));
    }

    static System.Collections.Generic.Dictionary<string, ITensor> Feed(int seed)
    {
        var rnd = new System.Random(seed);
        float NextF() => (float)rnd.NextDouble() * 2f - 1f;
        var x = new float[1 * 2 * 4 * 4];
        var a = new float[1 * 3 * 4 * 4];
        var b = new float[1 * 3 * 4 * 4];
        for (int i = 0; i < x.Length; i++) x[i] = NextF();
        for (int i = 0; i < a.Length; i++) a[i] = NextF();
        for (int i = 0; i < b.Length; i++) b[i] = NextF();
        var ai = new int[2];
        var bi = new int[2];
        for (int i = 0; i < 2; i++) { ai[i] = rnd.Next(-5, 6); bi[i] = rnd.Next(-5, 6); }
        var xd = new double[1 * 1 * 3 * 3];
        for (int i = 0; i < xd.Length; i++) xd[i] = rnd.NextDouble() * 2 - 1;
        // Exceptional values ride along in fixed lanes every run.
        x[0] = float.PositiveInfinity;
        x[1] = float.NegativeInfinity;
        x[2] = float.NaN;
        x[3] = -0f;
        a[0] = float.NaN;
        xd[0] = double.PositiveInfinity;
        xd[1] = double.NaN;
        return new System.Collections.Generic.Dictionary<string, ITensor>
        {
            ["x"] = new DenseTensor<float>(new System.Memory<float>(x), new int[] { 1, 2, 4, 4 }),
            ["a"] = new DenseTensor<float>(new System.Memory<float>(a), new int[] { 1, 3, 4, 4 }),
            ["b"] = new DenseTensor<float>(new System.Memory<float>(b), new int[] { 1, 3, 4, 4 }),
            ["ai"] = new DenseTensor<int>(new System.Memory<int>(ai), new int[] { 1, 2 }),
            ["bi"] = new DenseTensor<int>(new System.Memory<int>(bi), new int[] { 1, 2 }),
            ["xd"] = new DenseTensor<double>(new System.Memory<double>(xd), new int[] { 1, 1, 3, 3 }),
        };
    }

    static void AssertOutputsBitwise(ComputationalGraph fused, ComputationalGraph plain, System.Collections.Generic.Dictionary<string, ITensor> feed)
    {
        Assert.True(fused.Execute(feed, true), fused.LastErrorMessage);
        Assert.True(plain.Execute(feed, true), plain.LastErrorMessage);
        foreach (var name in new[] { "y1", "y2", "yd", "cD", "yi", "yw" })
        {
            var fa = fused.Outputs[name]!.ToArray();
            var pa = plain.Outputs[name]!.ToArray();
            Assert.Equal(pa.Length, fa.Length);
            for (int i = 0; i < pa.Length; i++)
                Assert.True(System.BitConverter.DoubleToInt64Bits(System.Convert.ToDouble(pa.GetValue(i))) == System.BitConverter.DoubleToInt64Bits(System.Convert.ToDouble(fa.GetValue(i))),
                    name + " differs at " + i);
        }
    }

    [Fact]
    public void FusedMatchesUnfusedTwinBitwise()
    {
        var fused = Model.Load(EpilogueModel())!;
        var plain = Model.Load(EpilogueModel(), runOptimizer: false)!;
        AssertOutputsBitwise(fused, plain, Feed(21));
        AssertOutputsBitwise(fused, plain, Feed(22));
    }

    [Fact]
    public void DisabledKeepsPairs()
    {
        var graph = Model.Load(EpilogueModel(), runOptimizer: false)!;
        var report = Optimization.GraphOptimizer.Run(graph, new[] { "convrelu", "addrelu" });
        Assert.DoesNotContain(report, c => c.Pass == "convrelu" || c.Pass == "addrelu");
        Assert.Contains(graph.Nodes, n => n.Op == OpType.Relu && n.Outputs.Length == 1 && n.Outputs[0] == "y1");
        Assert.Contains(graph.Nodes, n => n.Op == OpType.Relu && n.Outputs.Length == 1 && n.Outputs[0] == "y2");
    }

    [Fact]
    public void SecondRunIsIdempotent()
    {
        var graph = Model.Load(EpilogueModel(), runOptimizer: false)!;
        var first = Optimization.GraphOptimizer.Run(graph);
        Assert.Contains(first, c => c.Pass == "convrelu" || c.Pass == "addrelu");
        int nodes = graph.Nodes.Count;
        var second = Optimization.GraphOptimizer.Run(graph);
        Assert.DoesNotContain(second, c => c.Pass == "convrelu" || c.Pass == "addrelu");
        Assert.Equal(nodes, graph.Nodes.Count);
    }
}
