using Lokad.Onnx.Optimization;

namespace Lokad.Onnx.Backend.Tests;

[Collection("NonParallelAllocation")]
public class HarmonicSumFusionTests
{
    static Node Node(OpType op, string[] inputs, string output, Dictionary<string, object>? attributes = null) => new()
    {
        Name = output, Op = op, OpTypeName = op.ToString(), Domain = "", OpsetVersion = 18,
        Inputs = inputs, Outputs = [output], Attributes = attributes
    };

    static Dictionary<string, ITensor> Feed(int width, bool special = false)
    {
        float[] phases = Enumerable.Range(0, 2 * width).Select(i => (i % 137 - 68) * .03125f).ToArray();
        float[] amplitude = Enumerable.Range(0, 2 * width).Select(i => (i % 7 - 3) * .25f).ToArray();
        if (special && width > 0)
        {
            float[] cycle = [-0f, 0f, float.NaN, float.PositiveInfinity, float.NegativeInfinity, float.MaxValue, float.Epsilon];
            for (int i = 0; i < phases.Length; i++) phases[i] = cycle[i % cycle.Length];
        }
        return new()
        {
            ["pitch"] = new DenseTensor<float>(Enumerable.Range(0, 2 * width).Select(i => (i % 11 - 5) * .125f).ToArray(), new[] { 2, 1, width }),
            ["harmonics"] = new DenseTensor<float>(new[] { 0f, 1f, 2f, 3f, 4f, -1f, .5f }, new[] { 1, 7, 1 }),
            ["limit"] = new DenseTensor<float>(new[] { 1f }, Array.Empty<int>()),
            ["amplitude"] = new DenseTensor<float>(amplitude, new[] { 2, 1, width }),
            ["frequencies"] = new DenseTensor<float>(new[] { 0f, .125f, .25f, 1f, 2f, 4f, 8f }, new[] { 1, 7, 1 }),
            ["phase"] = new DenseTensor<float>(phases, new[] { 2, 1, width })
        };
    }

    static ComputationalGraph Graph(Dictionary<string, ITensor> feed, long[]? axes = null, bool keep = false, bool noop = false)
    {
        var graph = new ComputationalGraph(); graph.Metadata["Name"] = "harmonic-sum"; graph.Opset[""] = 18;
        foreach (var pair in feed) graph.Inputs[pair.Key] = pair.Value;
        graph.Initializers["axes"] = new DenseTensor<long>(axes ?? [-2, 1], new[] { (axes ?? [-2, 1]).Length });
        graph.Outputs["y"] = feed["phase"];
        graph.Nodes.Add(Node(OpType.Mul, ["pitch", "harmonics"], "frequency"));
        graph.Nodes.Add(Node(OpType.Less, ["frequency", "limit"], "less"));
        graph.Nodes.Add(Node(OpType.Cast, ["less"], "mask", new() { ["to"] = (int)TensorElementType.Float }));
        graph.Nodes.Add(Node(OpType.Mul, ["mask", "amplitude"], "gain"));
        graph.Nodes.Add(Node(OpType.Mul, ["frequencies", "phase"], "argument"));
        graph.Nodes.Add(Node(OpType.Sin, ["argument"], "sine"));
        graph.Nodes.Add(Node(OpType.Mul, ["sine", "gain"], "product"));
        graph.Nodes.Add(Node(OpType.ReduceSum, ["product", "axes"], "y",
            new() { ["keepdims"] = keep ? 1 : 0, ["noop_with_empty_axes"] = noop ? 1 : 0 }));
        return graph;
    }

    static Tensor<float> Run(ComputationalGraph graph, Dictionary<string, ITensor> feed, ExecutionOptions options)
    {
        Assert.True(graph.Execute(feed, true, ExecutionProvider.CPU, options), graph.LastErrorMessage);
        return (Tensor<float>)graph.Outputs["y"];
    }

    [Theory]
    [InlineData(0)] [InlineData(1)] [InlineData(7)] [InlineData(8)] [InlineData(16)]
    [InlineData(19)] [InlineData(32)] [InlineData(264)] [InlineData(35400)]
    public void DefaultAndFallbackModesMatchOriginalArithmetic(int width)
    {
        foreach (bool special in new[] { false, true })
        foreach (bool keep in new[] { false, true })
        foreach (var options in new[] { ExecutionOptions.Default, ExecutionOptions.Simd, ExecutionOptions.Scalar, ExecutionOptions.Memory,
            new ExecutionOptions(OptimizationMode.Speed, new TensorExecutionOptions(true, false, 2)) })
        {
            var feed = Feed(width, special); var before = feed.ToDictionary(p => p.Key, p => ((Tensor<float>)p.Value).ToArray());
            var plain = Graph(feed, keep: keep); var fused = Graph(feed, keep: keep);
            Assert.Equal(1, GraphFusion.FuseHarmonicSum(fused, GraphFacts.Build(fused)));
            Assert.Equal(OpType.FusedHarmonicSum, Assert.Single(fused.Nodes).Op);
            var expected = Run(plain, feed, options); var actual = Run(fused, feed, options); var saved = actual.ToArray();
            EqualBits(expected.ToArray(), saved); Assert.Equal(expected.Dimensions.ToArray(), actual.Dimensions.ToArray());
            foreach (var pair in feed)
            {
                var tensor = (Tensor<float>)pair.Value;
                Assert.False(TensorAlias.SharesBackingMemory(tensor, actual));
                EqualBits(before[pair.Key], tensor.ToArray());
                if (tensor.Length == 0) continue;
                float previous = tensor.GetValue(0); tensor.SetValue(0, .375f);
                EqualBits(Run(plain, feed, options).ToArray(), Run(fused, feed, options).ToArray());
                EqualBits(saved, actual.ToArray()); tensor.SetValue(0, previous);
            }
        }
    }

    [Fact]
    public void OtherAxesLayoutsThresholdShapesAndNoOpUsePrimitives()
    {
        foreach (string variant in new[] { "reversed", "slice", "threshold-rank", "axes0", "axes2", "empty-axes", "noop" })
        {
            var feed = Feed(32);
            if (variant == "reversed") feed["phase"] = new DenseTensor<float>(((Tensor<float>)feed["phase"]).ToArray(), new[] { 2, 1, 32 }, true);
            if (variant == "slice")
                foreach (string name in new[] { "pitch", "phase", "amplitude" })
                    feed[name] = new TensorSlice<float>((Tensor<float>)feed[name], [SliceIndex.All, SliceIndex.All, new SliceIndex(0, 32, 2)]);
            if (variant == "threshold-rank") feed["limit"] = new DenseTensor<float>(new[] { .5f }, new[] { 1, 1, 1, 1 });
            long[] axes = variant switch { "axes0" => [0], "axes2" => [2], "empty-axes" or "noop" => [], _ => [1] };
            var plain = Graph(feed, axes, noop: variant == "noop"); var fused = Graph(feed, axes, noop: variant == "noop");
            Assert.Equal(1, GraphFusion.FuseHarmonicSum(fused, GraphFacts.Build(fused)));
            EqualBits(Run(plain, feed, ExecutionOptions.Default).ToArray(), Run(fused, feed, ExecutionOptions.Default).ToArray());
            Assert.Equal(plain.Outputs["y"].Dims, fused.Outputs["y"].Dims);
        }
    }

    [Theory]
    [InlineData("visible")] [InlineData("shared")] [InlineData("captured")]
    [InlineData("domain")] [InlineData("attribute")] [InlineData("cast")]
    [InlineData("unknown")] [InlineData("rebind")] [InlineData("commuted")]
    public void UnprovedOrObservablePatternsRemainUnchanged(string reason)
    {
        var feed = Feed(8); var graph = Graph(feed); var node = graph.Nodes[0];
        switch (reason)
        {
            case "visible": graph.Outputs["gain"] = feed["phase"]; break;
            case "shared": graph.Nodes.Add(Node(OpType.Identity, ["argument"], "other")); break;
            case "captured":
                var branch = new ComputationalGraph(); branch.Nodes.Add(Node(OpType.Identity, ["mask"], "z")); branch.Outputs["z"] = feed["phase"];
                var conditional = Node(OpType.If, ["condition"], "other", new() { ["then_branch"] = branch, ["else_branch"] = branch });
                graph.Nodes.Add(conditional); break;
            case "domain": node.Domain = "custom"; graph.Nodes[0] = node; break;
            case "attribute": node.Attributes = new() { ["unexpected"] = 1 }; graph.Nodes[0] = node; break;
            case "cast": graph.Nodes[2].Attributes!["to"] = (int)TensorElementType.Double; break;
            case "unknown": graph.Inputs["phase"] = null!; break;
            case "rebind": graph.Nodes.Insert(2, Node(OpType.Identity, ["pitch"], "phase")); break;
            case "commuted": node = graph.Nodes[6]; node.Inputs = ["gain", "sine"]; graph.Nodes[6] = node; break;
        }
        Assert.Equal(0, GraphFusion.FuseHarmonicSum(graph, GraphFacts.Build(graph)));
    }

    [Fact]
    public void AlignedKernelAllocatesOnlyTheSmallOutput()
    {
        if (!System.Numerics.Vector.IsHardwareAccelerated) return;
        var feed = Feed(8192); var axes = DenseTensor<int>.OfValues(new[] { 1 });
        OpResult Run() => CPUExecutionProvider.HarmonicSum(feed["pitch"], feed["harmonics"], feed["limit"], feed["amplitude"],
            feed["frequencies"], feed["phase"], axes, 0, 0, ExecutionOptions.Simd);
        for (int i = 0; i < 3; i++) Run();
        long before = GC.GetAllocatedBytesForCurrentThread(); var result = Run();
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.Equal(new[] { 2, 8192 }, Assert.Single(result.Outputs).Dims);
        Assert.True(allocated < 100000, $"Allocated {allocated} bytes for a 64 KB output.");
    }

    static void EqualBits(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            if (float.IsNaN(expected[i])) Assert.True(float.IsNaN(actual[i]), $"NaN at {i}");
            else Assert.Equal(BitConverter.SingleToInt32Bits(expected[i]), BitConverter.SingleToInt32Bits(actual[i]));
    }
}
