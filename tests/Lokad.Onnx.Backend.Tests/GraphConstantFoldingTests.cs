using System.Collections.Generic;

namespace Lokad.Onnx.Backend.Tests;

// Covers preparation-time folding of literal Constant nodes into
// initializers (GraphConstants): internal constants share attribute storage
// across runs instead of cloning per execution, with the exact safety
// posture of imported initializers. Graph outputs, fed names, collisions,
// sequences, and scalar spellings keep the per-execution clone.
public class GraphConstantFoldingTests
{
    static ComputationalGraph NewGraph()
    {
        var g = new ComputationalGraph();
        g.Metadata["Name"] = "const-fold";
        return g;
    }

    static void AddConstant(ComputationalGraph g, string node, string output, ITensor value)
    {
        g.Nodes.Add(new Node
        {
            Name = node,
            Op = OpType.Constant,
            Inputs = new string[0],
            Outputs = new[] { output },
            Attributes = new Dictionary<string, object> { { "value", value } },
        });
    }

    static void Add(ComputationalGraph g, string node, string left, string right, string output)
    {
        g.Nodes.Add(new Node { Name = node, Op = OpType.Add, Inputs = new[] { left, right }, Outputs = new[] { output } });
    }

    static float[] ToArray(ITensor t) => ((Tensor<float>)t).ToArray();

    [Fact]
    public void LiteralConstant_FoldsToInitializer()
    {
        var g = NewGraph();
        var attr = DenseTensor<float>.OfValues(new float[] { 1f, 2f });
        g.Inputs["x"] = DenseTensor<float>.OfValues(new float[] { 10f, 20f });
        g.Outputs["y"] = DenseTensor<float>.OfShape(2);
        AddConstant(g, "c", "c", attr);
        Add(g, "add", "c", "x", "y");
        g.IntermediateOutputs["c"] = null;
        g.RefreshLifetimeAnalysis();
        Assert.DoesNotContain(g.Nodes, n => n.Op == OpType.Constant);
        Assert.True(g.Initializers.ContainsKey("c"), "folded constant is missing.");
        Assert.Same(attr, g.Initializers["c"]);
        var inputs = new Dictionary<string, ITensor> { ["x"] = DenseTensor<float>.OfValues(new float[] { 10f, 20f }) };
        Assert.True(g.Execute(inputs, true), g.LastErrorMessage);
        Assert.Equal(new float[] { 11f, 22f }, ToArray(g.Outputs["y"]));
        Assert.True(g.Execute(inputs, true), g.LastErrorMessage);
        Assert.Equal(new float[] { 11f, 22f }, ToArray(g.Outputs["y"]));
        ((DenseTensor<float>)g.Outputs["y"]).Buffer.Span[0] = 999f;
        Assert.True(g.Execute(inputs, true), g.LastErrorMessage);
        Assert.Equal(new float[] { 11f, 22f }, ToArray(g.Outputs["y"]));
    }

    [Fact]
    public void GraphOutputConstant_KeepsNode()
    {
        var g = NewGraph();
        AddConstant(g, "c", "y", DenseTensor<float>.OfValues(new float[] { 1f, 2f }));
        g.Outputs["y"] = DenseTensor<float>.OfShape(2);
        g.RefreshLifetimeAnalysis();
        Assert.Contains(g.Nodes, n => n.Op == OpType.Constant);
        Assert.False(g.Initializers.ContainsKey("y"), "exposed constants must not share storage.");
        Assert.True(g.Execute(new Dictionary<string, ITensor>(), true), g.LastErrorMessage);
        Assert.Equal(new float[] { 1f, 2f }, ToArray(g.Outputs["y"]));
        ((DenseTensor<float>)g.Outputs["y"]).Buffer.Span[0] = 999f;
        Assert.True(g.Execute(new Dictionary<string, ITensor>(), true), g.LastErrorMessage);
        Assert.Equal(new float[] { 1f, 2f }, ToArray(g.Outputs["y"]));
    }

    [Fact]
    public void FedNameConstant_KeepsNode()
    {
        var g = NewGraph();
        g.Inputs["c"] = DenseTensor<float>.OfValues(new float[] { 7f, 8f });
        g.Inputs["x"] = DenseTensor<float>.OfValues(new float[] { 10f, 20f });
        g.Outputs["y"] = DenseTensor<float>.OfShape(2);
        AddConstant(g, "c", "c", DenseTensor<float>.OfValues(new float[] { 1f, 2f }));
        Add(g, "add", "c", "x", "y");
        g.IntermediateOutputs["c"] = null;
        g.RefreshLifetimeAnalysis();
        Assert.Contains(g.Nodes, n => n.Op == OpType.Constant);
        Assert.False(g.Initializers.ContainsKey("c"), "fed names must not gain initializers.");
    }

    [Fact]
    public void ScalarAttributeConstant_KeepsNode()
    {
        var g = NewGraph();
        g.Outputs["y"] = DenseTensor<float>.OfShape(1);
        g.Nodes.Add(new Node
        {
            Name = "c",
            Op = OpType.Constant,
            Inputs = new string[0],
            Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object> { { "value_float", 1f } },
        });
        g.RefreshLifetimeAnalysis();
        Assert.Contains(g.Nodes, n => n.Op == OpType.Constant);
    }

    [Fact]
    public void FoldedRun_SkipsPerRunClone()
    {
        const int n = 4 * 1024 * 1024;
        var folded = NewGraph();
        var attr = new DenseTensor<float>(new Memory<float>(new float[n]), new[] { n });
        for (int i = 0; i < n; i += 1024) attr.Buffer.Span[i] = 1f;
        folded.Inputs["x"] = new DenseTensor<float>(new Memory<float>(new float[n]), new[] { n });
        folded.Outputs["y"] = DenseTensor<float>.OfShape(n);
        AddConstant(folded, "c", "c", attr);
        Add(folded, "add", "c", "x", "y");
        folded.IntermediateOutputs["c"] = null;
        folded.RefreshLifetimeAnalysis();
        Assert.DoesNotContain(folded.Nodes, x => x.Op == OpType.Constant);
        var cloning = NewGraph();
        cloning.Inputs["x"] = new DenseTensor<float>(new Memory<float>(new float[n]), new[] { n });
        cloning.Outputs["y"] = DenseTensor<float>.OfShape(n);
        AddConstant(cloning, "c", "y2", attr);
        Add(cloning, "add", "y2", "x", "y");
        cloning.Outputs["y2"] = DenseTensor<float>.OfShape(n);
        cloning.IntermediateOutputs["y2"] = null;
        cloning.RefreshLifetimeAnalysis();
        Assert.Contains(cloning.Nodes, x => x.Op == OpType.Constant);
        var fx = new Dictionary<string, ITensor> { ["x"] = new DenseTensor<float>(new Memory<float>(new float[n]), new[] { n }) };
        var cx = new Dictionary<string, ITensor> { ["x"] = new DenseTensor<float>(new Memory<float>(new float[n]), new[] { n }) };
        Assert.True(folded.Execute(fx, true), folded.LastErrorMessage);
        Assert.True(cloning.Execute(cx, true), cloning.LastErrorMessage);
        GC.Collect(2, GCCollectionMode.Forced, true, true);
        GC.WaitForPendingFinalizers();
        long FoldedBytes()
        {
            long before = GC.GetAllocatedBytesForCurrentThread();
            Assert.True(folded.Execute(fx, true), folded.LastErrorMessage);
            Assert.True(folded.Execute(fx, true), folded.LastErrorMessage);
            return GC.GetAllocatedBytesForCurrentThread() - before;
        }
        long CloningBytes()
        {
            long before = GC.GetAllocatedBytesForCurrentThread();
            Assert.True(cloning.Execute(cx, true), cloning.LastErrorMessage);
            Assert.True(cloning.Execute(cx, true), cloning.LastErrorMessage);
            return GC.GetAllocatedBytesForCurrentThread() - before;
        }
        long foldedBytes = Math.Min(FoldedBytes(), FoldedBytes());
        long cloningBytes = Math.Min(CloningBytes(), CloningBytes());
        Assert.True(cloningBytes >= 4L * n * 4 - 4000000, "clone run must allocate two 16MB clones plus outputs, saw " + cloningBytes);
        Assert.True(foldedBytes <= 2L * n * 4 + 4000000, "folded run must allocate only the two Add outputs, saw " + foldedBytes);
    }

    [Fact]
    public void CustomDomainConstant_KeepsNode()
    {
        var g = NewGraph();
        g.Inputs["x"] = DenseTensor<float>.OfValues(new float[] { 10f, 20f });
        g.Outputs["y"] = DenseTensor<float>.OfShape(2);
        AddConstant(g, "c", "c", DenseTensor<float>.OfValues(new float[] { 1f, 2f }));
        var cn = g.Nodes[0];
        cn.Domain = "custom";
        g.Nodes[0] = cn;
        Add(g, "add", "c", "x", "y");
        g.IntermediateOutputs["c"] = null;
        g.RefreshLifetimeAnalysis();
        Assert.Contains(g.Nodes, n => n.Op == OpType.Constant);
        Assert.False(g.Initializers.ContainsKey("c"), "custom-domain constants must not fold.");
    }


    [Fact]
    public void ComputedChain_FoldsToInitializer()
    {
        var g = NewGraph();
        g.Opset[""] = 17;
        g.Inputs["x"] = DenseTensor<float>.OfValues(new float[,] { { 10f }, { 20f } });
        g.Outputs["y"] = DenseTensor<float>.OfShape(2, 1);
        AddConstant(g, "c1", "c1", DenseTensor<float>.OfValues(new float[] { 1f, 2f }));
        AddConstant(g, "c2", "c2", DenseTensor<float>.OfValues(new float[] { 3f, 4f }));
        g.Nodes.Add(new Node { Name = "cat", Op = OpType.Concat, Inputs = new[] { "c1", "c2" }, Outputs = new[] { "cc" }, Attributes = new Dictionary<string, object> { { "axis", 0L } } });
        AddConstant(g, "cs", "cs", DenseTensor<long>.OfValues(new long[] { 2L, 2L }));
        g.Nodes.Add(new Node { Name = "rs", Op = OpType.Reshape, Inputs = new[] { "cc", "cs" }, Outputs = new[] { "r" } });
        AddConstant(g, "st", "st", DenseTensor<long>.OfValues(new long[] { 0L, 0L }));
        AddConstant(g, "en", "en", DenseTensor<long>.OfValues(new long[] { 2L, 1L }));
        AddConstant(g, "ax", "ax", DenseTensor<long>.OfValues(new long[] { 0L, 1L }));
        g.Nodes.Add(new Node { Name = "sl", Op = OpType.Slice, Inputs = new[] { "r", "st", "en", "ax" }, Outputs = new[] { "s" } });
        Add(g, "add", "s", "x", "y");
        foreach (var n in new[] { "c1", "c2", "cc", "cs", "r", "st", "en", "ax", "s" }) g.IntermediateOutputs[n] = null;
        g.RefreshLifetimeAnalysis();
        Assert.DoesNotContain(g.Nodes, n => n.Op == OpType.Concat || n.Op == OpType.Reshape || n.Op == OpType.Slice || n.Op == OpType.Constant);
        Assert.True(g.Initializers.ContainsKey("s"), "folded chain output is missing.");
        Assert.Equal(new float[] { 1f, 3f }, ToArray(g.Initializers["s"]));
        var inputs = new Dictionary<string, ITensor> { ["x"] = DenseTensor<float>.OfValues(new float[,] { { 10f }, { 20f } }) };
        Assert.True(g.Execute(inputs, true), g.LastErrorMessage);
        Assert.Equal(new float[] { 11f, 23f }, ToArray(g.Outputs["y"]));
        Assert.True(g.Execute(inputs, true), g.LastErrorMessage);
        Assert.Equal(new float[] { 11f, 23f }, ToArray(g.Outputs["y"]));
    }

    [Fact]
    public void ComputedFold_DataDependent_KeepsNodes()
    {
        var g = NewGraph();
        g.Inputs["x"] = DenseTensor<float>.OfValues(new float[] { 10f, 20f });
        g.Outputs["y"] = DenseTensor<float>.OfShape(2);
        AddConstant(g, "c1", "c1", DenseTensor<float>.OfValues(new float[] { 1f, 2f }));
        g.Nodes.Add(new Node { Name = "cat", Op = OpType.Concat, Inputs = new[] { "c1", "x" }, Outputs = new[] { "cc" }, Attributes = new Dictionary<string, object> { { "axis", 0L } } });
        Add(g, "add", "cc", "cc", "y");
        g.IntermediateOutputs["c1"] = null;
        g.IntermediateOutputs["cc"] = null;
        g.RefreshLifetimeAnalysis();
        Assert.Contains(g.Nodes, n => n.Op == OpType.Concat);
    }

    [Fact]
    public void ComputedFold_AddStaysUnfolded()
    {
        var g = NewGraph();
        g.Outputs["y"] = DenseTensor<float>.OfShape(2);
        AddConstant(g, "c1", "c1", DenseTensor<float>.OfValues(new float[] { 1f, 2f }));
        AddConstant(g, "c2", "c2", DenseTensor<float>.OfValues(new float[] { 3f, 4f }));
        g.Nodes.Add(new Node { Name = "add", Op = OpType.Add, Inputs = new[] { "c1", "c2" }, Outputs = new[] { "y" } });
        g.IntermediateOutputs["c1"] = null;
        g.IntermediateOutputs["c2"] = null;
        g.RefreshLifetimeAnalysis();
        Assert.Contains(g.Nodes, n => n.Op == OpType.Add);
        Assert.True(g.Execute(new Dictionary<string, ITensor>(), true), g.LastErrorMessage);
        Assert.Equal(new float[] { 4f, 6f }, ToArray(g.Outputs["y"]));
    }

    [Fact]
    public void ComputedFold_RebuildsOnSourceReplace()
    {
        var g = NewGraph();
        g.Initializers["w"] = DenseTensor<float>.OfValues(new float[] { 1f, 2f });
        g.Outputs["y"] = DenseTensor<float>.OfShape(2, 1);
        AddConstant(g, "cs", "cs", DenseTensor<long>.OfValues(new long[] { 2L, 1L }));
        g.Nodes.Add(new Node { Name = "rs", Op = OpType.Reshape, Inputs = new[] { "w", "cs" }, Outputs = new[] { "r" } });
        Add(g, "add", "r", "r", "y");
        g.IntermediateOutputs["r"] = null;
        g.RefreshLifetimeAnalysis();
        Assert.True(g.Initializers.ContainsKey("r"), "folded reshape is missing.");
        Assert.Equal(new float[] { 1f, 2f }, ToArray(g.Initializers["r"]));
        g.Initializers["w"] = DenseTensor<float>.OfValues(new float[] { 5f, 6f });
        g.RefreshLifetimeAnalysis();
        Assert.Equal(new float[] { 5f, 6f }, ToArray(g.Initializers["r"]));
        var user = new Dictionary<string, ITensor>();
        Assert.True(g.Execute(user, true), g.LastErrorMessage);
        Assert.Equal(new float[] { 10f, 12f }, ToArray(g.Outputs["y"]));
        g.Initializers["r"] = DenseTensor<float>.OfValues(new float[] { 7f, 8f });
        g.RefreshLifetimeAnalysis();
        Assert.True(g.Execute(user, true), g.LastErrorMessage);
        Assert.Equal(new float[] { 14f, 16f }, ToArray(g.Outputs["y"]));
    }

    [Fact]
    public void ComputedFold_InvalidateRestoresNodes()
    {
        var g = NewGraph();
        g.Initializers["w"] = DenseTensor<float>.OfValues(new float[] { 1f, 2f });
        g.Outputs["y"] = DenseTensor<float>.OfShape(2, 1);
        AddConstant(g, "cs", "cs", DenseTensor<long>.OfValues(new long[] { 2L, 1L }));
        g.Nodes.Add(new Node { Name = "rs", Op = OpType.Reshape, Inputs = new[] { "w", "cs" }, Outputs = new[] { "r" } });
        Add(g, "add", "r", "r", "y");
        g.IntermediateOutputs["r"] = null;
        g.RefreshLifetimeAnalysis();
        Assert.DoesNotContain(g.Nodes, n => n.Op == OpType.Reshape);
        Assert.True(g.Initializers.ContainsKey("r"), "folded reshape is missing.");
        g.InvalidatePreparation();
        Assert.Contains(g.Nodes, n => n.Op == OpType.Reshape);
        Assert.False(g.Initializers.ContainsKey("r"), "folded value must drop on invalidate.");
        var user = new Dictionary<string, ITensor>();
        Assert.True(g.Execute(user, true), g.LastErrorMessage);
        Assert.Equal(new float[] { 2f, 4f }, ToArray(g.Outputs["y"]));
        g.RefreshLifetimeAnalysis();
        Assert.DoesNotContain(g.Nodes, n => n.Op == OpType.Reshape);
        Assert.Equal(new float[] { 1f, 2f }, ToArray(g.Initializers["r"]));
    }


    [SkippableFact]
    public void RealSegmentation_NoInternalConstantNodes()
    {
        var graph = ModelFixture.LoadRequiredModel("PyannoteSegmentation", "models", "speaker-diarization-community-1", "onnx", "segmentation", "model.onnx");
        graph.RefreshLifetimeAnalysis();
        foreach (var node in graph.Nodes)
        {
            if (node.Op != OpType.Constant) continue;
            Assert.True(node.Outputs.Length == 1 && graph.Outputs.ContainsKey(node.Outputs[0]),
                "internal Constant node survived folding: " + node.Name);
        }
    }
    [SkippableFact]
    public void Encoder_Folds336ComputedChains()
    {
        var graph = ModelFixture.LoadRequiredModel("ParakeetEncoder", "models", "parakeet-tdt-0.6b-v3", "onnx", "encoder-model.onnx");
        graph.RefreshLifetimeAnalysis();
        Assert.Equal(336, graph.FoldedComputationCount);
    }


}
