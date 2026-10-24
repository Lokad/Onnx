namespace Lokad.Onnx.Backend.Tests;

public class GraphFoldTransposeTests
{
    static ComputationalGraph MatrixTranspose(bool context)
    {
        var model = new OnnxModel { Name = "fold-lifecycle", Opset = { [""] = 13 },
            Initializers = { new OnnxTensor { Name = "w", ElementType = TensorElementType.Float,
                Dims = [2, 2], Data = new float[] { 1, 2, 3, 4 } } },
            Outputs = { new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Float, Dims = [2, 2] } },
            Nodes = { new OnnxNode { Name = "transpose", OpType = "Transpose", Inputs = ["w"], Outputs = ["y"],
                Attributes = new() { ["perm"] = new int[] { 1, 0 } } } } };
        var graph = Model.Load(model);
        return context ? graph.CreateExecution(ExecutionOptions.Scalar) : graph;
    }

    static Tensor<float> RunMatrix(ComputationalGraph graph)
    {
        Assert.True(graph.Execute(NoInputs(), true), graph.LastErrorMessage);
        return Assert.IsAssignableFrom<Tensor<float>>(graph.Outputs["y"]);
    }

    [Theory]
    [InlineData(false, false, false)] [InlineData(false, false, true)]
    [InlineData(false, true, false)] [InlineData(false, true, true)]
    [InlineData(true, false, false)] [InlineData(true, false, true)]
    [InlineData(true, true, false)] [InlineData(true, true, true)]
    public void ChangedPermutationUsesCurrentAxes(bool context, bool prepare, bool editArray)
    {
        var graph = MatrixTranspose(context);
        var held = RunMatrix(graph);
        if (editArray)
        {
            var axes = (int[])graph.Nodes[0].Attributes!["perm"];
            axes[0] = 0; axes[1] = 1;
        }
        else graph.Nodes[0].Attributes!["perm"] = new int[] { 0, 1 };
        if (prepare) graph.Prepare();
        Assert.Equal(new float[] { 1, 2, 3, 4 }, RunMatrix(graph).ToArray());
        Assert.Equal(new float[] { 1, 3, 2, 4 }, held.ToArray());
    }

    [Theory]
    [InlineData(false)] [InlineData(true)]
    public void EquivalentDefaultPermutationAndUnchangedPreparationReuseFold(bool context)
    {
        var graph = MatrixTranspose(context);
        var held = RunMatrix(graph);
        graph.Nodes[0].Attributes!.Remove("perm");
        graph.Prepare();
        Assert.Same(held, RunMatrix(graph));
        graph.Prepare();
        Assert.Same(held, RunMatrix(graph));
    }

    [Theory]
    [InlineData(false)] [InlineData(true)]
    public void InvalidPermutationDoesNotReturnAnOldFold(bool context)
    {
        var graph = MatrixTranspose(context);
        var held = RunMatrix(graph);
        graph.Nodes[0].Attributes!["perm"] = new int[] { 0, 0 };
        Assert.False(graph.Execute(NoInputs(), true));
        Assert.Equal("transpose", graph.LastFailedNodeName);
        Assert.Equal(new float[] { 1, 3, 2, 4 }, held.ToArray());
    }

    [Theory]
    [InlineData(false, "rename")] [InlineData(true, "rename")]
    [InlineData(false, "remove")] [InlineData(true, "remove")]
    [InlineData(false, "replace")] [InlineData(true, "replace")]
    public void ChangedNodesDropObsoleteFoldedPayloads(bool context, string edit)
    {
        var graph = MatrixTranspose(context);
        var held = RunMatrix(graph);
        if (edit == "rename")
            for (int i = 0; i < 20; i++)
            {
                var node = graph.Nodes[0]; node.Name = "transpose-" + i; graph.Nodes[0] = node;
                graph.Prepare();
                Assert.Single(graph.Initializers.Keys.Where(n => n.StartsWith("folded:")));
            }
        else
        {
            if (edit == "remove")
            {
                graph.Nodes.Clear(); graph.OutputDescs.Clear(); graph.Outputs.Clear();
            }
            else
            {
                var node = graph.Nodes[0]; node.Op = OpType.Identity; node.OpTypeName = "Identity";
                node.Attributes = null; graph.Nodes[0] = node;
            }
            graph.Prepare();
            Assert.DoesNotContain(graph.Initializers.Keys, n => n.StartsWith("folded:"));
        }
        Assert.Equal(new float[] { 1, 3, 2, 4 }, held.ToArray());
        Assert.True(graph.Execute(NoInputs(), true), graph.LastErrorMessage);
    }

    [Theory]
    [InlineData(false, false)] [InlineData(false, true)]
    [InlineData(true, false)] [InlineData(true, true)]
    public void CacheCleanupPreservesCallerReplacement(bool context, bool invalidate)
    {
        var graph = MatrixTranspose(context);
        var held = RunMatrix(graph);
        var replacement = DenseTensor<float>.OfValues(new float[,] { { 9, 9 }, { 9, 9 } });
        graph.Initializers["folded:transpose"] = replacement;
        if (invalidate) graph.InvalidatePreparation();
        else
        {
            var node = graph.Nodes[0]; node.Name = "renamed"; graph.Nodes[0] = node;
            graph.Prepare();
        }
        Assert.Same(replacement, graph.Initializers["folded:transpose"]);
        Assert.Equal(new float[] { 1, 3, 2, 4 }, RunMatrix(graph).ToArray());
        Assert.Equal(new float[] { 1, 3, 2, 4 }, held.ToArray());
    }

    [Theory]
    [InlineData(false)] [InlineData(true)]
    public void CallerReplacementIsNotMistakenForCachedOutput(bool context)
    {
        var graph = MatrixTranspose(context);
        RunMatrix(graph);
        graph.Initializers["folded:transpose"] = DenseTensor<float>.OfValues(new float[,] { { 9, 9 }, { 9, 9 } });
        Assert.Equal(new float[] { 1, 3, 2, 4 }, RunMatrix(graph).ToArray());
    }

    static ComputationalGraph NewGraph()
    {
        var g = new ComputationalGraph();
        g.Metadata["Name"] = "fold";
        return g;
    }

    static void AddTranspose(ComputationalGraph g, string node, string input, string output)
    {
        g.Nodes.Add(new Node { Name = node, Op = OpType.Transpose, Inputs = new[] { input }, Outputs = new[] { output } });
    }

    static Dictionary<string, ITensor> NoInputs()
    {
        return new Dictionary<string, ITensor>();
    }

    [Fact]
    public void ConstantTranspose_FoldsAndReuses()
    {
        var g = NewGraph();
        g.Initializers["w"] = DenseTensor<float>.OfValues(new float[2, 3] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        g.Outputs["zt"] = DenseTensor<float>.OfShape(3, 2);
        AddTranspose(g, "t", "w", "zt");
        g.RefreshLifetimeAnalysis();
        Assert.True(g.Execute(NoInputs(), true), g.LastErrorMessage);
        var expected = new float[] { 1f, 4f, 2f, 5f, 3f, 6f };
        Assert.Equal(expected, ((Tensor<float>)g.Outputs["zt"]).ToArray());
        Assert.True(g.Initializers.ContainsKey("folded:t"));
        g.Reset();
        Assert.True(g.Execute(NoInputs(), true), g.LastErrorMessage);
        var second = (Tensor<float>)g.Outputs["zt"];
        Assert.Same(g.Initializers["folded:t"], second);
        Assert.Equal(expected, second.ToArray());
    }

    [Fact]
    public void ConstantTranspose_FoldsEagerlyAtRefresh()
    {
        var g = NewGraph();
        g.Initializers["w"] = DenseTensor<float>.OfValues(new float[2, 3] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        g.Outputs["zt"] = DenseTensor<float>.OfShape(3, 2);
        AddTranspose(g, "t", "w", "zt");
        g.RefreshLifetimeAnalysis();
        Assert.True(g.Initializers.ContainsKey("folded:t"));
        Assert.Equal(new float[] { 1f, 4f, 2f, 5f, 3f, 6f }, ((Tensor<float>)g.Initializers["folded:t"]).ToArray());
    }

    [Fact]
    public void ReplacedInitializer_Refolds()
    {
        var g = NewGraph();
        g.Initializers["w"] = DenseTensor<float>.OfValues(new float[2, 3] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        g.Outputs["zt"] = DenseTensor<float>.OfShape(3, 2);
        AddTranspose(g, "t", "w", "zt");
        g.RefreshLifetimeAnalysis();
        Assert.True(g.Execute(NoInputs(), true), g.LastErrorMessage);
        var w2 = DenseTensor<float>.OfValues(new float[2, 3] { { 10f, 20f, 30f }, { 40f, 50f, 60f } });
        g.Initializers["w"] = w2;
        g.Reset();
        Assert.True(g.Execute(NoInputs(), true), g.LastErrorMessage);
        Assert.Equal(new float[] { 10f, 40f, 20f, 50f, 30f, 60f }, ((Tensor<float>)g.Outputs["zt"]).ToArray());
        Assert.Equal(new float[] { 10f, 40f, 20f, 50f, 30f, 60f }, ((Tensor<float>)g.Initializers["folded:t"]).ToArray());
        g.Reset();
        Assert.True(g.Execute(NoInputs(), true), g.LastErrorMessage);
        Assert.Same(g.Initializers["folded:t"], g.Outputs["zt"]);
    }

    [Fact]
    public void ComputedInput_NeverFolds()
    {
        var g = NewGraph();
        g.Inputs["x"] = DenseTensor<float>.OfShape(2, 3);
        g.Outputs["zt"] = DenseTensor<float>.OfShape(3, 2);
        g.Nodes.Add(new Node { Name = "r", Op = OpType.Relu, Inputs = new[] { "x" }, Outputs = new[] { "h" } });
        AddTranspose(g, "t", "h", "zt");
        g.RefreshLifetimeAnalysis();
        var user = new Dictionary<string, ITensor>
        {
            { "x", DenseTensor<float>.OfValues(new float[2, 3] { { 1f, -2f, 3f }, { -4f, 5f, -6f } }) },
        };
        Assert.True(g.Execute(user, true), g.LastErrorMessage);
        Assert.Equal(new float[] { 1f, 0f, 0f, 5f, 3f, 0f }, ((Tensor<float>)g.Outputs["zt"]).ToArray());
        Assert.False(g.Initializers.ContainsKey("folded:t"));
    }

    [Fact]
    public void InvalidatePreparation_DropsFold()
    {
        var g = NewGraph();
        g.Initializers["w"] = DenseTensor<float>.OfValues(new float[2, 3] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
        g.Outputs["zt"] = DenseTensor<float>.OfShape(3, 2);
        AddTranspose(g, "t", "w", "zt");
        g.RefreshLifetimeAnalysis();
        Assert.True(g.Execute(NoInputs(), true), g.LastErrorMessage);
        Assert.True(g.Initializers.ContainsKey("folded:t"));
        g.InvalidatePreparation();
        Assert.False(g.Initializers.ContainsKey("folded:t"));
        g.Reset();
        Assert.True(g.Execute(NoInputs(), true), g.LastErrorMessage);
        Assert.True(g.Initializers.ContainsKey("folded:t"));
        Assert.Equal(new float[] { 1f, 4f, 2f, 5f, 3f, 6f }, ((Tensor<float>)g.Outputs["zt"]).ToArray());
    }
}
