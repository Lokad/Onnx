namespace Lokad.Onnx.Backend.Tests;

public class GraphFoldTransposeTests
{
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
