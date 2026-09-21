using Google.Protobuf;
using Onnx;

namespace Lokad.Onnx.Backend.Tests;

// The invalid-budget contract reads OnnxImport's process-wide error state.
[Collection("SequentialLogSink")]
public class PackedWeightBudgetTests
{
    static ComputationalGraph Graph(long budget, params int[] widths)
    {
        var graph = new ComputationalGraph(budget);
        graph.Metadata["Name"] = "packed-budget";
        graph.Inputs["x"] = DenseTensor<float>.OfValues(new float[,] { { 1, 2 }, { 3, 4 } });
        for (int i = 0; i < widths.Length; i++)
        {
            var weight = new DenseTensor<float>(Enumerable.Range(1, 2 * widths[i]).Select(x => (float)x).ToArray(), new[] { 2, widths[i] });
            graph.Initializers["w" + i] = weight;
            graph.Outputs["y" + i] = Tensor<float>.Zeros(2, widths[i]);
            graph.Nodes.Add(new Node { Name = "mm" + i, Op = OpType.MatMul, OpTypeName = "MatMul",
                Inputs = new[] { "x", "w" + i }, Outputs = new[] { "y" + i } });
        }
        return graph;
    }

    [Theory]
    [InlineData(0, 0)]
    [InlineData(15, 0)]
    [InlineData(16, 1)]
    [InlineData(31, 1)]
    [InlineData(32, 2)]
    [InlineData(long.MaxValue, 2)]
    public void AggregateBudget_ControlsPreparationWithoutChangingResults(long budget, int packedCount)
    {
        var graph = Graph(budget, 2, 2);
        var input = Assert.IsType<DenseTensor<float>>(graph.Inputs["x"]);
        var source = Assert.IsType<DenseTensor<float>>(graph.Initializers["w0"]);
        float[] inputBefore = input.ToArray(), weightBefore = source.ToArray();
        Assert.True(graph.Execute(new Dictionary<string, ITensor> { ["x"] = input }, true), graph.LastErrorMessage);
        Assert.Equal(packedCount * 16L, graph.RetainedPackedWeightBytes);
        Assert.Equal(packedCount, graph.Initializers.Keys.Count(x => x.StartsWith("packed:", StringComparison.Ordinal)));
        foreach (string name in new[] { "y0", "y1" })
            Assert.Equal(new float[] { 7, 10, 15, 22 }, Assert.IsAssignableFrom<Tensor<float>>(graph.Outputs[name]).ToArray());
        Assert.Equal(inputBefore, input.ToArray());
        Assert.Equal(weightBefore, source.ToArray());
        var context = graph.CreateExecution(ExecutionOptions.Default);
        Assert.Equal(budget, context.MaximumPackedWeightBytes);
        Assert.Equal(graph.RetainedPackedWeightBytes, context.RetainedPackedWeightBytes);
        Assert.Same(graph.Initializers, context.Initializers);
        Assert.True(context.Execute(new Dictionary<string, ITensor> { ["x"] = input }, true), context.LastErrorMessage);
        Assert.Equal(packedCount * 16L, graph.RetainedPackedWeightBytes);
    }

    [Fact]
    public void OversizedCandidate_DoesNotPreventLaterSmallerWeights()
    {
        var graph = Graph(16, 3, 2, 1);
        graph.RefreshLifetimeAnalysis();
        Assert.False(graph.Initializers.ContainsKey("packed:w0"));
        Assert.True(graph.Initializers.ContainsKey("packed:w1"));
        Assert.False(graph.Initializers.ContainsKey("packed:w2"));
        Assert.Equal(16, graph.RetainedPackedWeightBytes);
        var clone = graph.Initializers["packed:w1"];
        graph.RefreshLifetimeAnalysis();
        Assert.Same(clone, graph.Initializers["packed:w1"]);
        Assert.Equal(16, graph.RetainedPackedWeightBytes);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(4224)]
    public void PackedAndFallbackKernels_AgreeWithIndependentProduct(long budget)
    {
        var graph = new ComputationalGraph(budget);
        graph.Metadata["Name"] = "packed-budget-kernel";
        float[] x = Enumerable.Range(0, 8 * 24).Select(i => (float)Math.Sin(i)).ToArray();
        float[] w = Enumerable.Range(0, 24 * 44).Select(i => (float)Math.Cos(i)).ToArray();
        float[] xBefore = (float[])x.Clone(), wBefore = (float[])w.Clone();
        graph.Inputs["x"] = new DenseTensor<float>(x, new[] { 8, 24 });
        graph.Initializers["w"] = new DenseTensor<float>(w, new[] { 24, 44 });
        graph.Outputs["y"] = Tensor<float>.Zeros(8, 44);
        graph.Nodes.Add(new Node { Name = "mm", Op = OpType.MatMul, OpTypeName = "MatMul",
            Inputs = new[] { "x", "w" }, Outputs = new[] { "y" } });
        Assert.True(graph.Execute(new Dictionary<string, ITensor> { ["x"] = graph.Inputs["x"] }, true), graph.LastErrorMessage);
        float[] actual = Assert.IsAssignableFrom<Tensor<float>>(graph.Outputs["y"]).ToArray();
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 44; j++)
            {
                double expected = 0;
                for (int k = 0; k < 24; k++) expected += (double)x[i * 24 + k] * w[k * 44 + j];
                Assert.True(Math.Abs(actual[i * 44 + j] - expected) < 1e-5);
            }
        Assert.Equal(budget, graph.RetainedPackedWeightBytes);
        Assert.Equal(xBefore, x); Assert.Equal(wBefore, w);
    }

    [Fact]
    public void ReplacedWeightsAndInvalidation_ReleaseBudgetBeforeRepacking()
    {
        var graph = Graph(16, 2, 2);
        graph.RefreshLifetimeAnalysis();
        var old = graph.Initializers["packed:w0"];
        graph.Initializers["w0"] = new DenseTensor<float>(new float[6], new[] { 2, 3 });
        graph.RefreshLifetimeAnalysis();
        Assert.False(graph.Initializers.ContainsKey("packed:w0"));
        Assert.True(graph.Initializers.ContainsKey("packed:w1"));
        Assert.Equal(16, graph.RetainedPackedWeightBytes);
        graph.InvalidatePreparation();
        Assert.Equal(0, graph.RetainedPackedWeightBytes);
        Assert.DoesNotContain(graph.Initializers.Keys, x => x.StartsWith("packed:", StringComparison.Ordinal));
        graph.Initializers["w0"] = DenseTensor<float>.OfValues(new float[,] { { 5, 6 }, { 7, 8 } });
        graph.RefreshLifetimeAnalysis();
        Assert.NotSame(old, graph.Initializers["packed:w0"]);
        Assert.False(graph.Initializers.ContainsKey("packed:w1"));
        Assert.Equal(16, graph.RetainedPackedWeightBytes);
    }

    [Fact]
    public void SharedBackingArray_UsesOneAccountedClone()
    {
        var graph = Graph(32, 2, 2);
        var array = new float[] { 1, 2, 3, 4 };
        graph.Initializers["w0"] = new DenseTensor<float>(array, new[] { 2, 2 });
        graph.Initializers["w1"] = new DenseTensor<float>(array, new[] { 2, 2 });
        graph.RefreshLifetimeAnalysis();
        Assert.Single(graph.Initializers.Keys.Where(x => x.StartsWith("packed:", StringComparison.Ordinal)));
        Assert.Equal(16, graph.RetainedPackedWeightBytes);
        graph.RefreshLifetimeAnalysis();
        Assert.Single(graph.Initializers.Keys.Where(x => x.StartsWith("packed:", StringComparison.Ordinal)));
        Assert.Equal(16, graph.RetainedPackedWeightBytes);
        var input = Assert.IsType<DenseTensor<float>>(graph.Inputs["x"]);
        Assert.True(graph.Execute(new Dictionary<string, ITensor> { ["x"] = input }, true), graph.LastErrorMessage);
        Assert.Equal(new float[] { 7, 10, 15, 22 }, Assert.IsAssignableFrom<Tensor<float>>(graph.Outputs["y1"]).ToArray());
    }

    [Fact]
    public void DefaultAndInvalidBudgets_HaveExplicitContracts()
    {
        Assert.Equal(long.MaxValue, new ComputationalGraph().MaximumPackedWeightBytes);
        Assert.Throws<ArgumentOutOfRangeException>(() => new ComputationalGraph(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => Model.Load(new OnnxModel(), -1L));
        Assert.Null(OnnxImport.Load("unused-model-path", -1));
        Assert.IsType<ArgumentOutOfRangeException>(OnnxImport.LastErrorCause);
    }

    static OnnxValueInfo Desc(string name, TensorElementType type, params int[] dims) =>
        new() { Name = name, ElementType = type, Dims = dims };

    static OnnxSubgraph Branch() => new()
    {
        Outputs = { Desc("z", TensorElementType.Float, 2, 2) },
        Initializers = { new OnnxTensor { Name = "w", ElementType = TensorElementType.Float,
            Dims = new[] { 2, 2 }, Data = new float[] { 1, 2, 3, 4 } } },
        Nodes = { new OnnxNode { Name = "mm", OpType = "MatMul", Inputs = new[] { "x", "w" }, Outputs = new[] { "z" } } },
    };

    [Theory]
    [InlineData(0)]
    [InlineData(16)]
    public void ModelLoad_PropagatesIndependentBudgetToBothBranches(long budget)
    {
        var model = new OnnxModel
        {
            Opset = { [""] = 14 },
            Inputs = { Desc("cond", TensorElementType.Bool), Desc("x", TensorElementType.Float, 2, 2) },
            Outputs = { Desc("y", TensorElementType.Float, 2, 2) },
            Nodes = { new OnnxNode { Name = "choose", OpType = "If", Inputs = new[] { "cond" }, Outputs = new[] { "y" },
                Attributes = new() { ["then_branch"] = Branch(), ["else_branch"] = Branch() } } },
        };
        var graph = Model.Load(model, budget);
        foreach (string name in new[] { "then_branch", "else_branch" })
        {
            var attributes = Assert.IsType<Dictionary<string, object>>(graph.Nodes[0].Attributes);
            var branch = Assert.IsType<ComputationalGraph>(attributes[name]);
            Assert.Equal(budget, branch.MaximumPackedWeightBytes);
            Assert.Equal(budget, branch.RetainedPackedWeightBytes);
        }
        foreach (bool condition in new[] { true, false })
        {
            Assert.True(graph.Execute(new Dictionary<string, ITensor>
            {
                ["cond"] = new DenseTensor<bool>(new[] { condition }, Array.Empty<int>()),
                ["x"] = DenseTensor<float>.OfValues(new float[,] { { 1, 2 }, { 3, 4 } }),
            }, true), graph.LastErrorMessage);
            Assert.Equal(new float[] { 7, 10, 15, 22 }, Assert.IsAssignableFrom<Tensor<float>>(graph.Outputs["y"]).ToArray());
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(16)]
    public void FileImport_AppliesBudgetBeforePreparation(long budget)
    {
        var model = new ModelProto { IrVersion = 8, Graph = new GraphProto { Name = "packed-budget" } };
        model.OpsetImport.Add(new OperatorSetIdProto { Domain = "", Version = 14 });
        var weight = new TensorProto { Name = "w", DataType = 1 };
        weight.Dims.Add(new long[] { 2, 2 });
        weight.FloatData.Add(new float[] { 1, 2, 3, 4 });
        model.Graph.Initializer.Add(weight);
        var shape = new TensorShapeProto();
        shape.Dim.Add(new[] { new TensorShapeProto.Types.Dimension { DimValue = 2 }, new TensorShapeProto.Types.Dimension { DimValue = 2 } });
        var type = new TypeProto { TensorType = new TypeProto.Types.Tensor { ElemType = 1, Shape = shape } };
        model.Graph.Input.Add(new ValueInfoProto { Name = "x", Type = type });
        model.Graph.Output.Add(new ValueInfoProto { Name = "y", Type = type.Clone() });
        var node = new NodeProto { Name = "mm", OpType = "MatMul" };
        node.Input.Add(new[] { "x", "w" }); node.Output.Add("y"); model.Graph.Node.Add(node);
        string path = Path.Combine(Path.GetTempPath(), "packed-budget-" + Guid.NewGuid().ToString("N") + ".onnx");
        try
        {
            File.WriteAllBytes(path, model.ToByteArray());
            var imported = OnnxImport.Load(path, budget);
            Assert.True(imported is not null, OnnxImport.LastErrorMessage + " " + OnnxImport.LastErrorCause);
            var graph = Assert.IsType<ComputationalGraph>(imported);
            Assert.Equal(budget, graph.MaximumPackedWeightBytes);
            Assert.Equal(budget, graph.RetainedPackedWeightBytes);
        }
        finally { File.Delete(path); }
    }
}
