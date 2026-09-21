namespace Lokad.Onnx.Backend.Tests;

public class PackedBoundaryTests
{
    const long OneWeightBytes = 4096L * 32 * sizeof(float);

    static ComputationalGraph Graph(int rows, int reduction, int columns, long budget) =>
        Graph(rows, reduction, columns, budget, 1);

    static ComputationalGraph Graph(int rows, int reduction, int columns, long budget, int weights)
    {
        var random = new Random(456);
        float[] Values(int count) => Enumerable.Range(0, count).Select(_ => random.NextSingle() - .5f).ToArray();
        var graph = new ComputationalGraph(budget);
        graph.Metadata["Name"] = "inclusive-packed-boundary";
        graph.Inputs["x"] = new DenseTensor<float>(Values(rows * reduction), new[] { rows, reduction });
        for (int index = 0; index < weights; index++)
        {
            graph.Initializers["w" + index] = new DenseTensor<float>(Values(reduction * columns), new[] { reduction, columns });
            graph.Outputs["y" + index] = Tensor<float>.Zeros(rows, columns);
            graph.Nodes.Add(new Node { Name = "mm" + index, Op = OpType.MatMul, OpTypeName = "MatMul",
                Inputs = new[] { "x", "w" + index }, Outputs = new[] { "y" + index } });
        }
        return graph;
    }

    static float[] Execute(ComputationalGraph graph)
    {
        var input = Assert.IsAssignableFrom<Tensor<float>>(graph.Inputs["x"]);
        float[] before = input.ToArray();
        var weights = graph.Initializers.Where(p => !p.Key.StartsWith("packed:", StringComparison.Ordinal))
            .ToDictionary(p => p.Key, p => Assert.IsAssignableFrom<Tensor<float>>(p.Value).ToArray());
        Assert.True(graph.Execute(new Dictionary<string, ITensor> { ["x"] = input }, true), graph.LastErrorMessage);
        Assert.Equal(before, input.ToArray());
        foreach (var pair in weights) Assert.Equal(pair.Value, ((Tensor<float>)graph.Initializers[pair.Key]).ToArray());
        return ((Tensor<float>)graph.Outputs["y0"]).ToArray();
    }

    [Fact]
    public void Reduction4096_IsAdmitted()
    {
        var graph = Graph(2, 4096, 32, OneWeightBytes);
        graph.RefreshLifetimeAnalysis();
        Assert.Equal(OneWeightBytes, graph.RetainedPackedWeightBytes);
        Assert.True(graph.Initializers.ContainsKey("packed:w0"));
        Assert.Equal(Execute(Graph(2, 4096, 32, 0)), Execute(graph));
    }

    [Theory]
    [InlineData(4095, true)]
    [InlineData(4097, false)]
    public void AdjacentReductionBounds_PreserveAdmission(int reduction, bool packed)
    {
        var graph = Graph(2, reduction, 32, long.MaxValue);
        var actual = Execute(graph);
        Assert.Equal(packed ? (long)reduction * 32 * sizeof(float) : 0, graph.RetainedPackedWeightBytes);
        Assert.Equal(packed, graph.Initializers.ContainsKey("packed:w0"));
        Assert.Equal(Execute(Graph(2, reduction, 32, 0)), actual);
    }

    [Theory]
    [InlineData(0, 0)]
    [InlineData(OneWeightBytes - 1, 0)]
    [InlineData(OneWeightBytes, 1)]
    [InlineData(OneWeightBytes * 2 - 1, 1)]
    [InlineData(OneWeightBytes * 2, 2)]
    public void InclusiveBoundary_RespectsAggregateBudget(long budget, int count)
    {
        var graph = Graph(2, 4096, 32, budget, 2);
        Execute(graph);
        Assert.Equal(count * OneWeightBytes, graph.RetainedPackedWeightBytes);
        Assert.Equal(count, graph.Initializers.Keys.Count(k => k.StartsWith("packed:", StringComparison.Ordinal)));
        graph.RefreshLifetimeAnalysis();
        Assert.Equal(count * OneWeightBytes, graph.RetainedPackedWeightBytes);
        graph.InvalidatePreparation();
        Assert.Equal(0, graph.RetainedPackedWeightBytes);
        Execute(graph);
        Assert.Equal(count * OneWeightBytes, graph.RetainedPackedWeightBytes);
    }

    [Theory]
    [InlineData(2, 32)]
    [InlineData(3, 33)]
    [InlineData(5, 32)]
    [InlineData(64, 33)]
    public void InclusiveBoundary_RowAndColumnTailsPreserveProduct(int rows, int columns)
    {
        var graph = Graph(rows, 4096, columns, 4096L * columns * sizeof(float));
        var x = ((Tensor<float>)graph.Inputs["x"]).ToArray();
        var w = ((Tensor<float>)graph.Initializers["w0"]).ToArray();
        var actual = Execute(graph);
        Assert.Equal(4096L * columns * sizeof(float), graph.RetainedPackedWeightBytes);
        var fallback = Execute(Graph(rows, 4096, columns, 0));
        Assert.Equal(fallback.Select(BitConverter.SingleToInt32Bits), actual.Select(BitConverter.SingleToInt32Bits));
        for (int row = 0; row < rows; row++) for (int col = 0; col < columns; col++)
        {
            double expected = 0;
            for (int k = 0; k < 4096; k++) expected += (double)x[row * 4096 + k] * w[k * columns + col];
            Assert.True(Math.Abs(actual[row * columns + col] - expected) <= 2e-4 * Math.Max(1, Math.Abs(expected)),
                $"Product mismatch at {row},{col}");
        }
    }
}
