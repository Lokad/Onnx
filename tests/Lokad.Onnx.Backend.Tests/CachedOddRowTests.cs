using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx.Backend.Tests;

public class CachedOddRowTests
{
    static ComputationalGraph Graph(int rows, int reduction, int columns)
    {
        var random = new Random(20261028);
        float[] Values(int count) => Enumerable.Range(0, count).Select(_ => random.NextSingle() - .5f).ToArray();
        var graph = new ComputationalGraph(32L * 1024 * 1024);
        graph.Metadata["Name"] = "cached-odd-rows";
        graph.Inputs["x"] = new DenseTensor<float>(Values(rows * reduction), new[] { rows, reduction });
        graph.Initializers["w"] = new DenseTensor<float>(Values(reduction * columns), new[] { reduction, columns });
        graph.Outputs["y"] = Tensor<float>.Zeros(rows, columns);
        graph.Nodes.Add(new Node { Name = "mm", Op = OpType.MatMul, OpTypeName = "MatMul", Domain = "",
            Inputs = new[] { "x", "w" }, Outputs = new[] { "y" } });
        graph.Prepare();
        Assert.Single(graph.PackedWeights);
        return graph;
    }

    static float[] Run(ComputationalGraph graph)
    {
        Assert.True(graph.Execute(new Dictionary<string, ITensor> { ["x"] = graph.Inputs["x"]! }, true,
            ExecutionProvider.CPU, ExecutionOptions.Memory), graph.LastErrorMessage);
        return ((Tensor<float>)graph.Outputs["y"]!).ToArray();
    }

    static void Oracle(float[] input, float[] weight, float[] output, int m, int n, int k)
    {
        int fmaColumns = k / 32 * 32 + k % 32 / 8 * 8;
        foreach (int row in new[] { 0, Math.Min(11, m - 1), m - 1 }.Distinct())
        for (int col = 0; col < k; col++)
        {
            float expected = 0;
            for (int d = 0; d < n; d++)
                expected = col < fmaColumns
                    ? MathF.FusedMultiplyAdd(weight[d * k + col], input[row * n + d], expected)
                    : expected + input[row * n + d] * weight[d * k + col];
            Assert.Equal(BitConverter.SingleToInt32Bits(expected), BitConverter.SingleToInt32Bits(output[row * k + col]));
        }
    }

    [SkippableTheory]
    [InlineData(13, 65, 33)]
    [InlineData(25, 65, 95)]
    [InlineData(97, 128, 128)]
    [InlineData(751, 1024, 1024)]
    public void OddGroupsConsumeTheCachedWeightWithoutRepacking(int m, int n, int k)
    {
        Skip.If(!Avx512F.IsSupported || !Fma.IsSupported || !AblationSwitches.EnablePackedAvx512Rows,
            "The arbitrary-row packed consumer requires enabled AVX-512.");
        var graph = Graph(m, n, k);
        var input = ((Tensor<float>)graph.Inputs["x"]!).ToArray();
        var weight = ((Tensor<float>)graph.Initializers["w"]).ToArray();
        var packed = Assert.Single(graph.PackedWeights).Value.Packed;
        var packedBefore = packed.ToArray();
        var expected = Run(graph);
        var held = (Tensor<float>)graph.Outputs["y"]!;
        Oracle(input, weight, expected, m, n, k);
        Assert.Equal(0, graph.LastScratchBytes);
        Assert.Equal(0, graph.LastCopyBytes);
        Assert.Equal(expected, Run(graph));
        Assert.Equal(expected, held.ToArray());
        Assert.Equal(input, ((Tensor<float>)graph.Inputs["x"]!).ToArray());
        Assert.Equal(weight, ((Tensor<float>)graph.Initializers["w"]).ToArray());
        Assert.Equal(packedBefore, packed.ToArray());
        // Replacing the source must invalidate the mapping rather than consuming
        // the previous clone, even though the new shape is identical.
        graph.Initializers["w"] = new DenseTensor<float>(new float[n * k], new[] { n, k });
        Assert.All(Run(graph), value => Assert.Equal(0f, value));
        Assert.Equal(expected, held.ToArray());
    }

    [SkippableFact]
    public void LegacyOddFallbackRemainsAvailableWhenWideRowsAreDisabled()
    {
        Skip.If(!Fma.IsSupported, "The intrinsic fallback requires FMA.");
        Skip.If(Avx512F.IsSupported && AblationSwitches.EnablePackedAvx512Rows, "Run with wide rows disabled.");
        var graph = Graph(97, 1024, 1024);
        var input = ((Tensor<float>)graph.Inputs["x"]!).ToArray();
        var weight = ((Tensor<float>)graph.Initializers["w"]).ToArray();
        Oracle(input, weight, Run(graph), 97, 1024, 1024);
        Assert.True(graph.LastScratchBytes > 0);
    }
}
