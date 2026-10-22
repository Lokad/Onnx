using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx.Backend.Tests;

public class PackedOperandLayoutTests
{
    [SkippableTheory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void DifferentLayoutsOverOneArray_KeepTheirOwnMatrix(bool batched, bool replaceAfterPreparation)
    {
        Skip.If(!Fma.IsSupported, "Packed dispatch requires FMA.");
        float[] values = [1, 2, 3, 4];
        var ordinary = new DenseTensor<float>(values.AsMemory(), new[] { 2, 2 });
        var reversed = new DenseTensor<float>(values.AsMemory(), new[] { 2, 2 }, reverseStride: true);
        var input = new DenseTensor<float>(batched ? new float[] { 1, 2, 3, 4, 1, 2, 3, 4 } : new float[] { 1, 2, 3, 4 },
            batched ? new[] { 2, 2, 2 } : new[] { 2, 2 });
        var graph = new ComputationalGraph();
        graph.Metadata["Name"] = "packed-layout";
        graph.Inputs["x"] = input;
        graph.Initializers["w"] = ordinary;
        graph.Outputs["y"] = Tensor<float>.Zeros(input.Dimensions.ToArray());
        graph.Nodes.Add(new Node { Name = "mm", Op = OpType.MatMul, Inputs = ["x", "w"], Outputs = ["y"] });
        if (!replaceAfterPreparation)
        {
            graph.Initializers["r"] = reversed;
            graph.Outputs["z"] = Tensor<float>.Zeros(input.Dimensions.ToArray());
            graph.Nodes.Add(new Node { Name = "reversed-mm", Op = OpType.MatMul, Inputs = ["x", "r"], Outputs = ["z"] });
        }
        graph.Prepare();
        Assert.Single(graph.PackedWeights);
        if (replaceAfterPreparation) graph.Initializers["w"] = reversed;
        string target = replaceAfterPreparation ? "y" : "z";
        float[] expected = batched ? [5, 11, 11, 25, 5, 11, 11, 25] : [5, 11, 11, 25];
        var feeds = new Dictionary<string, ITensor> { ["x"] = input };
        foreach (var options in new[] { ExecutionOptions.Default, ExecutionOptions.Scalar })
        {
            Assert.True(graph.Execute(feeds, true, ExecutionProvider.CPU, options), graph.LastErrorMessage);
            var held = Assert.IsAssignableFrom<Tensor<float>>(graph.Outputs[target]);
            Assert.Equal(expected, held.ToArray());
            Assert.False(TensorAlias.SharesBackingMemory(held, reversed));
            Assert.False(TensorAlias.SharesBackingMemory(held, input));
            Assert.True(graph.Execute(feeds, true, ExecutionProvider.CPU, options), graph.LastErrorMessage);
            Assert.Equal(expected, held.ToArray());
            Assert.False(TensorAlias.SharesBackingMemory(held, (Tensor<float>)graph.Outputs[target]!));
        }
        Assert.Equal(new float[] { 1, 2, 3, 4 }, values);
    }

    [Theory]
    [InlineData(0, true)]
    [InlineData(1, true)]
    [InlineData(2, true)]
    [InlineData(3, true)]
    [InlineData(4, false)]
    [InlineData(5, false)]
    [InlineData(6, false)]
    [InlineData(7, false)]
    [InlineData(8, false)]
    public void Resolver_RequiresEquivalentMatrixStorage(int variant, bool eligible)
    {
        float[] values = [1, 2, 3, 4, 5, 6, 7, 8];
        var source = new DenseTensor<float>(values.AsMemory(0, 4), new[] { 2, 2 });
        var graph = new ComputationalGraph();
        graph.Inputs["x"] = Tensor<float>.Zeros(2, 2);
        graph.Initializers["w"] = source;
        graph.Outputs["y"] = Tensor<float>.Zeros(2, 2);
        graph.Nodes.Add(new Node { Name = "mm", Op = OpType.MatMul, Inputs = ["x", "w"], Outputs = ["y"] });
        graph.Prepare();
        var packed = Assert.Single(graph.PackedWeights).Value.Packed;
        var reversed = new DenseTensor<float>(values.AsMemory(0, 4), new[] { 2, 2 }, reverseStride: true);
        Tensor<float> operand = variant switch
        {
            0 => source,
            1 => new DenseTensor<float>(values.AsMemory(0, 4), new[] { 2, 2 }),
            2 => source.Reshape(1, 2, 2),
            3 => Tensor<float>.BroadcastTo(source, [3, 2, 2]),
            4 => reversed,
            5 => Tensor<float>.BroadcastTo(reversed, [3, 2, 2]),
            6 => new DenseTensor<float>(values.AsMemory(2, 4), new[] { 2, 2 }),
            7 => source.Reshape(1, 4),
            8 => new DenseTensor<float>(values.AsMemory(), new[] { 2, 2, 2 }),
            _ => throw new ArgumentOutOfRangeException(nameof(variant))
        };
        if (eligible) Assert.Same(packed, GraphPacking.ResolvePacked(graph.PackedWeights, operand));
        else Assert.Null(GraphPacking.ResolvePacked(graph.PackedWeights, operand));
        Assert.Same(packed, GraphPacking.ResolvePacked(graph.PackedWeights, packed));
        Assert.Equal(new float[] { 1, 2, 3, 4, 5, 6, 7, 8 }, values);
    }
}
