using Lokad.Onnx.Optimization;

namespace Lokad.Onnx.Backend.Tests;

public class ConstantDeduplicationAllocationTests
{
    [Theory]
    [InlineData(8)]
    [InlineData(32)]
    public void DistinctEqualShapePayloadsAllocateProportionally(int count)
    {
        const int length = 65536;
        var graph = new ComputationalGraph();
        for (int i = 0; i < count; i++)
        {
            var values = new float[length];
            values[^1] = i;
            AddConstant(graph, "c" + i, new DenseTensor<float>(values, [length]));
        }
        var facts = GraphFacts.Build(graph);
        ConstFold.FoldConstants(graph, facts);
        long start = GC.GetAllocatedBytesForCurrentThread();
        ConstFold.FoldConstants(graph, facts);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - start;
        long payloadBytes = (long)count * length * sizeof(float);
        Assert.True(allocated < 3 * payloadBytes, $"{count} constants: {allocated} bytes allocated for {payloadBytes} payload bytes.");
        Assert.Equal(count, graph.Nodes.Count(n => n.Op == OpType.Constant));
    }

    [Fact]
    public void FullSnapshotBudgetStillMergesDuplicatesOfRetainedSurvivors()
    {
        var graph = new ComputationalGraph();
        const int length = 262144;
        for (int i = 0; i < 10; i++)
        {
            var data = new float[length];
            data[^1] = i;
            AddConstant(graph, "c" + i, new DenseTensor<float>(data, [length]));
        }
        AddConstant(graph, "duplicate", new DenseTensor<float>(new float[length], [length]));
        ConstFold.FoldConstants(graph, GraphFacts.Build(graph));
        Assert.Equal(10, graph.Nodes.Count(n => n.Op == OpType.Constant));
        Assert.Equal("c0", graph.Nodes.Single(n => n.Name == "duplicateOut").Inputs[0]);
    }

    [Fact]
    public void EqualBytesWithDifferentTypesOrShapesDoNotMerge()
    {
        var graph = new ComputationalGraph();
        AddConstant(graph, "float", new DenseTensor<float>(new float[4], [2, 2]));
        AddConstant(graph, "otherShape", new DenseTensor<float>(new float[4], [4]));
        AddConstant(graph, "otherType", new DenseTensor<int>(new int[4], [2, 2]));
        ConstFold.FoldConstants(graph, GraphFacts.Build(graph));
        Assert.Equal(3, graph.Nodes.Count(n => n.Op == OpType.Constant));
    }

    static void AddConstant(ComputationalGraph graph, string name, ITensor value)
    {
        graph.Nodes.Add(new Node { Name = name, Op = OpType.Constant, OpTypeName = "Constant", Domain = "", OpsetVersion = 17,
            Inputs = [], Outputs = [name], Attributes = new() { ["value"] = value } });
        graph.Nodes.Add(new Node { Name = name + "Out", Op = OpType.Identity, OpTypeName = "Identity", Domain = "", OpsetVersion = 17,
            Inputs = [name], Outputs = [name + "Out"], Attributes = new() });
        graph.Outputs[name + "Out"] = value;
    }
}
