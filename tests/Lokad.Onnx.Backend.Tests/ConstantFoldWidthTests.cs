using System.Numerics;
using Lokad.Onnx.Optimization;

namespace Lokad.Onnx.Backend.Tests;

[Collection("NonParallelAllocation")]
public class ConstantFoldWidthTests
{
    [Theory]
    [InlineData(TensorElementType.Complex128, 65_536, true)]
    [InlineData(TensorElementType.Complex128, 65_537, false)]
    [InlineData(TensorElementType.Complex128, 131_072, false)]
    [InlineData(TensorElementType.Complex128, 131_073, false)]
    [InlineData(TensorElementType.Float16, 131_072, true)]
    [InlineData(TensorElementType.Float16, 131_073, true)]
    [InlineData(TensorElementType.Float16, 524_288, true)]
    [InlineData(TensorElementType.Float16, 524_289, false)]
    [InlineData(TensorElementType.BFloat16, 131_072, true)]
    [InlineData(TensorElementType.BFloat16, 131_073, true)]
    [InlineData(TensorElementType.BFloat16, 524_288, true)]
    [InlineData(TensorElementType.BFloat16, 524_289, false)]
    [InlineData(TensorElementType.Float, 262_144, true)]
    [InlineData(TensorElementType.Float, 262_145, false)]
    public void GatherIntermediateHonorsActualByteLimit(TensorElementType type, int length, bool folds)
    {
        var optimized = Model.Load(GatherModel(type, length));
        var ordinary = Model.Load(GatherModel(type, length), runOptimizer: false);
        Assert.Equal(folds ? OpType.Constant : OpType.Gather,
            optimized.Nodes.Single(n => n.Outputs.Contains("mid")).Op);
        AssertSameOutputs(ordinary, optimized);
    }

    [Theory]
    [InlineData(TensorElementType.Complex128, 131_072, false)]
    [InlineData(TensorElementType.Float16, 524_288, true)]
    [InlineData(TensorElementType.BFloat16, 524_288, true)]
    public void TransposeIntermediateHonorsActualByteLimit(TensorElementType type, int length, bool folds)
    {
        OnnxModel Make()
        {
            var model = new OnnxModel { Opset = { [""] = 14 } };
            model.Outputs.Add(new OnnxValueInfo { Name = "y", ElementType = type, Dims = [length, 1] });
            AddConstant(model, "data", TensorBase.CreateDenseTensor(type, Values(type, length), [1, length]));
            model.Nodes.Add(new OnnxNode { OpType = "Transpose", Inputs = ["data"], Outputs = ["mid"],
                Attributes = new() { ["perm"] = new long[] { 1, 0 } } });
            model.Nodes.Add(new OnnxNode { OpType = "Identity", Inputs = ["mid"], Outputs = ["y"] });
            return model;
        }
        var optimized = Model.Load(Make());
        var ordinary = Model.Load(Make(), runOptimizer: false);
        Assert.Equal(folds ? OpType.Constant : OpType.Transpose,
            optimized.Nodes.Single(n => n.Outputs.Contains("mid")).Op);
        AssertSameOutputs(ordinary, optimized);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(300_000)]
    public void GraphOutputRemainsUnfoldedRegardlessOfSize(int length)
    {
        var model = GatherModel(TensorElementType.Float, length);
        model.Nodes.RemoveAt(model.Nodes.Count - 1);
        model.Outputs[0].Name = "mid";
        var graph = Model.Load(model);
        Assert.Equal(OpType.Gather, graph.Nodes.Single(n => n.Outputs.Contains("mid")).Op);
        Assert.True(graph.Execute(new Dictionary<string, ITensor>(), true), graph.LastErrorMessage);
        Assert.All(((Tensor<float>)graph.Outputs["mid"]).ToArray(), value => Assert.Equal(1f, value));
    }

    [Theory]
    [InlineData(TensorElementType.Float)]
    [InlineData(TensorElementType.Complex128)]
    public async Task OversizedGatherIsRefusedBeforeAllocatingItsResult(TensorElementType type)
    {
        await Task.Factory.StartNew(() =>
        {
            for (int warm = 0; warm < 256; warm++)
            {
                var small = Model.Load(GatherModel(type, 2), runOptimizer: false);
                ConstFold.FoldConstants(small, GraphFacts.Build(small));
            }
            int length = type == TensorElementType.Complex128 ? 131_072 : 300_000;
            var graph = Model.Load(GatherModel(type, length), runOptimizer: false);
            // Exclude the existing index payload from deduplication's snapshot work.
            graph.Outputs["indices"] = (ITensor)graph.Nodes.Single(n => n.Outputs.Contains("indices")).Attributes["value"];
            var facts = GraphFacts.Build(graph);
            long before = GC.GetAllocatedBytesForCurrentThread();
            ConstFold.FoldConstants(graph, facts);
            long bytes = GC.GetAllocatedBytesForCurrentThread() - before;
            Assert.InRange(bytes, 0, 65_536);
            Assert.Equal(OpType.Gather, graph.Nodes.Single(n => n.Outputs.Contains("mid")).Op);
        }, CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default);
    }

    [Theory]
    [InlineData(TensorElementType.Complex128, 65_536, 1)]
    [InlineData(TensorElementType.Complex128, 65_537, 2)]
    [InlineData(TensorElementType.Float16, 524_288, 1)]
    [InlineData(TensorElementType.Float16, 524_289, 2)]
    [InlineData(TensorElementType.BFloat16, 524_288, 1)]
    [InlineData(TensorElementType.BFloat16, 524_289, 2)]
    public void DeduplicationHonorsActualByteLimit(TensorElementType type, int length, int constants)
    {
        var graph = Model.Load(DuplicateModel(type, length));
        Assert.Equal(constants, graph.Nodes.Count(n => n.Op == OpType.Constant));
        Assert.True(graph.Execute(new Dictionary<string, ITensor>(), true), graph.LastErrorMessage);
        Assert.Equal(graph.Outputs["aOut"].ToArray().Cast<object>(), graph.Outputs["bOut"].ToArray().Cast<object>());
    }

    [Fact]
    public async Task OversizedComplexConstantIsRefusedBeforeSnapshotAllocation()
    {
        await Task.Factory.StartNew(() =>
        {
            for (int warm = 0; warm < 256; warm++)
            {
                var small = Model.Load(DuplicateModel(TensorElementType.Complex128, 2), runOptimizer: false);
                ConstFold.FoldConstants(small, GraphFacts.Build(small));
            }
            var graph = Model.Load(DuplicateModel(TensorElementType.Complex128, 65_537), runOptimizer: false);
            var facts = GraphFacts.Build(graph);
            long before = GC.GetAllocatedBytesForCurrentThread();
            ConstFold.FoldConstants(graph, facts);
            long bytes = GC.GetAllocatedBytesForCurrentThread() - before;
            Assert.InRange(bytes, 0, 65_536);
            Assert.Equal(2, graph.Nodes.Count(n => n.Op == OpType.Constant));
        }, CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default);
    }

    internal static OnnxModel GatherModel(TensorElementType type, int length)
    {
        var model = new OnnxModel { Opset = { [""] = 14 } };
        model.Outputs.Add(new OnnxValueInfo { Name = "y", ElementType = type, Dims = [length] });
        AddConstant(model, "data", TensorBase.CreateDenseTensor(type, Values(type, 1), [1]));
        AddConstant(model, "indices", new DenseTensor<int>(new int[length], [length]));
        model.Nodes.Add(new OnnxNode { OpType = "Gather", Inputs = ["data", "indices"], Outputs = ["mid"] });
        model.Nodes.Add(new OnnxNode { OpType = "Identity", Inputs = ["mid"], Outputs = ["y"] });
        return model;
    }

    static OnnxModel DuplicateModel(TensorElementType type, int length)
    {
        var model = new OnnxModel { Opset = { [""] = 14 } };
        foreach (string name in new[] { "a", "b" })
        {
            AddConstant(model, name, TensorBase.CreateDenseTensor(type, Values(type, length), [length]));
            model.Outputs.Add(new OnnxValueInfo { Name = name + "Out", ElementType = type, Dims = [length] });
            model.Nodes.Add(new OnnxNode { OpType = "Identity", Inputs = [name], Outputs = [name + "Out"] });
        }
        return model;
    }

    static Array Values(TensorElementType type, int length) => type switch
    {
        TensorElementType.Complex128 => Enumerable.Repeat(new Complex(1, 2), length).ToArray(),
        TensorElementType.Float16 => Enumerable.Repeat((Half)1, length).ToArray(),
        TensorElementType.BFloat16 => Enumerable.Repeat((BFloat16)1f, length).ToArray(),
        TensorElementType.Float => Enumerable.Repeat(1f, length).ToArray(),
        _ => throw new ArgumentOutOfRangeException(nameof(type))
    };

    static void AddConstant(OnnxModel model, string name, ITensor value) => model.Nodes.Add(new OnnxNode
    {
        OpType = "Constant", Inputs = [], Outputs = [name], Attributes = new() { ["value"] = value }
    });

    static void AssertSameOutputs(ComputationalGraph ordinary, ComputationalGraph optimized)
    {
        var feed = new Dictionary<string, ITensor>();
        Assert.True(ordinary.Execute(feed, true), ordinary.LastErrorMessage);
        Assert.True(optimized.Execute(feed, true), optimized.LastErrorMessage);
        Assert.Equal(ordinary.Outputs["y"].ToArray().Cast<object>(), optimized.Outputs["y"].ToArray().Cast<object>());
    }
}
