namespace Lokad.Onnx.Backend.Tests;

public class GraphAccountingTests
{
    static OnnxValueInfo Desc(string name, TensorElementType type, params int[] dims) =>
        new() { Name = name, ElementType = type, Dims = dims };

    static (OnnxModel Model, Dictionary<string, ITensor> Feed, float[] Expected) Workload(bool scratch, string prefix = "work")
    {
        int[] xShape = scratch ? new[] { 1, 1, 4, 4 } : new[] { 2, 2 };
        int[] wShape = scratch ? new[] { 1, 1, 2, 2 } : new[] { 2, 2 };
        int[] yShape = scratch ? new[] { 1, 1, 3, 3 } : new[] { 2, 2 };
        Tensor<float> x = scratch
            ? new DenseTensor<float>(Enumerable.Range(1, 16).Select(i => (float)i).ToArray(), xShape)
            : DenseTensor<float>.OfValues(new float[,] { { 1, 2, 3 }, { 4, 5, 6 } })
                .Slice(new SliceIndex(0, 2), new SliceIndex(1, 3));
        var w = scratch ? DenseTensor<float>.Ones(wShape)
            : DenseTensor<float>.OfValues(new float[,] { { 1, 0 }, { 0, 1 } });
        var model = new OnnxModel
        {
            Name = prefix, Opset = { [""] = 14 },
            Inputs = { Desc(prefix + "x", TensorElementType.Float, xShape), Desc(prefix + "w", TensorElementType.Float, wShape) },
            Outputs = { Desc(prefix + "y", TensorElementType.Float, yShape) },
            Nodes = { new OnnxNode { Name = prefix, OpType = scratch ? "Conv" : "MatMul",
                Inputs = new[] { prefix + "x", prefix + "w" }, Outputs = new[] { prefix + "y" } } }
        };
        return (model, new() { [prefix + "x"] = x, [prefix + "w"] = w },
            scratch ? new float[] { 14, 18, 22, 30, 34, 38, 46, 50, 54 } : new float[] { 2, 3, 5, 6 });
    }

    static ExecutionOptions Options(IScratchAccountant scratch, ICopyAccountant copy) =>
        ExecutionOptions.Scalar with { Tensor = TensorExecutionOptions.Scalar with { ScratchReporter = scratch, CopyReporter = copy } };

    static void LocalBytes(ComputationalGraph execution, bool scratch)
    {
        Assert.Equal(scratch ? 144 : 0, execution.LastScratchBytes);
        Assert.Equal(scratch ? 0 : 16, execution.LastCopyBytes);
    }

    static void CallerBytes(IScratchAccountant scratch, ICopyAccountant copy, long scratchBytes, long copyBytes)
    {
        Assert.Equal(scratchBytes, scratch.TotalScratchBytes);
        Assert.Equal(copyBytes, copy.TotalCopyBytes);
    }

    [Theory]
    [InlineData(false, false)] [InlineData(false, true)]
    [InlineData(true, false)] [InlineData(true, true)]
    public void SuppliedReportersAccumulateWhileMetricsRemainPerRun(bool explicitContext, bool scratchWork)
    {
        var fixture = Workload(scratchWork);
        var graph = Model.Load(fixture.Model);
        var scratch = new ScratchAccountant();
        var copy = new CopyAccountant();
        scratch.AddScratchBytes(11); copy.AddCopyBytes(23);
        var options = Options(scratch, copy);
        ComputationalGraph execution = explicitContext ? graph.CreateExecution(options) : graph;
        for (int run = 1; run <= 2; run++)
        {
            bool success = explicitContext ? execution.Execute(fixture.Feed, false)
                : execution.Execute(fixture.Feed, false, ExecutionProvider.CPU, options);
            Assert.True(success, execution.LastErrorMessage);
            Assert.Equal(fixture.Expected, ((Tensor<float>)execution.Outputs["worky"]).ToArray());
            LocalBytes(execution, scratchWork);
            CallerBytes(scratch, copy, 11 + (scratchWork ? 144 * run : 0), 23 + (scratchWork ? 0 : 16 * run));
        }
    }

    static OnnxNode If(string name, OnnxSubgraph then, OnnxSubgraph otherwise, string output) => new()
    {
        Name = name, OpType = "If", Inputs = new[] { "cond" }, Outputs = new[] { output },
        Attributes = new() { ["then_branch"] = then, ["else_branch"] = otherwise }
    };

    [Theory]
    [InlineData(false, false)] [InlineData(false, true)]
    [InlineData(true, false)] [InlineData(true, true)]
    public void NestedReportersIncludeOnlySelectedWorkWithoutInflatingParentMetrics(bool scratchLeaf, bool condition)
    {
        var root = Workload(!scratchLeaf, "root");
        var leaf = Workload(scratchLeaf, "leaf");
        int[] shape = leaf.Model.Outputs[0].Dims;
        var fallbackValues = Enumerable.Repeat(-7f, leaf.Expected.Length).ToArray();
        root.Model.Inputs.AddRange(leaf.Model.Inputs);
        root.Model.Inputs.Add(Desc("cond", TensorElementType.Bool));
        root.Model.Inputs.Add(Desc("fallback", TensorElementType.Float, shape));
        foreach (var entry in leaf.Feed) root.Feed.Add(entry.Key, entry.Value);
        root.Feed["cond"] = DenseTensor<bool>.Scalar(condition);
        root.Feed["fallback"] = new DenseTensor<float>(fallbackValues, shape);
        var selected = new OnnxSubgraph { Outputs = leaf.Model.Outputs, Nodes = leaf.Model.Nodes };
        var fallback = new OnnxSubgraph { Outputs = { Desc("fallback", TensorElementType.Float, shape) } };
        var middle = new OnnxSubgraph
        {
            Outputs = { Desc("middle", TensorElementType.Float, shape) },
            Nodes = { If("inner", selected, fallback, "middle") }
        };
        root.Model.Nodes.Add(If("outer", middle, fallback, "selected"));
        root.Model.Outputs.Add(Desc("selected", TensorElementType.Float, shape));
        var scratch = new ScratchAccountant(); var copy = new CopyAccountant();
        var execution = Model.Load(root.Model).CreateExecution(Options(scratch, copy));
        for (int run = 1; run <= 2; run++)
        {
            Assert.True(execution.Execute(root.Feed, false), execution.LastErrorMessage);
            Assert.Equal(root.Expected, ((Tensor<float>)execution.Outputs["rooty"]).ToArray());
            Assert.Equal(condition ? leaf.Expected : fallbackValues, ((Tensor<float>)execution.Outputs["selected"]).ToArray());
            LocalBytes(execution, !scratchLeaf);
            var child = Assert.Single(execution.LastSubgraphExecutions);
            Assert.Equal(0, child.ScratchBytes); Assert.Equal(0, child.CopyBytes);
            if (condition)
            {
                var grandchild = Assert.Single(child.Children);
                Assert.Equal(scratchLeaf ? 144 : 0, grandchild.ScratchBytes);
                Assert.Equal(scratchLeaf ? 0 : 16, grandchild.CopyBytes);
            }
            else Assert.Empty(child.Children);
            CallerBytes(scratch, copy, run * (!scratchLeaf || condition ? 144 : 0),
                run * (scratchLeaf || condition ? 16 : 0));
        }
    }

    [Theory]
    [InlineData(false, false)] [InlineData(false, true)]
    [InlineData(true, false)] [InlineData(true, true)]
    public async Task ConcurrentContextsKeepLocalMetricsAndHonorSharedOrDistinctReporters(bool scratchWork, bool shared)
    {
        var fixture = Workload(scratchWork);
        var graph = Model.Load(fixture.Model);
        var scratches = new[] { new ScratchAccountant(), new ScratchAccountant() };
        var copies = new[] { new CopyAccountant(), new CopyAccountant() };
        if (shared) { scratches[1] = scratches[0]; copies[1] = copies[0]; }
        var contexts = Enumerable.Range(0, 2).Select(i => graph.CreateExecution(Options(scratches[i], copies[i]))).ToArray();
        await Task.WhenAll(contexts.Select(context => Task.Run(() =>
        {
            for (int run = 0; run < 3; run++)
            {
                Assert.True(context.Execute(fixture.Feed, false), context.LastErrorMessage);
                LocalBytes(context, scratchWork);
                Assert.Equal(fixture.Expected, ((Tensor<float>)context.Outputs["worky"]).ToArray());
            }
        })));
        for (int i = 0; i < 2; i++)
            CallerBytes(scratches[i], copies[i], scratchWork ? 144 * (shared ? 6 : 3) : 0,
                scratchWork ? 0 : 16 * (shared ? 6 : 3));
    }

    [Theory]
    [InlineData(false)] [InlineData(true)]
    public void ModelFailureRetainsReportedWorkAndAllowsRetry(bool scratchWork)
    {
        var fixture = Workload(scratchWork);
        fixture.Model.Inputs.Add(Desc("shape", TensorElementType.Int64, 1));
        fixture.Model.Outputs.Clear();
        fixture.Model.Outputs.Add(Desc("final", TensorElementType.Float, fixture.Expected.Length));
        fixture.Model.Nodes.Add(new OnnxNode { Name = "reshape", OpType = "Reshape",
            Inputs = new[] { "worky", "shape" }, Outputs = new[] { "final" } });
        var scratch = new ScratchAccountant(); var copy = new CopyAccountant();
        var execution = Model.Load(fixture.Model).CreateExecution(Options(scratch, copy));
        fixture.Feed["shape"] = DenseTensor<long>.OfValues(new[] { (long)fixture.Expected.Length - 1 });
        Assert.False(execution.Execute(fixture.Feed, false));
        Assert.NotNull(execution.LastErrorMessage);
        LocalBytes(execution, scratchWork);
        CallerBytes(scratch, copy, scratchWork ? 144 : 0, scratchWork ? 0 : 16);
        fixture.Feed["shape"] = DenseTensor<long>.OfValues(new[] { (long)fixture.Expected.Length });
        Assert.True(execution.Execute(fixture.Feed, false), execution.LastErrorMessage);
        Assert.Equal(fixture.Expected, ((Tensor<float>)execution.Outputs["final"]).ToArray());
        LocalBytes(execution, scratchWork);
        CallerBytes(scratch, copy, scratchWork ? 288 : 0, scratchWork ? 0 : 32);
    }

    sealed class ThrowOnceReporter(bool fatal = false) : IScratchAccountant, ICopyAccountant
    {
        readonly ScratchAccountant scratch = new();
        readonly CopyAccountant copy = new();
        int first = 1;
        public long TotalScratchBytes => scratch.TotalScratchBytes;
        public long TotalCopyBytes => copy.TotalCopyBytes;
        public void AddScratchBytes(long bytes) { scratch.AddScratchBytes(bytes); FailOnce(); }
        public void AddCopyBytes(long bytes) { copy.AddCopyBytes(bytes); FailOnce(); }
        void FailOnce()
        {
            if (Interlocked.Exchange(ref first, 0) != 0)
            {
                if (fatal) throw new OutOfMemoryException("reporter fatal failure");
                throw new InvalidOperationException("reporter failure");
            }
        }
    }

    [Theory]
    [InlineData(false)] [InlineData(true)]
    public void ReporterFailureFollowsNodeFailurePolicyAndAllowsContextReuse(bool scratchWork)
    {
        var fixture = Workload(scratchWork);
        var reporter = new ThrowOnceReporter();
        var execution = Model.Load(fixture.Model).CreateExecution(Options(reporter, reporter));
        Assert.False(execution.Execute(fixture.Feed, false));
        Assert.Equal("reporter failure", Assert.IsType<InvalidOperationException>(execution.LastErrorCause).Message);
        LocalBytes(execution, scratchWork);
        var scratch = new ScratchAccountant(); var copy = new CopyAccountant();
        Assert.True(execution.Execute(fixture.Feed, false, ExecutionProvider.CPU, Options(scratch, copy)), execution.LastErrorMessage);
        Assert.Equal(fixture.Expected, ((Tensor<float>)execution.Outputs["worky"]).ToArray());
        LocalBytes(execution, scratchWork);
        CallerBytes(scratch, copy, scratchWork ? 144 : 0, scratchWork ? 0 : 16);
    }

    [Theory]
    [InlineData(false)] [InlineData(true)]
    public void ReporterFatalFailurePropagatesAndAllowsContextReuse(bool scratchWork)
    {
        var fixture = Workload(scratchWork);
        var reporter = new ThrowOnceReporter(fatal: true);
        var execution = Model.Load(fixture.Model).CreateExecution(Options(reporter, reporter));
        var error = Assert.Throws<OutOfMemoryException>(() => execution.Execute(fixture.Feed, false));
        Assert.Equal("reporter fatal failure", error.Message);
        LocalBytes(execution, scratchWork);
        var scratch = new ScratchAccountant(); var copy = new CopyAccountant();
        Assert.True(execution.Execute(fixture.Feed, false, ExecutionProvider.CPU, Options(scratch, copy)), execution.LastErrorMessage);
        Assert.Equal(fixture.Expected, ((Tensor<float>)execution.Outputs["worky"]).ToArray());
        LocalBytes(execution, scratchWork);
        CallerBytes(scratch, copy, scratchWork ? 144 : 0, scratchWork ? 0 : 16);
    }
}
