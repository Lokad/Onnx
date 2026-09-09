using System;
using System.Collections.Generic;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;

namespace Lokad.Onnx.Backend.Tests;

// R5: preparation must match the structure that runs, wiring violations must
// fail named at entry, and rejected overlapping calls must not alter the
// active options, outputs, or diagnostics.
public class GraphPreparationConcurrencyTests
{
    static FieldInfo ExecutingFlag() =>
        typeof(ComputationalGraph).GetField("_executing", BindingFlags.Instance | BindingFlags.NonPublic)!;

    static ComputationalGraph ChainGraph()
    {
        var g = new ComputationalGraph();
        g.Metadata["Name"] = "test";
        g.Inputs["x"] = DenseTensor<float>.OfShape(2);
        g.Outputs["y"] = DenseTensor<float>.OfShape(2);
        g.Nodes.Add(new Node { Name = "a", Op = OpType.Relu, Inputs = new[] { "x" }, Outputs = new[] { "t" } });
        g.Nodes.Add(new Node { Name = "b", Op = OpType.Relu, Inputs = new[] { "t" }, Outputs = new[] { "y" } });
        g.IntermediateOutputs["t"] = null;
        g.RefreshLifetimeAnalysis();
        return g;
    }

    static ComputationalGraph UnpreparedChainGraph()
    {
        var g = new ComputationalGraph();
        g.Metadata["Name"] = "test";
        g.Inputs["x"] = DenseTensor<float>.OfShape(2);
        g.Outputs["y"] = DenseTensor<float>.OfShape(2);
        g.Nodes.Add(new Node { Name = "a", Op = OpType.Relu, Inputs = new[] { "x" }, Outputs = new[] { "t" } });
        g.Nodes.Add(new Node { Name = "b", Op = OpType.Relu, Inputs = new[] { "t" }, Outputs = new[] { "y" } });
        g.IntermediateOutputs["t"] = null;
        return g;
    }

    static Dictionary<string, ITensor> Good() =>
        new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[] { -1f, 2f }) } };

    [Fact]
    public void SameCountEdit_Reanalyzes()
    {
        var g = ChainGraph();
        Assert.True(g.Execute(Good(), false));
        Assert.Equal(1, g.LastUseIndex["t"]);
        var edited = g.Nodes[1];
        edited.Inputs = new[] { "x" };
        g.Nodes[1] = edited;
        Assert.True(g.Execute(Good(), false));
        Assert.Equal(0, g.LastUseIndex["t"]);
        Assert.Equal(new float[] { 0f, 2f }, ((Tensor<float>)g.Outputs["y"]).ToArray());
    }

    [Fact]
    public void OutOfOrderConsumption_FailsAtEntry()
    {
        var g = new ComputationalGraph();
        g.Metadata["Name"] = "test";
        g.Inputs["x"] = DenseTensor<float>.OfShape(2);
        g.Outputs["y"] = DenseTensor<float>.OfShape(2);
        g.Nodes.Add(new Node { Name = "a", Op = OpType.Relu, Inputs = new[] { "t" }, Outputs = new[] { "y" } });
        g.Nodes.Add(new Node { Name = "b", Op = OpType.Relu, Inputs = new[] { "x" }, Outputs = new[] { "t" } });
        g.IntermediateOutputs["t"] = null;
        g.RefreshLifetimeAnalysis();
        Assert.False(g.Execute(Good(), false));
        Assert.Null(g.LastFailedNodeName);
        Assert.Contains("t", g.LastErrorMessage ?? "");
    }

    [Fact]
    public void DuplicateProducers_FailNamed()
    {
        var g = ChainGraph();
        var dup = g.Nodes[1];
        dup.Outputs = new[] { "t" };
        g.Nodes[1] = dup;
        Assert.False(g.Execute(Good(), false));
        Assert.Null(g.LastFailedNodeName);
        Assert.Contains("t", g.LastErrorMessage ?? "");
    }

    [Fact]
    public void FileOrder_Preserved()
    {
        var g = new ComputationalGraph();
        g.Metadata["Name"] = "test";
        g.Inputs["x"] = DenseTensor<float>.OfShape(2);
        g.Initializers["c"] = DenseTensor<float>.OfValues(new float[] { 1f, 1f });
        g.Outputs["y"] = DenseTensor<float>.OfShape(2);
        g.Nodes.Add(new Node { Name = "n0", Op = OpType.Sub, Inputs = new[] { "x", "c" }, Outputs = new[] { "t" } });
        g.Nodes.Add(new Node { Name = "n1", Op = OpType.Sub, Inputs = new[] { "t", "c" }, Outputs = new[] { "y" } });
        g.IntermediateOutputs["t"] = null;
        g.RefreshLifetimeAnalysis();
        var user = new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[] { 5f, 1f }) } };
        Assert.True(g.Execute(user, false));
        Assert.Equal(new float[] { 3f, -1f }, ((Tensor<float>)g.Outputs["y"]).ToArray());
    }

    [Fact]
    public void RejectedGraphCall_PreservesState()
    {
        var g = ChainGraph();
        Assert.True(g.Execute(Good(), false));
        Assert.Equal(new float[] { 0f, 2f }, ((Tensor<float>)g.Outputs["y"]).ToArray());
        var flag = ExecutingFlag();
        flag.SetValue(g, 1);
        try
        {
            Assert.False(g.Execute(Good(), false, ExecutionProvider.CPU, ExecutionOptions.Scalar));
            Assert.Equal(new float[] { 0f, 2f }, ((Tensor<float>)g.Outputs["y"]).ToArray());
            Assert.Null(g.LastErrorMessage);
        }
        finally
        {
            flag.SetValue(g, 0);
        }
        Assert.True(g.Execute(Good(), false));
        Assert.Equal(new float[] { 0f, 2f }, ((Tensor<float>)g.Outputs["y"]).ToArray());
    }

    [Fact]
    public async Task ConcurrentFirstCreateExecution_Agrees()
    {
        var g = UnpreparedChainGraph();
        using var barrier = new Barrier(2);
        System.Func<GraphExecution> create = () =>
        {
            barrier.SignalAndWait();
            return g.CreateExecution(null);
        };
        var left = Task.Run(create);
        var right = Task.Run(create);
        await Task.WhenAll(left, right);
        var execLeft = await left;
        var execRight = await right;
        Assert.True(execLeft.Execute(Good(), false));
        Assert.True(execRight.Execute(Good(), false));
        Assert.Equal(new float[] { 0f, 2f }, ((Tensor<float>)execLeft.Outputs["y"]).ToArray());
        Assert.Equal(new float[] { 0f, 2f }, ((Tensor<float>)execRight.Outputs["y"]).ToArray());
        Assert.True(g.Execute(Good(), false));
        Assert.Equal(new float[] { 0f, 2f }, ((Tensor<float>)g.Outputs["y"]).ToArray());
    }

    [Fact]
    public async Task ConcurrentFirstPrepare_Agrees()
    {
        var g = UnpreparedChainGraph();
        using var barrier = new Barrier(2);
        System.Action prepare = () =>
        {
            barrier.SignalAndWait();
            g.Prepare();
        };
        await Task.WhenAll(Task.Run(prepare), Task.Run(prepare));
        Assert.True(g.Execute(Good(), false));
        Assert.Equal(1, g.LastUseIndex["t"]);
        Assert.Equal(new float[] { 0f, 2f }, ((Tensor<float>)g.Outputs["y"]).ToArray());
    }

    [Fact]
    public async Task ConcurrentFirstExecute_Agrees()
    {
        // C09: a genuine four-way race of first executions; losers return false
        // without touching state, so end state holds spec outputs either way.
        var g = UnpreparedChainGraph();
        using var barrier = new Barrier(4);
        var tasks = new Task<bool>[4];
        for (int i = 0; i < tasks.Length; i++)
            tasks[i] = Task.Run(() =>
            {
                barrier.SignalAndWait();
                return g.Execute(Good(), false);
            });
        bool[] results = await Task.WhenAll(tasks);
        Assert.Contains(true, results);
        Assert.Equal(new float[] { 0f, 2f }, ((Tensor<float>)g.Outputs["y"]).ToArray());
        Assert.True(g.Execute(Good(), false));
        Assert.Equal(new float[] { 0f, 2f }, ((Tensor<float>)g.Outputs["y"]).ToArray());
    }

    [Fact]
    public void PrepareDuringExecution_ThrowsAndPreservesState()
    {
        var g = ChainGraph();
        Assert.True(g.Execute(Good(), false));
        Assert.Equal(new float[] { 0f, 2f }, ((Tensor<float>)g.Outputs["y"]).ToArray());
        var flag = ExecutingFlag();
        flag.SetValue(g, 1);
        try
        {
            Assert.Throws<InvalidOperationException>(() => g.Prepare());
            Assert.Equal(new float[] { 0f, 2f }, ((Tensor<float>)g.Outputs["y"]).ToArray());
            Assert.Null(g.LastErrorMessage);
        }
        finally
        {
            flag.SetValue(g, 0);
        }
        Assert.True(g.Execute(Good(), false));
        Assert.Equal(new float[] { 0f, 2f }, ((Tensor<float>)g.Outputs["y"]).ToArray());
    }

    [Fact]
    public void InvalidateDuringExecution_ThrowsAndPreservesState()
    {
        var g = ChainGraph();
        Assert.True(g.Execute(Good(), false));
        Assert.Equal(new float[] { 0f, 2f }, ((Tensor<float>)g.Outputs["y"]).ToArray());
        var flag = ExecutingFlag();
        flag.SetValue(g, 1);
        try
        {
            Assert.Throws<InvalidOperationException>(() => g.InvalidatePreparation());
            Assert.Equal(new float[] { 0f, 2f }, ((Tensor<float>)g.Outputs["y"]).ToArray());
            Assert.Null(g.LastErrorMessage);
        }
        finally
        {
            flag.SetValue(g, 0);
        }
        Assert.True(g.Execute(Good(), false));
        Assert.Equal(new float[] { 0f, 2f }, ((Tensor<float>)g.Outputs["y"]).ToArray());
    }

    [Fact]
    public void RejectedContextCall_PreservesOptionsOutputsDiagnostics()
    {
        var g = ChainGraph();
        var ctx = g.CreateExecution(null);
        var before = ctx.Options;
        Assert.True(ctx.Execute(Good(), false));
        Assert.Equal(new float[] { 0f, 2f }, ((Tensor<float>)ctx.Outputs["y"]).ToArray());
        var flag = ExecutingFlag();
        flag.SetValue(ctx, 1);
        try
        {
            Assert.False(ctx.Execute(Good(), false, ExecutionProvider.CPU, ExecutionOptions.Intrinsics));
            Assert.Equal(before, ctx.Options);
            Assert.Equal(new float[] { 0f, 2f }, ((Tensor<float>)ctx.Outputs["y"]).ToArray());
            Assert.Null(ctx.LastErrorMessage);
        }
        finally
        {
            flag.SetValue(ctx, 0);
        }
        Assert.True(ctx.Execute(Good(), false));
        Assert.Equal(new float[] { 0f, 2f }, ((Tensor<float>)ctx.Outputs["y"]).ToArray());
    }
}
