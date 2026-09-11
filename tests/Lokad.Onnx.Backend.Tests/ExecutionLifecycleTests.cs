extern alias OnnxSharp;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Google.Protobuf;
using OnnxSharp::Onnx;
using Xunit;

namespace Lokad.Onnx.Backend.Tests;

[CollectionDefinition("SequentialLogSink", DisableParallelization = true)]
public class SequentialLogSinkCollection { }

[Collection("SequentialLogSink")]
public class ExecutionLifecycleTests
{
    static ComputationalGraph NewReluGraph()
    {
        var g = new ComputationalGraph();
        g.Metadata["Name"] = "test";
        g.Inputs["x"] = DenseTensor<float>.OfShape(2);
        g.Outputs["y"] = DenseTensor<float>.OfShape(2);
        g.Nodes.Add(new Node { Name = "r", Op = OpType.Relu, Inputs = new[] { "x" }, Outputs = new[] { "y" } });
        g.RefreshLifetimeAnalysis();
        return g;
    }

    static List<string> CaptureLog(Action exercise)
    {
        var lines = new List<string>();
        var previous = Log.Sink;
        Log.Sink = (level, message) => { lock (lines) lines.Add(level + ":" + message); };
        try { exercise(); }
        finally { Log.Sink = previous; }
        return lines;
    }

    [Fact]
    public void RunTime_ReconcilesNodeProfiles()
    {
        var g = NewReluGraph();
        Assert.Equal(TimeSpan.Zero, g.LastRunTime);
        var good = new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[] { -1f, 2f }) } };
        Assert.True(g.Execute(good, false));
        Assert.NotNull(g.LastProfile);
        Assert.True(g.LastRunTime > TimeSpan.Zero);
        var nodes = TimeSpan.Zero;
        foreach (var node in g.LastProfile!)
        {
            foreach (var stage in node.OpsProfile) nodes += stage.Time;
        }
        Assert.True(nodes <= g.LastRunTime, "Node time " + nodes + " exceeds run time " + g.LastRunTime + ".");
        Assert.True(g.LastRunTime - nodes >= TimeSpan.Zero);
    }

    [Fact]
    public void RunTime_PublishedBackToFacade()
    {
        var g = NewReluGraph();
        var ctx = g.CreateExecution(null);
        var good = new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[] { -1f, 2f }) } };
        Assert.True(ctx.Execute(good, false));
        Assert.True(ctx.LastRunTime > TimeSpan.Zero);
        Assert.True(g.Execute(good, false));
        Assert.True(g.LastRunTime > TimeSpan.Zero);
    }

    [Fact]
    public void MissingInput_FailsClean_AndRetrySucceeds()
    {
        var g = NewReluGraph();
        var bad = new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[] { 1f, 2f, 3f }) } };
        Assert.False(g.Execute(bad, false));
        Assert.NotNull(g.LastErrorMessage);
        Assert.Null(g.LastFailedNodeName);
        Assert.Empty(g.Outputs);
        var good = new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[] { -1f, 2f }) } };
        Assert.True(g.Execute(good, false));
        Assert.Null(g.LastErrorMessage);
        Assert.Equal(new float[] { 0f, 2f }, ((Tensor<float>)g.Outputs["y"]).ToArray());
    }

    [Fact]
    public void DictBind_Success_CompletesScopeWithoutAbandon()
    {
        var lines = CaptureLog(() =>
        {
            var g = NewReluGraph();
            var good = new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[] { 1f, 2f }) } };
            Assert.True(g.Execute(good, false));
        });
        Assert.Contains(lines, l => l.Contains("graph inputs for execution completed"));
        Assert.DoesNotContain(lines, l => l.Contains("abandoned"));
    }

    [Fact]
    public void DictBind_Failure_AbandonsScope_AndInvalidatesOutputs()
    {
        var g = NewReluGraph();
        var good = new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[] { 1f, 2f }) } };
        Assert.True(g.Execute(good, false));
        var lines = CaptureLog(() =>
        {
            var bad = new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[] { 1f }) } };
            Assert.False(g.Execute(bad, false));
        });
        Assert.Contains(lines, l => l.Contains("abandoned"));
        Assert.Empty(g.Outputs);
    }

    [Fact]
    public void UnsupportedOperator_FailsWithNodeIdentity_AndRetrySucceeds()
    {
        var g = NewReluGraph();
        g.Nodes.Add(new Node { Name = "u", Op = OpType.Unknown, OpTypeName = "FancyNewOp", Inputs = new[] { "y" }, Outputs = new[] { "z" } });
        g.Outputs["z"] = DenseTensor<float>.OfShape(2);
        g.RefreshLifetimeAnalysis();
        var good = new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[] { 1f, 2f }) } };
        Assert.False(g.Execute(good, false));
        Assert.Equal("u", g.LastFailedNodeName);
        Assert.Equal(OpType.Unknown, g.LastFailedNodeOp);
        Assert.Contains("FancyNewOp", g.LastErrorMessage ?? "");
        Assert.Empty(g.Outputs);
    }

    [Fact]
    public void ParseFailure_LoadReturnsNull()
    {
        Assert.Null(OnnxImport.Load(Path.Combine(Path.GetTempPath(), Path.GetRandomFileName() + ".onnx")));
        Assert.Null(OnnxImport.Load(new byte[] { 1, 2, 3, 4 }));
    }

    [Fact]
    public void PreparationFailure_LoadReturnsNullInsteadOfThrowing()
    {
        var model = new ModelProto { Graph = new GraphProto() };
        foreach (var name in new[] { "w", "w" })
        {
            var init = new TensorProto { Name = name, DataType = (int)TensorElementType.Float };
            init.Dims.Add(2);
            init.FloatData.Add(1f);
            init.FloatData.Add(2f);
            model.Graph.Initializer.Add(init);
        }
        Assert.Null(OnnxImport.Load(model.ToByteArray()));
    }

    [Fact]
    public void InstrumentedRun_EmitsBindingNodeAndModelLines()
    {
        var modelLines = CaptureLog(() =>
        {
            var mp = new OnnxModel { Name = "logged" };
            mp.Opset[""] = 11;
            Model.Load(mp);
        });
        Assert.Contains(modelLines, l => l.Contains("Model details:"));
        var g = NewReluGraph();
        g.Initializers["x"] = DenseTensor<float>.OfValues(new float[] { 1f, 2f });
        var bindLines = CaptureLog(() =>
        {
            Assert.True(g.Execute(new Dictionary<string, ITensor>(), true));
        });
        Assert.Contains(bindLines, l => l.Contains("Using initializer value"));
        Assert.Equal(new float[] { 1f, 2f }, ((Tensor<float>)g.Outputs["y"]).ToArray());
        var nodeLines = CaptureLog(() =>
        {
            var user = new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[] { -1f, 2f }) } };
            Assert.True(g.ExecuteNode(user, "r", false));
        });
        Assert.Contains(nodeLines, l => l.Contains("Executing node r"));
    }

    [Fact]
    public void LoadFailure_RecordsCause_AccessibleWithoutSink()
    {
        var previous = Log.Sink;
        Log.Sink = null;
        try
        {
            Assert.Null(OnnxImport.Load(new byte[] { 1, 2, 3, 4 }));
            Assert.NotNull(OnnxImport.LastErrorMessage);
            Assert.NotNull(OnnxImport.LastErrorCause);
            var missing = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName() + ".onnx");
            Assert.Null(OnnxImport.Load(missing));
            Assert.Contains(".onnx", OnnxImport.LastErrorMessage ?? "");
            Assert.NotNull(OnnxImport.LastErrorCause);
        }
        finally { Log.Sink = previous; }
    }

    [Fact]
    public void OperatorException_PreservesCause_AndRetrySucceeds()
    {
        var g = new ComputationalGraph();
        g.Metadata["Name"] = "test";
        g.Inputs["x"] = DenseTensor<float>.OfShape(2);
        g.Inputs["s"] = DenseTensor<long>.OfShape(1);
        g.Outputs["y"] = DenseTensor<float>.OfShape(2);
        g.Nodes.Add(new Node
        {
            Name = "rs",
            Op = OpType.Reshape,
            Inputs = new[] { "x", "s" },
            Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object> { { "allowzero", "yes" } },
        });
        g.RefreshLifetimeAnalysis();
        var lines = CaptureLog(() =>
        {
            var user = new Dictionary<string, ITensor>
            {
                { "x", DenseTensor<float>.OfValues(new float[] { 1f, 2f }) },
                { "s", DenseTensor<long>.OfValues(new long[] { 2 }) },
            };
            Assert.False(g.Execute(user, false));
        });
        Assert.Contains("allowzero", g.LastErrorMessage ?? "");
        Assert.Equal("rs", g.LastFailedNodeName);
        var cause = (Exception?)typeof(ComputationalGraph).GetProperty("LastErrorCause")?.GetValue(g);
        Assert.IsType<ArgumentException>(cause);
        Assert.Contains(lines, l => l.Contains("ArgumentException"));
        Assert.Empty(g.Outputs);
        var fixed_ = g.Nodes[0];
        fixed_.Attributes = new Dictionary<string, object>();
        g.Nodes[0] = fixed_;
        var retry = new Dictionary<string, ITensor>
        {
            { "x", DenseTensor<float>.OfValues(new float[] { 1f, 2f }) },
            { "s", DenseTensor<long>.OfValues(new long[] { 2 }) },
        };
        Assert.True(g.Execute(retry, false));
        Assert.Null(g.LastErrorMessage);
        Assert.Equal(new float[] { 1f, 2f }, ((Tensor<float>)g.Outputs["y"]).ToArray());
    }

    [Fact]
    public void MemoryDiagnostics_StartAtZero()
    {
        var g = NewReluGraph();
        Assert.Equal(0, g.LastAllocatedBytes);
        Assert.NotNull(g.LastGcCollections);
        Assert.Equal(3, g.LastGcCollections.Length);
        Assert.All(g.LastGcCollections, c => Assert.Equal(0, c));
    }

    [Fact]
    public void MemoryDiagnostics_RecordsAllocation_AndPublishesBack()
    {
        var g = NewReluGraph();
        var ctx = g.CreateExecution(null);
        var good = new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[] { -1f, 2f }) } };
        Assert.True(ctx.Execute(good, false));
        Assert.True(ctx.LastAllocatedBytes > 0, "Run allocated " + ctx.LastAllocatedBytes + " bytes.");
        Assert.NotNull(ctx.LastGcCollections);
        Assert.Equal(3, ctx.LastGcCollections.Length);
        Assert.All(ctx.LastGcCollections, c => Assert.True(c >= 0));
        Assert.True(g.Execute(good, false));
        Assert.True(g.LastAllocatedBytes > 0, "Facade published " + g.LastAllocatedBytes + " bytes.");
        Assert.Equal(3, g.LastGcCollections.Length);
    }

    [Fact]
    public void PoolPeakOutstanding_PublishesBack()
    {
        // MatMul destinations rent cleared pool storage, so the 2x2 float
        // output alone keeps at least 16 bytes outstanding at run end.
        var g = new ComputationalGraph();
        g.Metadata["Name"] = "test";
        g.Inputs["a"] = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f } });
        g.Inputs["b"] = DenseTensor<float>.OfValues(new float[,] { { 5f, 6f }, { 7f, 8f } });
        g.Outputs["y"] = DenseTensor<float>.OfShape(2, 2);
        g.Nodes.Add(new Node { Name = "m", Op = OpType.MatMul, Inputs = new[] { "a", "b" }, Outputs = new[] { "y" } });
        g.RefreshLifetimeAnalysis();
        var ctx = g.CreateExecution(null);
        var good = new Dictionary<string, ITensor>
        {
            { "a", DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f } }) },
            { "b", DenseTensor<float>.OfValues(new float[,] { { 5f, 6f }, { 7f, 8f } }) },
        };
        Assert.True(ctx.Execute(good, false));
        Assert.True(ctx.LastPoolPeakOutstandingBytes >= 16, "Run peaked at " + ctx.LastPoolPeakOutstandingBytes + " bytes.");
        Assert.True(g.Execute(good, false));
        Assert.True(g.LastPoolPeakOutstandingBytes >= 16, "Facade published " + g.LastPoolPeakOutstandingBytes + " bytes.");
    }

    [Fact]
    public void ScratchAccount_ParallelAdds_Exact()
    {
        var acc = new ScratchAccountant();
        System.Threading.Tasks.Parallel.For(0, 8, _ => acc.AddScratchBytes(1000));
        Assert.Equal(8000, acc.TotalScratchBytes);
    }

    [Fact]
    public void ConvScratch_ReportsPatchBytes()
    {
        // Single-batch VALID conv rents one im2col patch: 1*2*2*3*3 extents.
        static ExecutionOptions WithScratch(IScratchAccountant acc, TensorExecutionOptions tensor) =>
            ExecutionOptions.Default with { Tensor = tensor with { ScratchReporter = acc } };
        var x = DenseTensor<float>.OfValues(new float[1, 1, 4, 4]
        {
            { { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f }, { 9f, 10f, 11f, 12f }, { 13f, 14f, 15f, 16f } } },
        });
        var w = DenseTensor<float>.OfValues(new float[1, 1, 2, 2] { { { { 1f, 1f }, { 1f, 1f } } } });
        var acc = new ScratchAccountant();
        var r = CPUExecutionProvider.Conv(x, w, null, null, null, 1, null, null, null, WithScratch(acc, TensorExecutionOptions.Scalar));
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(36 * 4, acc.TotalScratchBytes);
        var xd = DenseTensor<double>.OfValues(new double[1, 1, 4, 4]
        {
            { { { 1.0, 2.0, 3.0, 4.0 }, { 5.0, 6.0, 7.0, 8.0 }, { 9.0, 10.0, 11.0, 12.0 }, { 13.0, 14.0, 15.0, 16.0 } } },
        });
        var wd = DenseTensor<double>.OfValues(new double[1, 1, 2, 2] { { { { 1.0, 1.0 }, { 1.0, 1.0 } } } });
        var accd = new ScratchAccountant();
        var rd = CPUExecutionProvider.Conv(xd, wd, null, null, null, 1, null, null, null, WithScratch(accd, TensorExecutionOptions.Scalar));
        Assert.Equal(OpStatus.Success, rd.Status);
        Assert.Equal(36 * 8, accd.TotalScratchBytes);
    }

    [SkippableFact]
    public void MatMulPackedScratch_ReportsPackBytes()
    {
        Skip.If(!System.Runtime.Intrinsics.X86.Fma.IsSupported, "x86 FMA not available on this machine.");
        // 64+ rows route the panel-packed kernel: one n*k pack of floats.
        var a = DenseTensor<float>.OfShape(64, 8);
        a.Fill(1f);
        var b = DenseTensor<float>.OfShape(8, 8);
        b.Fill(1f);
        var acc = new ScratchAccountant();
        var options = ExecutionOptions.Default with { Tensor = TensorExecutionOptions.Intrinsics with { ScratchReporter = acc } };
        var r = CPUExecutionProvider.MatMul(a, b, options, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(8 * 8 * 4, acc.TotalScratchBytes);
    }

    [Fact]
    public void ScratchBytes_PublishThroughExecution()
    {
        // The 2x2 VALID conv above rents one 36-float patch per run through
        // dispatch, published to the context and back to the facade graph.
        var g = new ComputationalGraph();
        g.Metadata["Name"] = "test";
        g.Inputs["x"] = DenseTensor<float>.OfValues(new float[1, 1, 4, 4]
        {
            { { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f }, { 9f, 10f, 11f, 12f }, { 13f, 14f, 15f, 16f } } },
        });
        g.Inputs["w"] = DenseTensor<float>.OfValues(new float[1, 1, 2, 2] { { { { 1f, 1f }, { 1f, 1f } } } });
        g.Outputs["z"] = DenseTensor<float>.OfShape(1, 1, 3, 3);
        g.Nodes.Add(new Node { Name = "c", Op = OpType.Conv, Inputs = new[] { "x", "w" }, Outputs = new[] { "z" } });
        g.RefreshLifetimeAnalysis();
        var ctx = g.CreateExecution(null);
        var good = new Dictionary<string, ITensor>
        {
            { "x", DenseTensor<float>.OfValues(new float[1, 1, 4, 4]
            {
                { { { 1f, 2f, 3f, 4f }, { 5f, 6f, 7f, 8f }, { 9f, 10f, 11f, 12f }, { 13f, 14f, 15f, 16f } } },
            }) },
            { "w", DenseTensor<float>.OfValues(new float[1, 1, 2, 2] { { { { 1f, 1f }, { 1f, 1f } } } }) },
        };
        Assert.True(ctx.Execute(good, false));
        Assert.Equal(36 * 4, ctx.LastScratchBytes);
        Assert.True(g.Execute(good, false));
        Assert.Equal(36 * 4, g.LastScratchBytes);
    }

    [Fact]
    public void LivePeakBytes_CoversBoundTensors()
    {
        // The 2x2 float output alone keeps 16 bytes bound through the run,
        // however aliasing resolves; inputs only add to the peak.
        var g = NewMatMulGraph(2);
        var ctx = g.CreateExecution(null);
        var good = MatMulInputs(2);
        Assert.True(ctx.Execute(good, false));
        Assert.True(ctx.LastPeakLiveBytes >= 16, "Run peaked at " + ctx.LastPeakLiveBytes + " bytes.");
        Assert.True(g.Execute(good, false));
        Assert.True(g.LastPeakLiveBytes >= 16, "Facade published " + g.LastPeakLiveBytes + " bytes.");
    }

    [Fact]
    public void LivePeakBytes_GrowsWithWorkloadAndRepeatsExactly()
    {
        var small = NewMatMulGraph(2);
        Assert.True(small.Execute(MatMulInputs(2), false));
        var big = NewMatMulGraph(4);
        Assert.True(big.Execute(MatMulInputs(4), false));
        Assert.True(big.LastPeakLiveBytes > small.LastPeakLiveBytes,
            "4x4 peak " + big.LastPeakLiveBytes + " vs 2x2 peak " + small.LastPeakLiveBytes + ".");
        Assert.True(small.Execute(MatMulInputs(2), false));
        var rerun = NewMatMulGraph(2);
        Assert.True(rerun.Execute(MatMulInputs(2), false));
        Assert.Equal(small.LastPeakLiveBytes, rerun.LastPeakLiveBytes);
    }

    [Fact]
    public void NodeProfiles_CarryInputShapesAndDtypes()
    {
        var g = NewMatMulGraph(2);
        var good = MatMulInputs(2);
        using (Profiler.BeginExecution(true))
        {
            Assert.True(g.Execute(good, false));
        }
        Assert.NotNull(g.LastProfile);
        var node = g.LastProfile!.Peek();
        Assert.Equal(OpType.MatMul, node.Op);
        Assert.Contains("float", node.Detail);
        Assert.Contains("2x2", node.Detail);
    }
    static ComputationalGraph NewMatMulGraph(int n)
    {
        var g = new ComputationalGraph();
        g.Metadata["Name"] = "test";
        g.Inputs["a"] = DenseTensor<float>.OfShape(n, n);
        g.Inputs["b"] = DenseTensor<float>.OfShape(n, n);
        g.Outputs["y"] = DenseTensor<float>.OfShape(n, n);
        g.Nodes.Add(new Node { Name = "m", Op = OpType.MatMul, Inputs = new[] { "a", "b" }, Outputs = new[] { "y" } });
        g.RefreshLifetimeAnalysis();
        return g;
    }

    static Dictionary<string, ITensor> MatMulInputs(int n)
    {
        var a = DenseTensor<float>.OfShape(n, n);
        a.Fill(1f);
        var b = DenseTensor<float>.OfShape(n, n);
        b.Fill(1f);
        return new Dictionary<string, ITensor> { { "a", a }, { "b", b } };
    }

    [Fact]
    public void MemoryDiagnostics_StampsFreshDeltasPerRun()
    {
        var g = NewReluGraph();
        var good = new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[] { -1f, 2f }) } };
        Assert.True(g.Execute(good, false));
        var firstBytes = g.LastAllocatedBytes;
        var firstGc = g.LastGcCollections;
        Assert.True(firstBytes > 0);
        Assert.True(g.Execute(good, false));
        Assert.True(g.LastAllocatedBytes > 0);
        Assert.False(ReferenceEquals(firstGc, g.LastGcCollections), "Second run must stamp a fresh generation array, not accumulate.");
        Assert.Equal(3, g.LastGcCollections.Length);
    }

    [Fact]
    public void CopyAccount_ParallelAdds_Exact()
    {
        var acc = new CopyAccountant();
        System.Threading.Tasks.Parallel.For(0, 8, _ => acc.AddCopyBytes(1000));
        Assert.Equal(8000, acc.TotalCopyBytes);
    }

    [Fact]
    public void SlicedViewMatMul_ReportsCopyBytes()
    {
        // A sliced 2x2 view (shared storage, exotic strides) densifies to
        // 16 payload bytes on the way into the float MatMul kernel; dense
        // operands copy nothing. (Tensor.Transpose eagerly materializes,
        // so views here come from Slice, as in ViewConsistencyTests.)
        var v = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } })
            .Slice(new SliceIndex(0, 2), new SliceIndex(1, 3));
        var b = DenseTensor<float>.OfValues(new float[,] { { 1f, 0f }, { 0f, 1f } });
        var acc = new CopyAccountant();
        var options = TensorExecutionOptions.Scalar with { CopyReporter = acc };
        var got = Tensor<float>.MatMul(v, b, options);
        Assert.Equal(new float[] { 2f, 3f, 5f, 6f }, got.ToArray());
        Assert.Equal(4 * 4, acc.TotalCopyBytes);
        var acc2 = new CopyAccountant();
        var dense = DenseTensor<float>.OfValues(new float[,] { { 2f, 3f }, { 5f, 6f } });
        var options2 = TensorExecutionOptions.Scalar with { CopyReporter = acc2 };
        Tensor<float>.MatMul(dense, b, options2);
        Assert.Equal(0, acc2.TotalCopyBytes);
    }

    [Fact]
    public void CopyBytes_PublishThroughExecution()
    {
        // Slice emits a view intermediate; MatMul densifies it through
        // dispatch, published to the context and back to the facade graph.
        // Payload: one 2x2 float view, 16 bytes; nothing else copies.
        var g = new ComputationalGraph();
        g.Metadata["Name"] = "test";
        g.Inputs["x"] = DenseTensor<float>.OfShape(2, 3);
        g.Inputs["b"] = DenseTensor<float>.OfShape(2, 2);
        g.Inputs["st"] = DenseTensor<long>.OfShape(2);
        g.Inputs["e"] = DenseTensor<long>.OfShape(2);
        g.Inputs["ax"] = DenseTensor<long>.OfShape(2);
        g.Inputs["sp"] = DenseTensor<long>.OfShape(2);
        g.Outputs["z"] = DenseTensor<float>.OfShape(2, 2);
        g.Nodes.Add(new Node
        {
            Name = "s", Op = OpType.Slice, OpTypeName = OpType.Slice.ToString(), Domain = "",
            OpsetVersion = 13, IsFused = false,
            Inputs = new[] { "x", "st", "e", "ax", "sp" }, Outputs = new[] { "v" },
            Attributes = new Dictionary<string, object>(),
        });
        g.Nodes.Add(new Node { Name = "m", Op = OpType.MatMul, Inputs = new[] { "v", "b" }, Outputs = new[] { "z" } });
        g.RefreshLifetimeAnalysis();
        var good = new Dictionary<string, ITensor>
        {
            { "x", DenseTensor<float>.OfValues(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } }) },
            { "b", DenseTensor<float>.OfValues(new float[,] { { 1f, 0f }, { 0f, 1f } }) },
            { "st", DenseTensor<long>.OfValues(new long[] { 0L, 1L }) },
            { "e", DenseTensor<long>.OfValues(new long[] { 2L, 3L }) },
            { "ax", DenseTensor<long>.OfValues(new long[] { 0L, 1L }) },
            { "sp", DenseTensor<long>.OfValues(new long[] { 1L, 1L }) },
        };
        var ctx = g.CreateExecution(null);
        Assert.True(ctx.Execute(good, false), ctx.LastErrorMessage);
        Assert.Equal(new float[] { 2f, 3f, 5f, 6f }, ((Tensor<float>)ctx.Outputs["z"]).ToArray());
        Assert.Equal(16, ctx.LastCopyBytes);
        Assert.True(g.Execute(good, false), g.LastErrorMessage);
        Assert.Equal(16, g.LastCopyBytes);
    }

    [Fact]
    public void ExecuteNode_FailedThenSuccessful_ClearsErrorSnapshot()
    {
        var g = NewReluGraph();
        var user = new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[] { 1f, 2f }) } };
        Assert.False(g.ExecuteNode(user, "nope", false));
        Assert.NotNull(g.LastErrorMessage);
        Assert.True(g.ExecuteNode(user, "r", false));
        Assert.Null(g.LastErrorMessage);
        Assert.Null(g.LastFailedNodeName);
    }

    [Fact]
    public void ReshapeMismatch_FailsCleanlySecondRun()
    {
        // ORT refuses mismatched feed shapes at run; a second Execute
        // with different shapes must fail descriptively, not corrupt
        // the first run or throw.
        var g = NewMatMulGraph(2);
        Assert.True(g.Execute(MatMulInputs(2), false));
        Assert.False(g.Execute(MatMulInputs(3), false));
        Assert.NotNull(g.LastErrorMessage);
        var h = NewMatMulGraph(3);
        Assert.True(h.Execute(MatMulInputs(3), false));
        Assert.False(h.Execute(MatMulInputs(2), false));
    }

    [Fact]
    public void MismatchThenRetry_RecoversWithValues()
    {
        // A shape-mismatch failure must not poison the graph: the next
        // good run recomputes exact values and clears diagnostics.
        var g = NewMatMulGraph(2);
        Assert.True(g.Execute(MatMulInputs(2), false));
        Assert.False(g.Execute(MatMulInputs(3), false));
        Assert.True(g.Execute(MatMulInputs(2), false));
        var y = (Tensor<float>)g.Outputs["y"];
        Assert.Equal(new float[] { 2f, 2f, 2f, 2f }, y.ToArray());
        Assert.Null(g.LastErrorMessage);
        Assert.Null(g.LastFailedNodeName);
    }

    [Fact]
    public void SymbolicDims_ExecuteVaryingShapes()
    {
        // A 0 extent marks a symbolic dimension: it accepts the first
        // concrete shape, then that binding wins and a later shape fails
        // descriptively with diagnostics (hand-built graphs carry no
        // retained InputDescs; imported graphs keep symbolic descs and
        // re-execute at new shapes, e5 does this live).
        var g = new ComputationalGraph();
        g.Metadata["Name"] = "test";
        g.Inputs["x"] = DenseTensor<float>.OfShape(0);
        g.Outputs["y"] = DenseTensor<float>.OfShape(0);
        g.Nodes.Add(new Node { Name = "r", Op = OpType.Relu, Inputs = new[] { "x" }, Outputs = new[] { "y" } });
        g.RefreshLifetimeAnalysis();
        var two = new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[] { -1f, 2f }) } };
        Assert.True(g.Execute(two, false));
        Assert.Equal(new float[] { 0f, 2f }, ((Tensor<float>)g.Outputs["y"]).ToArray());
        var three = new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[] { -1f, 2f, -3f }) } };
        Assert.False(g.Execute(three, false));
        Assert.NotNull(g.LastErrorMessage);
    }
}
