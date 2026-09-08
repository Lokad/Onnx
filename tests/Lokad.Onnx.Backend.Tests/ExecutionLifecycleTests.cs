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
}
