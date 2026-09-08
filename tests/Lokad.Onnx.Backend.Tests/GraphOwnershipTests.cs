using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace Lokad.Onnx.Backend.Tests;

public class GraphOwnershipTests
{
    static float[] ToArray(ITensor t) => ((Tensor<float>)t).ToArray();

    static ComputationalGraph NewGraph()
    {
        var g = new ComputationalGraph();
        g.Metadata["Name"] = "test";
        return g;
    }

    [Fact]
    public void ConstantRepeatedRun_IsStable()
    {
        var graph = NewGraph();
        var constTensor = DenseTensor<float>.OfValues(new float[] { 1f, 2f });
        graph.Inputs["x"] = DenseTensor<float>.OfValues(new float[] { 1f, 1f });
        graph.Outputs["y"] = DenseTensor<float>.OfShape(2);
        graph.Nodes.Add(new Node
        {
            Name = "c",
            Op = OpType.Constant,
            Inputs = new string[0],
            Outputs = new[] { "c" },
            Attributes = new Dictionary<string, object> { { "value", constTensor } },
        });
        graph.Nodes.Add(new Node { Name = "add1", Op = OpType.Add, Inputs = new[] { "c", "x" }, Outputs = new[] { "t" } });
        graph.Nodes.Add(new Node { Name = "add2", Op = OpType.Add, Inputs = new[] { "t", "x" }, Outputs = new[] { "y" } });
        graph.IntermediateOutputs["c"] = null;
        graph.IntermediateOutputs["t"] = null;
        graph.RefreshLifetimeAnalysis();

        var x = DenseTensor<float>.OfValues(new float[] { 1f, 1f });
        var inputs = new Dictionary<string, ITensor> { { "x", x } };
        Assert.True(graph.Execute(inputs, true));
        Assert.Equal(new float[] { 3f, 4f }, ToArray(graph.Outputs["y"]));
        Assert.True(graph.Execute(inputs, true));
        Assert.Equal(new float[] { 3f, 4f }, ToArray(graph.Outputs["y"]));
    }

    [Fact]
    public void ExpandNoOp_PreservesCallerInput()
    {
        var graph = NewGraph();
        var callerX = DenseTensor<float>.OfValues(new float[] { 1f, 2f });
        graph.Inputs["x"] = callerX;
        graph.Initializers["eshape"] = DenseTensor<long>.OfValues(new long[] { 2 });
        graph.Outputs["y"] = DenseTensor<float>.OfShape(2);
        graph.Nodes.Add(new Node { Name = "expand", Op = OpType.Expand, Inputs = new[] { "x", "eshape" }, Outputs = new[] { "e" } });
        graph.Nodes.Add(new Node { Name = "add1", Op = OpType.Add, Inputs = new[] { "e", "x" }, Outputs = new[] { "t" } });
        graph.Nodes.Add(new Node { Name = "add2", Op = OpType.Add, Inputs = new[] { "t", "t" }, Outputs = new[] { "y" } });
        graph.IntermediateOutputs["e"] = null;
        graph.IntermediateOutputs["t"] = null;
        graph.RefreshLifetimeAnalysis();

        var inputs = new Dictionary<string, ITensor> { { "x", callerX } };
        Assert.True(graph.Execute(inputs, true));
        Assert.Equal(new float[] { 4f, 8f }, ToArray(graph.Outputs["y"]));
        Assert.Equal(new float[] { 1f, 2f }, ToArray(callerX));
    }

    [Fact]
    public void SameObjectAlias_LiveNamePinsBuffer()
    {
        var graph = NewGraph();
        var shared = DenseTensor<float>.OfValues(new float[] { 1f, 2f });
        graph.Inputs["x"] = shared;
        graph.Inputs["y"] = shared;
        graph.Outputs["z"] = DenseTensor<float>.OfShape(2);
        graph.Nodes.Add(new Node { Name = "add", Op = OpType.Add, Inputs = new[] { "x", "y" }, Outputs = new[] { "z" } });
        graph.RefreshLifetimeAnalysis();
        var inputs = new Dictionary<string, ITensor> { { "x", shared }, { "y", shared } };
        Assert.True(graph.Execute(inputs, true));
        Assert.Equal(new float[] { 2f, 4f }, ToArray(graph.Outputs["z"]));
        Assert.Equal(new float[] { 1f, 2f }, ToArray(shared));
    }

    [Fact]
    public void ReshapeView_PinsParentStorage()
    {
        var graph = NewGraph();
        graph.Inputs["x"] = DenseTensor<float>.OfValues(new float[] { 1f, 2f });
        graph.Inputs["w"] = DenseTensor<float>.OfValues(new float[] { 10f, 20f });
        graph.Initializers["rshape"] = DenseTensor<long>.OfValues(new long[] { 2, 1 });
        graph.Outputs["z"] = DenseTensor<float>.OfShape(2, 1);
        graph.Nodes.Add(new Node { Name = "add", Op = OpType.Add, Inputs = new[] { "x", "w" }, Outputs = new[] { "t" } });
        graph.Nodes.Add(new Node { Name = "reshape", Op = OpType.Reshape, Inputs = new[] { "t", "rshape" }, Outputs = new[] { "r" } });
        graph.Nodes.Add(new Node { Name = "add2", Op = OpType.Add, Inputs = new[] { "r", "r" }, Outputs = new[] { "z" } });
        graph.IntermediateOutputs["t"] = null;
        graph.IntermediateOutputs["r"] = null;
        graph.RefreshLifetimeAnalysis();
        var inputs = new Dictionary<string, ITensor>
        {
            { "x", graph.Inputs["x"] },
            { "w", graph.Inputs["w"] },
        };
        Assert.True(graph.Execute(inputs, true));
        var z = (Tensor<float>)graph.Outputs["z"];
        Assert.Equal(new[] { 2, 1 }, z.Dimensions.ToArray());
        Assert.Equal(new float[] { 22f, 44f }, ToArray(z));
        // Parent t may have been cleared, but view r must still have been correct at use time.
        // If t survived, it must hold 11,22.
        if (graph.IntermediateOutputs.TryGetValue("t", out var t) && t is not null)
            Assert.Equal(new float[] { 11f, 22f }, ToArray(t));
    }

    [Fact]
    public void RetainedOutput_StaysValidAfterLaterNodes()
    {
        var graph = NewGraph();
        graph.Inputs["x"] = DenseTensor<float>.OfValues(new float[] { 1f, 2f });
        graph.Inputs["w"] = DenseTensor<float>.OfValues(new float[] { 10f, 20f });
        graph.Inputs["u"] = DenseTensor<float>.OfValues(new float[] { 100f, 200f });
        graph.Outputs["t"] = DenseTensor<float>.OfShape(2);
        graph.Outputs["z"] = DenseTensor<float>.OfShape(2);
        graph.Nodes.Add(new Node { Name = "add", Op = OpType.Add, Inputs = new[] { "x", "w" }, Outputs = new[] { "t" } });
        graph.Nodes.Add(new Node { Name = "add2", Op = OpType.Add, Inputs = new[] { "t", "u" }, Outputs = new[] { "z" } });
        graph.RefreshLifetimeAnalysis();
        var inputs = new Dictionary<string, ITensor>
        {
            { "x", graph.Inputs["x"] },
            { "w", graph.Inputs["w"] },
            { "u", graph.Inputs["u"] },
        };
        Assert.True(graph.Execute(inputs, true));
        Assert.Equal(new float[] { 11f, 22f }, ToArray(graph.Outputs["t"]));
        Assert.Equal(new float[] { 111f, 222f }, ToArray(graph.Outputs["z"]));
        Assert.True(graph.Execute(inputs, true));
        Assert.Equal(new float[] { 11f, 22f }, ToArray(graph.Outputs["t"]));
        Assert.Equal(new float[] { 111f, 222f }, ToArray(graph.Outputs["z"]));
    }

    [Fact]
    public void SequenceAt_PinsSequenceElements()
    {
        var graph = NewGraph();
        graph.Inputs["x"] = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f } });
        graph.Initializers["split"] = DenseTensor<long>.OfValues(new long[] { 1, 1 });
        graph.Initializers["idx0"] = DenseTensor<long>.OfValues(new long[] { 0 });
        graph.Outputs["z"] = DenseTensor<float>.OfShape(1, 2);
        graph.Nodes.Add(new Node { Name = "split", Op = OpType.SplitToSequence, Inputs = new[] { "x", "split" }, Outputs = new[] { "seq" }, Attributes = new Dictionary<string, object> { { "axis", 0 }, { "keepdims", 1 } } });
        graph.Nodes.Add(new Node { Name = "at", Op = OpType.SequenceAt, Inputs = new[] { "seq", "idx0" }, Outputs = new[] { "z" } });
        graph.IntermediateOutputs["seq"] = null;
        graph.RefreshLifetimeAnalysis();
        var inputs = new Dictionary<string, ITensor> { { "x", graph.Inputs["x"] } };
        Assert.True(graph.Execute(inputs, true));
        Assert.Equal(new float[] { 1f, 2f }, ToArray(graph.Outputs["z"]));
        Assert.True(graph.Execute(inputs, true));
        Assert.Equal(new float[] { 1f, 2f }, ToArray(graph.Outputs["z"]));
    }

    [Fact]
    public void DirtyPooledBuffer_StillProducesCorrectResult()
    {
        var options = ExecutionOptions.Default;
        var pool = new TensorBufferPool();
        var a = DenseTensor<float>.OfValues(new float[] { 1f, 2f });
        var b = DenseTensor<float>.OfValues(new float[] { 10f, 20f });
        var first = CPUExecutionProvider.Add(a, b, options, pool);
        Assert.Equal(OpStatus.Success, first.Status);
        Assert.True(MemoryMarshal.TryGetArray<float>(((DenseTensor<float>)first.Outputs[0]).Buffer, out var seg) && seg.Array is not null);
        for (int i = 0; i < seg.Array.Length; i++) seg.Array[i] = 999f;
        pool.Return(seg.Array!);
        var second = CPUExecutionProvider.Add(a, b, options, pool);
        Assert.Equal(OpStatus.Success, second.Status);
        Assert.Equal(new float[] { 11f, 22f }, ToArray(second.Outputs[0]));
    }
}
