using System;
using System.Linq;
using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

public class CpuExecutionProviderPadTests
{
    [Fact]
    public void Conv_HonorsExplicitPadsWhenAutoPadAbsent()
    {
        var x = DenseTensor<float>.OfShape(1, 1, 4, 4);
        x.Fill(1f);
        var w = DenseTensor<float>.OfShape(1, 1, 3, 3);
        w.Fill(1f);
        var r = CPU.Conv(x, w, null, null, null, 1, new int[] { 3, 3 }, new int[] { 1, 1, 1, 1 }, new int[] { 1, 1 }, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs![0];
        Assert.Equal(new[] { 1, 1, 4, 4 }, y.Dimensions.ToArray());
        Assert.Equal(4f, y[0, 0, 0, 0], 5);
        Assert.Equal(9f, y[0, 0, 1, 1], 5);
    }

    [Fact]
    public void MaxPool_HonorsExplicitPadsWhenAutoPadAbsent()
    {
        var x = DenseTensor<float>.OfShape(1, 1, 4, 4);
        for (int i = 0; i < 16; i++) x.SetValue(i, i);
        var r = CPU.MaxPool(x, null, null, null, new int[] { 3, 3 }, new int[] { 1, 1, 1, 1 }, null, new int[] { 1, 1 }, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<float>)r.Outputs![0];
        Assert.Equal(new[] { 1, 1, 4, 4 }, y.Dimensions.ToArray());
    }

    [Fact]
    public void LegacyPad_HonorsFloatValueAttribute()
    {
        var graph = new ComputationalGraph
        {
            Opset = new System.Collections.Generic.Dictionary<string, int> { [""] = 10 },
            Metadata = new System.Collections.Generic.Dictionary<string, object> { ["Name"] = "test" },
        };
        graph.Inputs["x"] = new DenseTensor<float>(new float[] { 1f, 2f }, new[] { 2 });
        var node = new Node
        {
            Name = "pad",
            Op = OpType.Pad,
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Attributes = new System.Collections.Generic.Dictionary<string, object> { ["pads"] = new[] { 1, 1 }, ["value"] = 5f },
        };
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new float[] { 5f, 1f, 2f, 5f }, ((Tensor<float>)r.Outputs[0]).ToArray());
    }

    [Fact]
    public void ModernPad_UsesInputsAtOpset11()
    {
        var graph = new ComputationalGraph
        {
            Opset = new System.Collections.Generic.Dictionary<string, int> { [""] = 13 },
            Metadata = new System.Collections.Generic.Dictionary<string, object> { ["Name"] = "test" },
        };
        graph.Inputs["x"] = new DenseTensor<float>(new float[] { 1f, 2f }, new[] { 2 });
        graph.Inputs["pads"] = new DenseTensor<long>(new long[] { 1L, 1L }, new[] { 2 });
        graph.Inputs["value"] = new DenseTensor<float>(new float[] { 5f }, new[] { 1 });
        var node = new Node
        {
            Name = "pad",
            Op = OpType.Pad,
            Inputs = new[] { "x", "pads", "value" },
            Outputs = new[] { "y" },
            Attributes = new System.Collections.Generic.Dictionary<string, object>(),
        };
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new float[] { 5f, 1f, 2f, 5f }, ((Tensor<float>)r.Outputs[0]).ToArray());
    }
}