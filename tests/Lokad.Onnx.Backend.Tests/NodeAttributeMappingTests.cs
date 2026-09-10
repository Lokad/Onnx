using System;
using System.Collections.Generic;
using System.Linq;

namespace Lokad.Onnx.Backend.Tests;

public class NodeAttributeMappingTests
{
    private static ComputationalGraph CreateGraph(int opsetVersion)
    {
        return new ComputationalGraph
        {
            Opset = new Dictionary<string, int> { [""] = opsetVersion },
            Metadata = new Dictionary<string, object> { ["Name"] = "test" },
        };
    }

    [Fact]
    public void Transpose_UsesPermAttribute()
    {
        var graph = CreateGraph(13);
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f } });
        graph.Inputs["x"] = x;

        var node = new Node
        {
            Name = "transpose",
            Op = OpType.Transpose,
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object> { ["perm"] = new[] { 1, 0 } }
        };

        var result = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Success, result.Status);
        var y = (Tensor<float>)result.Outputs[0];
        Assert.Equal(new[] { 2, 2 }, y.Dimensions.ToArray());
        Assert.Equal(2f, y[1, 0], 5);
        Assert.Equal(3f, y[0, 1], 5);
    }

    [Fact]
    public void Concat_UsesAxisAttribute()
    {
        var graph = CreateGraph(13);
        graph.Inputs["a"] = DenseTensor<int>.OfValues(new int[,] { { 1, 2 } });
        graph.Inputs["b"] = DenseTensor<int>.OfValues(new int[,] { { 3, 4 } });

        var node = new Node
        {
            Name = "concat",
            Op = OpType.Concat,
            Inputs = new[] { "a", "b" },
            Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object> { ["axis"] = 0 }
        };

        var result = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Success, result.Status);
        var y = (Tensor<int>)result.Outputs[0];
        Assert.Equal(new[] { 2, 2 }, y.Dimensions.ToArray());
        Assert.Equal(3, y[1, 0]);
    }

    [Fact]
    public void Softmax_UsesAxisAttribute()
    {
        var graph = CreateGraph(13);
        graph.Inputs["x"] = DenseTensor<float>.OfValues(new float[,] { { 0f, 1f }, { 1f, 1f } });

        var node = new Node
        {
            Name = "softmax",
            Op = OpType.Softmax,
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object> { ["axis"] = 1 }
        };

        var result = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Success, result.Status);
        var y = (Tensor<float>)result.Outputs[0];
        Assert.Equal(1f, y[0, 0] + y[0, 1], 5);
        Assert.Equal(0.5f, y[1, 0], 5);
        Assert.Equal(0.5f, y[1, 1], 5);
    }

    [Fact]
    public void Softmax_OmittedAxis_ResolvesByOpset()
    {
        // ORT 1.29 on arange(6) shaped [1,2,3]: opset 11 omits to axis 1,
        // opset 13 omits to axis -1 (provider defaults pinned separately).
        var g13 = CreateGraph(13);
        g13.Inputs["x"] = DenseTensor<float>.OfValues(new float[1, 2, 3] { { { 0f, 1f, 2f }, { 3f, 4f, 5f } } });
        var n13 = new Node
        {
            Name = "softmax",
            Op = OpType.Softmax,
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object>()
        };
        var r13 = n13.Execute(g13, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Success, r13.Status);
        var y13 = ((Tensor<float>)r13.Outputs[0]).ToArray();
        Assert.Equal(0.09003f, y13[0], 5);
        Assert.Equal(0.66524f, y13[2], 5);
        var g11 = CreateGraph(11);
        g11.Inputs["x"] = DenseTensor<float>.OfValues(new float[1, 2, 3] { { { 0f, 1f, 2f }, { 3f, 4f, 5f } } });
        var n11 = new Node
        {
            Name = "softmax",
            Op = OpType.Softmax,
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object>()
        };
        var r11 = n11.Execute(g11, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Success, r11.Status);
        var y11 = ((Tensor<float>)r11.Outputs[0]).ToArray();
        Assert.Equal(0.00427f, y11[0], 5);
        Assert.Equal(0.63369f, y11[5], 5);
    }

    [Fact]
    public void Conv_OmittedAttributes_UseDefaults()
    {
        // ORT 1.29: a bare two-input Conv infers the kernel from weights
        // with VALID pads and unit stride/dilation ([[1,2],[3,4]] x 2).
        var graph = CreateGraph(13);
        graph.Inputs["x"] = DenseTensor<float>.OfValues(new float[1, 1, 2, 2] { { { { 1f, 2f }, { 3f, 4f } } } });
        graph.Inputs["w"] = DenseTensor<float>.OfValues(new float[1, 1, 1, 1] { { { { 2f } } } });
        var node = new Node
        {
            Name = "conv",
            Op = OpType.Conv,
            Inputs = new[] { "x", "w" },
            Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object>()
        };
        var result = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Success, result.Status);
        var y = (Tensor<float>)result.Outputs[0];
        Assert.Equal(new[] { 1, 1, 2, 2 }, y.Dimensions.ToArray());
        Assert.Equal(new float[] { 2f, 4f, 6f, 8f }, y.ToArray());
    }

    [Fact]
    public void MaxPool_OmittedStrides_DefaultToOne()
    {
        // ORT 1.29: omitted strides default to 1; [1,1,3,3] with kernel 2
        // yields [1,1,2,2] [[5,6],[8,9]] (provider default pinned
        // separately, this covers the node fill path).
        var graph = CreateGraph(13);
        graph.Inputs["x"] = DenseTensor<float>.OfValues(new float[1, 1, 3, 3] { { { { 1f, 2f, 3f }, { 4f, 5f, 6f }, { 7f, 8f, 9f } } } });
        var node = new Node
        {
            Name = "pool",
            Op = OpType.MaxPool,
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object> { ["kernel_shape"] = new long[] { 2L, 2L } }
        };
        var result = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Success, result.Status);
        var y = (Tensor<float>)result.Outputs[0];
        Assert.Equal(new[] { 1, 1, 2, 2 }, y.Dimensions.ToArray());
        Assert.Equal(new float[] { 5f, 6f, 8f, 9f }, y.ToArray());
    }

    [Fact]
    public void Reshape_AllowZeroAttribute_Maps()
    {
        // ORT 1.29: allowzero=1 keeps literal zeros ([2,0,3] @ [0,-1,0]
        // -> [0,6,0]); the node fill maps the attribute through.
        var graph = CreateGraph(14);
        graph.Inputs["x"] = DenseTensor<float>.OfShape(2, 0, 3);
        graph.Inputs["s"] = DenseTensor<long>.OfValues(new long[] { 0L, -1L, 0L });
        var node = new Node
        {
            Name = "reshape",
            Op = OpType.Reshape,
            Inputs = new[] { "x", "s" },
            Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object> { ["allowzero"] = 1L }
        };
        var result = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.Equal(new[] { 0, 6, 0 }, ((Tensor<float>)result.Outputs[0]).Dimensions.ToArray());
    }

    [Fact]
    public void Conv_UsesPadsStridesKernelShapeAttributes()
    {
        var graph = CreateGraph(13);
        var x = DenseTensor<float>.OfValues(new float[1, 1, 3, 3] { { {
            { 1f, 2f, 3f }, { 4f, 5f, 6f }, { 7f, 8f, 9f }
        } } });
        var w = DenseTensor<float>.OfValues(new float[1, 1, 2, 2] { { { { 1f, 1f }, { 1f, 1f } } } });
        var b = DenseTensor<float>.OfValues(new float[] { 0f });
        graph.Inputs["x"] = x;
        graph.Inputs["w"] = w;
        graph.Inputs["b"] = b;

        var node = new Node
        {
            Name = "conv",
            Op = OpType.Conv,
            Inputs = new[] { "x", "w", "b" },
            Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object>
            {
                ["auto_pad"] = "NOTSET",
                ["pads"] = new[] { 1, 1, 1, 1 },
                ["strides"] = new[] { 1, 1 },
                ["kernel_shape"] = new[] { 2, 2 }
            }
        };

        var result = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Success, result.Status);
        var y = (Tensor<float>)result.Outputs[0];
        var expected = Tensor<float>.Conv2D(x, w, 1, MathOps.PadType.Value, padvalue: 1, bias: (Tensor<float>)b, kernelshape: new[] { 2, 2 }, strides: new[] { 1, 1 }, dilations: null);
        Assert.Equal(expected, y);
    }

    [Fact]
    public void MaxPool_UsesKernelStridesAttributes()
    {
        var graph = CreateGraph(13);
        var x = DenseTensor<float>.OfValues(new float[1, 1, 4, 4] { { {
            {12.0f, 20.0f, 30.0f, 0.0f  }, { 8.0f, 12.0f, 2.0f, 0.0f }, { 34.0f, 70.0f, 37.0f, 4.0f }, { 112.0f, 100.0f, 25.0f, 12.0f } } } });
        graph.Inputs["x"] = x;

        var node = new Node
        {
            Name = "maxpool",
            Op = OpType.MaxPool,
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object>
            {
                ["auto_pad"] = "VALID",
                ["kernel_shape"] = new[] { 2, 2 },
                ["strides"] = new[] { 2, 2 }
            }
        };

        var result = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Success, result.Status);
        var y = (Tensor<float>)result.Outputs[0];
        var expected = Tensor<float>.MaxPool2D(x, new[] { 2, 2 }, MathOps.PadType.Valid, strides: new[] { 2, 2 }, padvalue: null, dilations: null, ceilMode: false);
        Assert.Equal(expected, y);
    }

    [Fact]
    public void Unsqueeze_UsesAxesAttribute_ForOpset12()
    {
        var graph = CreateGraph(opsetVersion: 12);
        graph.Inputs["x"] = DenseTensor<int>.OfValues(new int[] { 7, 8 });

        var node = new Node
        {
            Name = "unsqueeze",
            Op = OpType.Unsqueeze,
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object> { ["axes"] = new[] { 0 } }
        };

        var result = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Success, result.Status);
        var y = (Tensor<int>)result.Outputs[0];
        Assert.Equal(new[] { 1, 2 }, y.Dimensions.ToArray());
        Assert.Equal(7, y[0, 0]);
    }

    [Fact]
    public void ReduceMean_UsesAxesAttribute_ForOpset12()
    {
        var graph = CreateGraph(opsetVersion: 12);
        graph.Inputs["x"] = DenseTensor<float>.OfValues(new float[2, 2] { { 1f, 2f }, { 3f, 4f } });

        var node = new Node
        {
            Name = "reducesum",
            Op = OpType.ReduceMean,
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object>
            {
                ["axes"] = new[] { 1 },
                ["keepdims"] = 0
            }
        };

        var result = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Success, result.Status);
        var y = (Tensor<float>)result.Outputs[0];
        Assert.Equal(new[] { 2 }, y.Dimensions.ToArray());
        Assert.Equal(1.5f, y[0], 5);
    }

    [Fact]
    public void ReduceMean_UsesAxesAttribute_ForOpset14()
    {
        var graph = CreateGraph(opsetVersion: 14);
        graph.Inputs["x"] = DenseTensor<float>.OfValues(new float[2, 2] { { 1f, 2f }, { 3f, 4f } });

        var node = new Node
        {
            Name = "reducemean",
            Op = OpType.ReduceMean,
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object>
            {
                ["axes"] = new[] { 1 },
                ["keepdims"] = 1
            }
        };

        var result = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Success, result.Status);
        var y = (Tensor<float>)result.Outputs[0];
        Assert.Equal(new[] { 2, 1 }, y.Dimensions.ToArray());
        Assert.Equal(1.5f, y[0, 0], 5);
        Assert.Equal(3.5f, y[1, 0], 5);
    }

    [Fact]
    public void ReduceMax_UsesAxesAttribute_ForOpset14()
    {
        var graph = CreateGraph(opsetVersion: 14);
        graph.Inputs["x"] = DenseTensor<float>.OfValues(new float[2, 2] { { 1f, 2f }, { 3f, 4f } });

        var node = new Node
        {
            Name = "reducemax",
            Op = OpType.ReduceMax,
            Inputs = new[] { "x" },
            Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object>
            {
                ["axes"] = new[] { 0 },
                ["keepdims"] = 0
            }
        };

        var result = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Success, result.Status);
        var y = (Tensor<float>)result.Outputs[0];
        Assert.Equal(new[] { 2 }, y.Dimensions.ToArray());
        Assert.Equal(3f, y[0], 5);
        Assert.Equal(4f, y[1], 5);
    }
}
