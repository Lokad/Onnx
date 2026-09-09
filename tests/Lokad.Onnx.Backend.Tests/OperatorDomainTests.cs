using System.Collections.Generic;
using System.Linq;

namespace Lokad.Onnx.Backend.Tests;

public class OperatorDomainTests
{
    static OnnxValueInfo IO(string name) =>
        new OnnxValueInfo { Name = name, ElementType = TensorElementType.Float, Dims = new[] { 2 } };

    static OnnxModel AddModel(string domain, string opType)
    {
        var mp = new OnnxModel { Name = "domain-test" };
        mp.Opset[""] = 11;
        mp.Inputs.Add(IO("x")); mp.Inputs.Add(IO("y"));
        mp.Outputs.Add(IO("z"));
        mp.Nodes.Add(new OnnxNode
        {
            Name = "n1", OpType = opType, Domain = domain,
            Inputs = new[] { "x", "y" }, Outputs = new[] { "z" },
            Attributes = new Dictionary<string, object>(),
        });
        return mp;
    }

    [Fact]
    public void StandardAdd_Executes()
    {
        var graph = Model.Load(AddModel("", "Add"))!;
        Assert.Equal(OpType.Add, graph.Nodes[0].Op);
        Assert.False(graph.Nodes[0].IsFused);
        var inputs = new Dictionary<string, ITensor>
        {
            { "x", DenseTensor<float>.OfValues(new float[] { 1f, 2f }) },
            { "y", DenseTensor<float>.OfValues(new float[] { 10f, 20f }) },
        };
        Assert.True(graph.Execute(inputs, true));
        Assert.Equal(new float[] { 11f, 22f }, ((Tensor<float>)graph.Outputs["z"]).ToArray());
        Assert.True(CPUExecutionProvider.SupportsNode(graph.Nodes[0]));
    }

    [Fact]
    public void CustomDomainAdd_LoadsButDoesNotDispatch()
    {
        var graph = Model.Load(AddModel("my.domain", "Add"))!;
        Assert.Equal("my.domain", graph.Nodes[0].Domain);
        Assert.Equal(OpType.Add, graph.Nodes[0].Op);
        Assert.False(CPUExecutionProvider.SupportsNode(graph.Nodes[0]));
        var inputs = new Dictionary<string, ITensor>
        {
            { "x", DenseTensor<float>.OfValues(new float[] { 1f, 2f }) },
            { "y", DenseTensor<float>.OfValues(new float[] { 10f, 20f }) },
        };
        Assert.False(graph.Execute(inputs, true));
        Assert.Contains("my.domain", graph.LastErrorMessage ?? "");
    }

    [Fact]
    public void UnknownOp_LoadsInspectablyAndFailsWithDiagnostic()
    {
        var graph = Model.Load(AddModel("", "FancyNewOp"))!;
        Assert.Equal(OpType.Unknown, graph.Nodes[0].Op);
        Assert.Equal("FancyNewOp", graph.Nodes[0].OpTypeName);
        Assert.False(CPUExecutionProvider.SupportsNode(graph.Nodes[0]));
        var inputs = new Dictionary<string, ITensor>
        {
            { "x", DenseTensor<float>.OfValues(new float[] { 1f, 2f }) },
            { "y", DenseTensor<float>.OfValues(new float[] { 10f, 20f }) },
        };
        Assert.False(graph.Execute(inputs, true));
        Assert.Contains("FancyNewOp", graph.LastErrorMessage ?? "");
    }

    [Fact]
    public void FusedLayerNorm_IsMarkedFused_WhileImportedIsNot()
    {
        var mp = new OnnxModel { Name = "tiny-ln" };
        mp.Opset[""] = 11;
        mp.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "z", ElementType = TensorElementType.Float, Dims = new[] { 2, 4 } });
        mp.Initializers.Add(new OnnxTensor { Name = "gamma", ElementType = TensorElementType.Float, Dims = new[] { 4 }, Data = new[] { 1f, 1f, 1f, 1f } });
        mp.Initializers.Add(new OnnxTensor { Name = "beta", ElementType = TensorElementType.Float, Dims = new[] { 4 }, Data = new[] { 0f, 0f, 0f, 0f } });
        mp.Initializers.Add(new OnnxTensor { Name = "two", ElementType = TensorElementType.Float, Dims = new int[0], Data = new[] { 2f } });
        var eps = Model.ToTensor(new OnnxTensor { ElementType = TensorElementType.Float, Dims = new int[0], Data = new[] { 1e-5f } });
        Dictionary<string, object> Attrs(params (string Name, object Value)[] a)
        {
            var d = new Dictionary<string, object>();
            foreach (var kv in a) d[kv.Name] = kv.Value;
            return d;
        }
        OnnxNode Nod(string op, string[] i, string[] o, Dictionary<string, object>? a) =>
            new OnnxNode { OpType = op, Inputs = i, Outputs = o, Attributes = a ?? new Dictionary<string, object>() };
        mp.Nodes.Add(Nod("ReduceMean", new[] { "x" }, new[] { "m" }, Attrs(("axes", new long[] { -1 }), ("keepdims", 1L))));
        mp.Nodes.Add(Nod("Sub", new[] { "x", "m" }, new[] { "d" }, null));
        mp.Nodes.Add(Nod("Pow", new[] { "d", "two" }, new[] { "s" }, null));
        mp.Nodes.Add(Nod("ReduceMean", new[] { "s" }, new[] { "v" }, Attrs(("axes", new long[] { -1 }), ("keepdims", 1L))));
        mp.Nodes.Add(Nod("Constant", new string[0], new[] { "e" }, Attrs(("value", eps))));
        mp.Nodes.Add(Nod("Add", new[] { "v", "e" }, new[] { "ve" }, null));
        mp.Nodes.Add(Nod("Sqrt", new[] { "ve" }, new[] { "sd" }, null));
        mp.Nodes.Add(Nod("Div", new[] { "d", "sd" }, new[] { "n" }, null));
        mp.Nodes.Add(Nod("Mul", new[] { "n", "gamma" }, new[] { "g" }, null));
        mp.Nodes.Add(Nod("Add", new[] { "g", "beta" }, new[] { "z" }, null));
        var graph = Model.Load(mp)!;
        Assert.Single(graph.Nodes);
        Assert.Equal(OpType.LayerNormalization, graph.Nodes[0].Op);
        Assert.True(graph.Nodes[0].IsFused);
        Assert.True(CPUExecutionProvider.SupportsNode(graph.Nodes[0]));
    }

    static OnnxModel UnsqueezeModel(string opsetDomain, int opset, string nodeDomain, bool inputForm)
    {
        var mp = new OnnxModel { Name = "unsqueeze-domain" };
        mp.Opset[opsetDomain] = opset;
        mp.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { 2, 3 } });
        mp.Outputs.Add(new OnnxValueInfo
        {
            Name = "z", ElementType = TensorElementType.Float,
            Dims = inputForm ? new[] { 2, 1, 3 } : new[] { 1, 2, 3 },
        });
        var node = new OnnxNode
        {
            Name = "u", OpType = "Unsqueeze", Domain = nodeDomain,
            Outputs = new[] { "z" }, Attributes = new Dictionary<string, object>(),
        };
        if (inputForm)
        {
            node.Inputs = new[] { "x", "axes" };
            mp.Initializers.Add(new OnnxTensor { Name = "axes", ElementType = TensorElementType.Int64, Dims = new[] { 1 }, Data = new long[] { 1L } });
        }
        else
        {
            node.Inputs = new[] { "x" };
            node.Attributes["axes"] = new long[] { 0L };
        }
        mp.Nodes.Add(node);
        return mp;
    }

    static float[] UnsqueezeInputValues() => new float[] { 1f, 2f, 3f, 4f, 5f, 6f };

    [Fact]
    public void AiOnnxDomain_Unsqueeze_Executes()
    {
        var graph = Model.Load(UnsqueezeModel("ai.onnx", 13, "ai.onnx", true))!;
        Assert.Equal(13, graph.Nodes[0].OpsetVersion);
        Assert.True(CPUExecutionProvider.SupportsNode(graph.Nodes[0]));
        var inputs = new Dictionary<string, ITensor>
        {
            { "x", DenseTensor<float>.OfValues(new float[2, 3] { { 1f, 2f, 3f }, { 4f, 5f, 6f } }) },
        };
        Assert.True(graph.Execute(inputs, true), graph.LastErrorMessage);
        var z = (Tensor<float>)graph.Outputs["z"];
        Assert.Equal(new int[] { 2, 1, 3 }, z.Dimensions.ToArray());
        Assert.Equal(UnsqueezeInputValues(), z.ToArray());
    }

    [Fact]
    public void EquivalentDomains_BehaveIdentically()
    {
        var std = Model.Load(UnsqueezeModel("", 13, "", true))!;
        var alt = Model.Load(UnsqueezeModel("ai.onnx", 13, "ai.onnx", true))!;
        var inputs = new Dictionary<string, ITensor>
        {
            { "x", DenseTensor<float>.OfValues(new float[2, 3] { { 1f, 2f, 3f }, { 4f, 5f, 6f } }) },
        };
        Assert.True(std.Execute(inputs, true), std.LastErrorMessage);
        Assert.True(alt.Execute(inputs, true), alt.LastErrorMessage);
        Assert.Equal(
            ((Tensor<float>)std.Outputs["z"]).ToArray(),
            ((Tensor<float>)alt.Outputs["z"]).ToArray());
    }

    [Fact]
    public void Unsqueeze_OldForm_Executes()
    {
        var graph = Model.Load(UnsqueezeModel("", 11, "", false))!;
        var inputs = new Dictionary<string, ITensor>
        {
            { "x", DenseTensor<float>.OfValues(new float[2, 3] { { 1f, 2f, 3f }, { 4f, 5f, 6f } }) },
        };
        Assert.True(graph.Execute(inputs, true), graph.LastErrorMessage);
        var z = (Tensor<float>)graph.Outputs["z"];
        Assert.Equal(new int[] { 1, 2, 3 }, z.Dimensions.ToArray());
        Assert.Equal(UnsqueezeInputValues(), z.ToArray());
    }

    [Fact]
    public void MixedDomains_ResolvePerNode()
    {
        var mp = new OnnxModel { Name = "mixed-domain" };
        mp.Opset[""] = 11;
        mp.Opset["ai.onnx"] = 13;
        mp.Inputs.Add(new OnnxValueInfo { Name = "x1", ElementType = TensorElementType.Float, Dims = new[] { 2, 3 } });
        mp.Inputs.Add(new OnnxValueInfo { Name = "x2", ElementType = TensorElementType.Float, Dims = new[] { 2, 3 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y1", ElementType = TensorElementType.Float, Dims = new[] { 1, 2, 3 } });
        mp.Outputs.Add(new OnnxValueInfo { Name = "y2", ElementType = TensorElementType.Float, Dims = new[] { 2, 3, 1 } });
        mp.Nodes.Add(new OnnxNode
        {
            Name = "u1", OpType = "Unsqueeze", Domain = "",
            Inputs = new[] { "x1" }, Outputs = new[] { "y1" },
            Attributes = new Dictionary<string, object> { { "axes", new long[] { 0L } } },
        });
        mp.Nodes.Add(new OnnxNode
        {
            Name = "u2", OpType = "Unsqueeze", Domain = "ai.onnx",
            Inputs = new[] { "x2", "axes" }, Outputs = new[] { "y2" },
            Attributes = new Dictionary<string, object>(),
        });
        mp.Initializers.Add(new OnnxTensor { Name = "axes", ElementType = TensorElementType.Int64, Dims = new[] { 1 }, Data = new long[] { 2L } });
        var graph = Model.Load(mp)!;
        Assert.Equal(11, graph.Nodes[0].OpsetVersion);
        Assert.Equal(13, graph.Nodes[1].OpsetVersion);
        var inputs = new Dictionary<string, ITensor>
        {
            { "x1", DenseTensor<float>.OfValues(new float[2, 3] { { 1f, 2f, 3f }, { 4f, 5f, 6f } }) },
            { "x2", DenseTensor<float>.OfValues(new float[2, 3] { { 1f, 2f, 3f }, { 4f, 5f, 6f } }) },
        };
        Assert.True(graph.Execute(inputs, true), graph.LastErrorMessage);
        Assert.Equal(new int[] { 1, 2, 3 }, ((Tensor<float>)graph.Outputs["y1"]).Dimensions.ToArray());
        Assert.Equal(new int[] { 2, 3, 1 }, ((Tensor<float>)graph.Outputs["y2"]).Dimensions.ToArray());
        Assert.Equal(UnsqueezeInputValues(), ((Tensor<float>)graph.Outputs["y1"]).ToArray());
        Assert.Equal(UnsqueezeInputValues(), ((Tensor<float>)graph.Outputs["y2"]).ToArray());
    }

    [Fact]
    public void SupportPredicate_AgreesWithDispatch()
    {
        var std = Model.Load(AddModel("", "Add"))!.Nodes[0];
        var custom = Model.Load(AddModel("my.domain", "Add"))!.Nodes[0];
        var unknown = Model.Load(AddModel("", "FancyNewOp"))!.Nodes[0];
        Assert.True(CPUExecutionProvider.SupportsOp(OpType.Add));
        Assert.True(CPUExecutionProvider.SupportsNode(std));
        Assert.False(CPUExecutionProvider.SupportsNode(custom));
        Assert.False(CPUExecutionProvider.SupportsNode(unknown));
        Assert.False(CPUExecutionProvider.SupportsOp(OpType.Unknown));
    }
}
