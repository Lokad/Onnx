using System;
using System.Collections.Generic;
using Onnx;

namespace Lokad.Onnx.Backend.Tests;

// If/subgraph coverage (PLAN.md Milestone 3). A hand-built model exercises
// branch selection, outer captures (inputs and initializers),
// branch-local initializers, failure propagation with branch context, and
// Graph-attribute decoding; the tracked segmentation case proves the real
// single-If graph with multi-node branches.
public class IfBranchTests
{
    static ComputationalGraph IfModel()
    {
        return IfModel("Mul");
    }

    static ComputationalGraph IfModel(string elseOp)
    {
        var model = new OnnxModel { Name = "if-test", Opset = new Dictionary<string, int> { [""] = 17 } };
        model.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { 2 } });
        model.Inputs.Add(new OnnxValueInfo { Name = "flag", ElementType = TensorElementType.Bool, Dims = Array.Empty<int>() });
        model.Outputs.Add(new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Float, Dims = new[] { 2 } });
        model.Initializers.Add(new OnnxTensor { Name = "two", ElementType = TensorElementType.Float, Dims = new[] { 1 }, Data = new float[] { 2f } });
        var then = new OnnxSubgraph { Name = "then" };
        then.Outputs.Add(new OnnxValueInfo { Name = "t", ElementType = TensorElementType.Float, Dims = new[] { 2 } });
        then.Initializers.Add(new OnnxTensor { Name = "tone", ElementType = TensorElementType.Float, Dims = new[] { 2 }, Data = new float[] { 10f, 20f } });
        then.Nodes.Add(new OnnxNode { Name = "tadd", OpType = "Add", Inputs = new[] { "x", "tone" }, Outputs = new[] { "t" } });
        var els = new OnnxSubgraph { Name = "else" };
        els.Outputs.Add(new OnnxValueInfo { Name = "e", ElementType = TensorElementType.Float, Dims = new[] { 2 } });
        els.Nodes.Add(new OnnxNode { Name = "emul", OpType = elseOp, Inputs = new[] { "x", "two" }, Outputs = new[] { "e" } });
        model.Nodes.Add(new OnnxNode
        {
            Name = "if",
            OpType = "If",
            Inputs = new[] { "flag" },
            Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object> { ["then_branch"] = then, ["else_branch"] = els },
        });
        return Model.Load(model);
    }

    static Dictionary<string, ITensor> Feeds(float[] x, bool flag)
    {
        return new Dictionary<string, ITensor>
        {
            ["x"] = new DenseTensor<float>(x, new[] { 2 }),
            ["flag"] = new DenseTensor<bool>(new bool[] { flag }, new int[0]),
        };
    }

    [Fact]
    public void ThenBranch_SeesCapturesAndLocalInitializers()
    {
        var graph = IfModel();
        Assert.True(graph.Execute(Feeds(new float[] { 1f, 2f }, true), true));
        Assert.Equal(new float[] { 11f, 22f }, ((Tensor<float>)graph.Outputs["y"]).ToArray());
    }

    [Fact]
    public void ElseBranch_SeesOuterInitializer()
    {
        var graph = IfModel();
        Assert.True(graph.Execute(Feeds(new float[] { 1f, 2f }, false), true));
        Assert.Equal(new float[] { 2f, 4f }, ((Tensor<float>)graph.Outputs["y"]).ToArray());
    }

    [Fact]
    public void BranchFailure_NamesBranchAndNode()
    {
        var graph = IfModel("Nope");
        Assert.False(graph.Execute(Feeds(new float[] { 1f, 2f }, false), true));
        Assert.Contains("else_branch", graph.LastErrorMessage ?? "");
        Assert.Contains("emul", graph.LastErrorMessage ?? "");
    }

    [Fact]
    public void MissingBranchAttribute_FailsCleanly()
    {
        var graph = new ComputationalGraph
        {
            Opset = new Dictionary<string, int> { [""] = 17 },
            Metadata = new Dictionary<string, object> { ["Name"] = "test" },
        };
        var node = new Node
        {
            Name = "if",
            Op = OpType.If,
            OpTypeName = "If",
            Domain = "",
            OpsetVersion = 17,
            Inputs = new[] { "flag" },
            Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object>(),
        };
        graph.Inputs["flag"] = new DenseTensor<bool>(new bool[] { false }, new int[0]);
        var r = node.Execute(graph, ExecutionProvider.CPU, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("else_branch", r.Message ?? "");
    }

    [Fact]
    public void GraphAttribute_DecodesToSubgraphDto()
    {
        var inner = new NodeProto { Name = "c", OpType = "Conv" };
        inner.Input.Add("a");
        inner.Input.Add("w");
        inner.Output.Add("o");
        var gp = new GraphProto { Name = "then" };
        gp.Node.Add(inner);
        var attr = new AttributeProto { Name = "then_branch", Type = AttributeProto.Types.AttributeType.Graph };
        attr.G = gp;
        var node = new NodeProto { Name = "if", OpType = "If" };
        node.Attribute.Add(attr);
        var dto = node.ToNodeDto(null);
        var sg = Assert.IsType<OnnxSubgraph>(dto.Attributes["then_branch"]);
        Assert.Equal("then", sg.Name);
        Assert.Single(sg.Nodes);
        Assert.Equal("Conv", sg.Nodes[0].OpType);
        Assert.Equal(new[] { "a", "w" }, sg.Nodes[0].Inputs);
    }

    static ComputationalGraph SiblingLocalModel()
    {
        var model = new OnnxModel { Name = "if-sibling", Opset = new Dictionary<string, int> { [""] = 17 } };
        model.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { 2 } });
        model.Outputs.Add(new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Float, Dims = new[] { 2 } });
        model.Initializers.Add(new OnnxTensor { Name = "cond", ElementType = TensorElementType.Bool, Dims = Array.Empty<int>(), Data = new bool[] { true } });
        OnnxSubgraph Branch(string name, float amount)
        {
            var g = new OnnxSubgraph { Name = name };
            g.Outputs.Add(new OnnxValueInfo { Name = name, ElementType = TensorElementType.Float, Dims = new[] { 2 } });
            g.Initializers.Add(new OnnxTensor { Name = "local_weight", ElementType = TensorElementType.Float, Dims = new[] { 2 }, Data = new float[] { amount, amount } });
            g.Nodes.Add(new OnnxNode { Name = name + "_add", OpType = "Add", Inputs = new[] { "x", "local_weight" }, Outputs = new[] { name } });
            return g;
        }
        model.Nodes.Add(new OnnxNode
        {
            Name = "first",
            OpType = "If",
            Inputs = new[] { "cond" },
            Outputs = new[] { "unused" },
            Attributes = new Dictionary<string, object> { ["then_branch"] = Branch("a", 2f), ["else_branch"] = Branch("aa", 2f) },
        });
        model.Nodes.Add(new OnnxNode
        {
            Name = "second",
            OpType = "If",
            Inputs = new[] { "cond" },
            Outputs = new[] { "y" },
            Attributes = new Dictionary<string, object> { ["then_branch"] = Branch("b", 3f), ["else_branch"] = Branch("bb", 3f) },
        });
        return Model.Load(model);
    }

    [Fact]
    public void SiblingBranchLocals_DoNotLeak()
    {
        var graph = SiblingLocalModel();
        Assert.True(graph.Execute(new Dictionary<string, ITensor> { ["x"] = new DenseTensor<float>(new float[] { 1f, 2f }, new[] { 2 }) }, true));
        Assert.Equal(new float[] { 4f, 5f }, ((Tensor<float>)graph.Outputs["y"]).ToArray());
    }

    [Fact]
    public void RepeatedConditionFlips_UseCurrentBranch()
    {
        var graph = IfModel();
        Assert.True(graph.Execute(Feeds(new float[] { 1f, 2f }, true), true));
        Assert.Equal(new float[] { 11f, 22f }, ((Tensor<float>)graph.Outputs["y"]).ToArray());
        Assert.True(graph.Execute(Feeds(new float[] { 1f, 2f }, false), true));
        Assert.Equal(new float[] { 2f, 4f }, ((Tensor<float>)graph.Outputs["y"]).ToArray());
        Assert.True(graph.Execute(Feeds(new float[] { 1f, 2f }, true), true));
        Assert.Equal(new float[] { 11f, 22f }, ((Tensor<float>)graph.Outputs["y"]).ToArray());
    }

}
