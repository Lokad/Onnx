using System;
using System.Collections.Generic;
using Onnx;

namespace Lokad.Onnx.Backend.Tests;

// Branch-plan constant folding (M3): literal Constants inside If branches
// become branch initializers and constant-only branch chains evaluate once
// at preparation, with outer captures visible and user-fed names protected.
public class ConstantBranchFoldingTests
{
    static OnnxSubgraph Branch(string name, string output, List<OnnxNode> nodes)
    {
        return Branch(name, output, nodes, new List<OnnxTensor>());
    }

    static OnnxSubgraph Branch(string name, string output, List<OnnxNode> nodes, List<OnnxTensor> inits)
    {
        var g = new OnnxSubgraph { Name = name };
        g.Outputs.Add(new OnnxValueInfo { Name = output, ElementType = TensorElementType.Float, Dims = new[] { 2 } });
        foreach (var t in inits)
            g.Initializers.Add(t);
        foreach (var n in nodes)
            g.Nodes.Add(n);
        return g;
    }

    static OnnxNode Const(string name, string output, float[] values)
    {
        return new OnnxNode
        {
            Name = name,
            OpType = "Constant",
            Inputs = Array.Empty<string>(),
            Outputs = new[] { output },
            Attributes = new Dictionary<string, object>
            {
                ["value"] = new DenseTensor<float>(values, new[] { values.Length }),
            },
        };
    }

    static OnnxNode Bin(string name, string op, string left, string right, string output)
    {
        return new OnnxNode { Name = name, OpType = op, Inputs = new[] { left, right }, Outputs = new[] { output } };
    }

    static ComputationalGraph IfModel()
    {
        var model = new OnnxModel { Name = "branch-fold", Opset = new Dictionary<string, int> { [""] = 17 } };
        model.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { 2 } });
        model.Inputs.Add(new OnnxValueInfo { Name = "flag", ElementType = TensorElementType.Bool, Dims = Array.Empty<int>() });
        model.Outputs.Add(new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Float, Dims = new[] { 2 } });
        model.Initializers.Add(new OnnxTensor { Name = "outer", ElementType = TensorElementType.Float, Dims = new[] { 2 }, Data = new float[] { 5f, 7f } });
        var then = Branch("then", "t", new List<OnnxNode>
        {
            Const("tc", "tc", new float[] { 1f, 2f }),
            Bin("tadd", "Add", "tc", "outer", "tadd"),
            Bin("t", "Add", "tadd", "x", "t"),
        });
        var els = Branch("else", "e", new List<OnnxNode>
        {
            Const("ec", "ec", new float[] { 3f, 4f }),
            Bin("em", "Mul", "x", "outer", "em"),
            Bin("e", "Add", "em", "ec", "e"),
        });
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

    static ComputationalGraph BranchOf(ComputationalGraph graph, string attr)
    {
        Assert.Single(graph.Nodes);
        return Assert.IsType<ComputationalGraph>(graph.Nodes[0].Attributes![attr]);
    }

    static Dictionary<string, ITensor> Feeds(float[] x, bool flag)
    {
        return new Dictionary<string, ITensor>
        {
            ["x"] = new DenseTensor<float>(x, new[] { 2 }),
            ["flag"] = new DenseTensor<bool>(new bool[] { flag }, new int[0]),
        };
    }

    static float[] Run(ComputationalGraph graph, float[] x, bool flag)
    {
        Assert.True(graph.Execute(Feeds(x, flag), true), graph.LastErrorMessage);
        return ((Tensor<float>)graph.Outputs["y"]).ToArray();
    }

    [Fact]
    public void BranchFolds_LiteralsAndCaptureChains()
    {
        var graph = IfModel();
        var then = BranchOf(graph, "then_branch");
        var els = BranchOf(graph, "else_branch");
        Assert.Single(then.Nodes);
        Assert.Equal(2, els.Nodes.Count);
        Assert.True(then.Initializers.ContainsKey("tc"), "folded branch literal is missing.");
        Assert.True(then.Initializers.ContainsKey("tadd"), "folded capture chain is missing.");
        Assert.True(els.Initializers.ContainsKey("ec"), "folded branch literal is missing.");
        Assert.False(els.Initializers.ContainsKey("em"), "input-dependent chain must not fold.");
        Assert.False(els.Initializers.ContainsKey("e"), "branch outputs must not fold.");
        Assert.Equal(new float[] { 6f, 9f }, ((Tensor<float>)then.Initializers["tadd"]).ToArray());
    }

    [Fact]
    public void BranchFolds_ExecuteBothBranches()
    {
        var graph = IfModel();
        Assert.Equal(new float[] { 7f, 10f }, Run(graph, new float[] { 1f, 1f }, true));
        Assert.Equal(new float[] { 8f, 11f }, Run(graph, new float[] { 1f, 1f }, false));
        Assert.Equal(new float[] { 8f, 12f }, Run(graph, new float[] { 2f, 3f }, true));
        Assert.Equal(new float[] { 13f, 25f }, Run(graph, new float[] { 2f, 3f }, false));
    }

    [Fact]
    public void BranchFolds_RebuildsOnSourceReplace()
    {
        var graph = IfModel();
        Assert.Equal(new float[] { 6f, 9f }, Run(graph, new float[] { 0f, 0f }, true));
        graph.Initializers["outer"] = new DenseTensor<float>(new float[] { 10f, 20f }, new[] { 2 });
        graph.RefreshLifetimeAnalysis();
        var then = BranchOf(graph, "then_branch");
        Assert.Single(then.Nodes);
        Assert.Equal(new float[] { 11f, 22f }, ((Tensor<float>)then.Initializers["tadd"]).ToArray());
        Assert.Equal(new float[] { 11f, 22f }, Run(graph, new float[] { 0f, 0f }, true));
        Assert.Equal(new float[] { 3f, 4f }, Run(graph, new float[] { 0f, 0f }, false));
    }

    [SkippableFact]
    public void RealSegmentation_BranchConstantsFold()
    {
        var graph = ModelFixture.LoadRequiredModel("PyannoteSegmentation", "models", "speaker-diarization-community-1", "onnx", "segmentation", "model.onnx");
        graph.RefreshLifetimeAnalysis();
        int computed = 0;
        int remaining = 0;
        foreach (var node in graph.Nodes)
        {
            if (node.Op != OpType.If || node.Attributes is null) continue;
            foreach (var value in node.Attributes.Values)
            {
                if (value is not ComputationalGraph branch) continue;
                computed += branch.FoldedComputationCount;
                foreach (var bn in branch.Nodes)
                {
                    if (bn.Op == OpType.Constant && bn.Outputs is not null && bn.Outputs.Length == 1 && !branch.Outputs.ContainsKey(bn.Outputs[0]))
                        remaining++;
                }
            }
        }
        Assert.Equal(0, remaining);
        Assert.Equal(6, computed);
    }
}
