using Google.Protobuf;
using Onnx;
using Lokad.Onnx.Optimization;

namespace Lokad.Onnx.Backend.Tests;

public class IfControlFlowTests
{
    static OnnxValueInfo Desc(string name, TensorElementType type, params int[] dims) =>
        new() { Name = name, ElementType = type, Dims = dims };
    static OnnxNode Op(string name, string op, string[] inputs, params string[] outputs) =>
        new() { Name = name, OpType = op, Inputs = inputs, Outputs = outputs };
    static OnnxSubgraph Direct(string name) => new() { Outputs = { Desc(name, TensorElementType.Float, 2) } };
    static OnnxNode If(string name, string cond, OnnxSubgraph then, OnnxSubgraph otherwise, params string[] outputs) =>
        new() { Name = name, OpType = "If", Inputs = new[] { cond }, Outputs = outputs,
            Attributes = new() { ["then_branch"] = then, ["else_branch"] = otherwise } };
    static OnnxModel Basic(OnnxSubgraph then, OnnxSubgraph otherwise) => new()
    {
        Opset = { [""] = 14 }, Inputs = { Desc("cond", TensorElementType.Bool), Desc("x", TensorElementType.Float, 2) },
        Outputs = { Desc("y", TensorElementType.Float, 2) }, Nodes = { If("choose", "cond", then, otherwise, "y") },
    };
    static Dictionary<string, ITensor> Feed(bool cond, params float[] x) => new()
    {
        ["cond"] = new DenseTensor<bool>(new[] { cond }, Array.Empty<int>()),
        ["x"] = DenseTensor<float>.OfValues(x),
    };
    static float[] Values(ComputationalGraph graph, string name) => ((Tensor<float>)graph.Outputs[name]!).ToArray();
    static OnnxSubgraph Negate() => new()
    {
        Outputs = { Desc("neg", TensorElementType.Float, 2) }, Nodes = { Op("negate", "Neg", new[] { "x" }, "neg") },
    };

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void CapturesAreIsolated_AndHeldOutputsSurviveResetFailureAndLaterCalls(bool condition)
    {
        var graph = Model.Load(Basic(Direct("x"), Negate()));
        var branch = (ComputationalGraph)graph.Nodes[0].Attributes!["then_branch"];
        var beforeInputs = branch.Inputs.ToArray();
        var beforeOutputs = branch.Outputs.ToArray();
        var feed = Feed(condition, -3, 8);
        feed["x"].Name = "original";
        Assert.True(graph.Execute(feed, true), graph.LastErrorMessage);
        var held = (Tensor<float>)graph.Outputs["y"]!;
        float[] expected = condition ? new[] { -3f, 8f } : new[] { 3f, -8f };
        Assert.Equal(expected, held.ToArray());
        Assert.Equal("original", feed["x"].Name);
        for (int i = 0; i < 6; i++)
        {
            Assert.True(graph.Execute(Feed(i % 2 == 0, i, i + 1), true), graph.LastErrorMessage);
            graph.Reset();
            Assert.Equal(expected, held.ToArray());
        }
        Assert.False(graph.Execute(new Dictionary<string, ITensor>(), true));
        Assert.Equal(expected, held.ToArray());
        Assert.Equal(beforeInputs, branch.Inputs.ToArray());
        Assert.Equal(beforeOutputs, branch.Outputs.ToArray());
        Assert.Empty(branch.LastSubgraphExecutions);
        Assert.Null(branch.LastProfile);
    }

    [Fact]
    public void DirectCaptureCanAlsoBeReadByBranchNode()
    {
        var then = Direct("x");
        then.Outputs.Add(Desc("neg", TensorElementType.Float, 2));
        then.Nodes.Add(Op("negate", "Neg", new[] { "x" }, "neg"));
        var model = Basic(then, then);
        model.Nodes[0].Outputs = new[] { "y", "z" };
        model.Outputs.Add(Desc("z", TensorElementType.Float, 2));
        var graph = Model.Load(model);
        Assert.True(graph.Execute(Feed(true, -3, 8), true), graph.LastErrorMessage);
        Assert.Equal(new[] { -3f, 8f }, Values(graph, "y"));
        Assert.Equal(new[] { 3f, -8f }, Values(graph, "z"));
    }

    [Fact]
    public void NestedCapturesRespectEachLocalScopeAndSiblingNames()
    {
        var local = new OnnxSubgraph
        {
            Initializers = { new OnnxTensor { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { 2 }, Data = new[] { 7f, 9f } } },
            Outputs = { Desc("nested", TensorElementType.Float, 2) },
            Nodes = { If("inner", "cond", Direct("x"), Negate(), "nested") },
        };
        var model = Basic(local, Negate());
        var graph = Model.Load(model);
        Assert.Equal(new[] { "cond", "x" }, GraphCaptures.NodeInputs(graph.Nodes[0]));
        Assert.True(graph.Execute(Feed(true, -3, 8), true), graph.LastErrorMessage);
        Assert.Equal(new[] { 7f, 9f }, Values(graph, "y"));
        Assert.True(graph.Execute(Feed(false, -3, 8), true), graph.LastErrorMessage);
        Assert.Equal(new[] { 3f, -8f }, Values(graph, "y"));
    }

    [Theory]
    [InlineData("Add")]
    [InlineData("Conv")]
    public void FusionCannotEraseCapturedPreReluValue(string producer)
    {
        var model = Basic(Direct("pre"), Direct("pre"));
        int[] dims = producer == "Conv" ? new[] { 1, 1, 2 } : new[] { 2 };
        model.Inputs[1].Dims = dims;
        model.Outputs[0].Dims = dims;
        foreach (var branch in model.Nodes[0].Attributes.Values.Cast<OnnxSubgraph>()) branch.Outputs[0].Dims = dims;
        model.Initializers.Add(new OnnxTensor { Name = "weight", ElementType = TensorElementType.Float,
            Dims = producer == "Conv" ? new[] { 1, 1, 1 } : new[] { 2 }, Data = producer == "Conv" ? new[] { 1f } : new[] { 0f, 0f } });
        model.Nodes.Insert(0, Op("producer", producer, new[] { "x", "weight" }, "pre"));
        model.Nodes.Insert(1, Op("relu", "Relu", new[] { "pre" }, "post"));
        model.Outputs.Add(Desc("post", TensorElementType.Float, dims));
        var graph = Model.Load(model);
        Assert.DoesNotContain(graph.Nodes, n => n.Op is OpType.ConvRelu or OpType.AddRelu);
        var feed = Feed(true, -1, 8);
        feed["x"] = ((Tensor<float>)feed["x"]).Reshape(dims);
        Assert.True(graph.Execute(feed, true), graph.LastErrorMessage);
        Assert.Equal(new[] { -1f, 8f }, Values(graph, "y"));
        Assert.Equal(new[] { 0f, 8f }, Values(graph, "post"));
    }

    [Fact]
    public void SweepingAndConstantDeduplicationPreserveImplicitReads()
    {
        var model = Basic(Direct("second"), Direct("second"));
        foreach (string name in new[] { "first", "second" })
        {
            var node = Op(name, "Constant", Array.Empty<string>(), name);
            node.Attributes["value"] = DenseTensor<float>.OfValues(new[] { -1f, 8f });
            model.Nodes.Insert(0, node);
        }
        model.Nodes.Insert(2, Op("visible", "Identity", new[] { "first" }, "other"));
        model.Outputs.Add(Desc("other", TensorElementType.Float, 2));
        var graph = Model.Load(model);
        Assert.Contains(graph.Nodes, n => n.Outputs.Contains("second"));
        Assert.True(graph.Execute(Feed(true, 1, 2), true), graph.LastErrorMessage);
        Assert.Equal(new[] { -1f, 8f }, Values(graph, "y"));
    }

    [Fact]
    public void LastUseIncludesDeepCaptureAndOutputViewsRemainValid()
    {
        var inner = Direct("view");
        var outer = new OnnxSubgraph { Outputs = { Desc("nested", TensorElementType.Float, 2) },
            Nodes = { If("nested", "cond", inner, inner, "nested") } };
        var model = Basic(outer, outer);
        model.Initializers.Add(new OnnxTensor { Name = "shape", ElementType = TensorElementType.Int64, Dims = new[] { 1 }, Data = new long[] { 2 } });
        model.Nodes.Insert(0, Op("produce", "Neg", new[] { "x" }, "temp"));
        model.Nodes.Insert(1, Op("view", "Reshape", new[] { "temp", "shape" }, "view"));
        model.Nodes.Insert(2, Op("other", "Neg", new[] { "x" }, "other"));
        model.Outputs.Add(Desc("other", TensorElementType.Float, 2));
        var graph = Model.Load(model);
        Assert.Equal(graph.Nodes.Count - 1, graph.LastUseIndex["view"]);
        Assert.True(graph.Execute(Feed(true, -1, 8), true), graph.LastErrorMessage);
        var held = (Tensor<float>)graph.Outputs["y"]!;
        for (int i = 0; i < 6; i++) Assert.True(graph.Execute(Feed(false, i, -i), true), graph.LastErrorMessage);
        graph.Reset();
        Assert.Equal(new[] { 1f, -8f }, held.ToArray());
    }

    [Fact]
    public async Task ExplicitContextsSharePlansWithoutSharingCapturesOrResults()
    {
        var graph = Model.Load(Basic(Direct("x"), Negate()));
        var contexts = Enumerable.Range(0, 8).Select(_ => graph.CreateExecution(null)).ToArray();
        await Task.WhenAll(contexts.Select((context, i) => Task.Run(() =>
        {
            for (int k = 0; k < 10; k++)
            {
                Assert.True(context.Execute(Feed(i % 2 == 0, i, -i), true), context.LastErrorMessage);
                Assert.Equal(i % 2 == 0 ? new[] { (float)i, -i } : new[] { -(float)i, i }, Values(context, "y"));
            }
        })));
    }

    [Theory]
    [InlineData("missing-then")]
    [InlineData("missing-else")]
    [InlineData("output-count")]
    [InlineData("output-type")]
    [InlineData("unsupported-op")]
    [InlineData("unsupported-domain")]
    [InlineData("missing-capture")]
    [InlineData("late-producer")]
    [InlineData("non-bool")]
    [InlineData("many-bool")]
    public void MalformedBranchesFailCleanly(string issue)
    {
        var model = Basic(Direct("x"), Negate());
        switch (issue)
        {
            case "missing-then": model.Nodes[0].Attributes.Remove("then_branch"); break;
            case "missing-else": model.Nodes[0].Attributes.Remove("else_branch"); break;
            case "output-count": ((OnnxSubgraph)model.Nodes[0].Attributes["else_branch"]).Outputs.Clear(); break;
            case "output-type": ((OnnxSubgraph)model.Nodes[0].Attributes["else_branch"]).Outputs[0].ElementType = TensorElementType.Int64; break;
            case "unsupported-op": ((OnnxSubgraph)model.Nodes[0].Attributes["else_branch"]).Nodes[0].OpType = "Loop"; break;
            case "unsupported-domain": ((OnnxSubgraph)model.Nodes[0].Attributes["else_branch"]).Nodes[0].Domain = "untrusted"; break;
            case "missing-capture": model.Nodes[0].Attributes["else_branch"] = Direct("missing"); break;
            case "late-producer": model.Nodes[0].Attributes["else_branch"] = Direct("late"); model.Nodes.Add(Op("late", "Neg", new[] { "x" }, "late")); break;
            case "non-bool": model.Inputs[0].ElementType = TensorElementType.Float; break;
            case "many-bool": model.Inputs[0].Dims = new[] { 2 }; break;
        }
        var graph = Model.Load(model);
        var feed = Feed(true, -1, 8);
        if (issue == "non-bool") feed["cond"] = new DenseTensor<float>(new[] { 1f }, Array.Empty<int>());
        if (issue == "many-bool") feed["cond"] = DenseTensor<bool>.OfValues(new[] { true, false });
        Assert.False(graph.Execute(feed, true));
        Assert.NotNull(graph.LastErrorMessage);
        Assert.Empty(graph.Outputs);
    }

    [Fact]
    public void SingletonVectorConditionMatchesOrtContract()
    {
        var model = Basic(Direct("x"), Negate()); model.Inputs[0].Dims = new[] { 1 };
        var graph = Model.Load(model); var feed = Feed(true, -1, 8);
        feed["cond"] = DenseTensor<bool>.OfValues(new[] { true });
        Assert.True(graph.Execute(feed, true), graph.LastErrorMessage);
        Assert.Equal(new[] { -1f, 8f }, Values(graph, "y"));
    }

    [Fact]
    public void CyclicDtoAndExecutableAttributesAreRejected()
    {
        var loop = Direct("x"); loop.Nodes.Add(If("cycle", "cond", loop, loop, "z"));
        Assert.Throws<InvalidOperationException>(() => Model.Load(Basic(loop, loop)));
        var graph = Model.Load(Basic(Direct("x"), Negate()));
        graph.Nodes[0].Attributes!["then_branch"] = graph;
        Assert.Throws<InvalidOperationException>(() => graph.InvalidatePreparation());
        Assert.Throws<InvalidOperationException>(() => graph.Prepare());
    }

    [Fact]
    public void ReplacingGraphAttributesRebuildsCaptureLifetimeAnalysis()
    {
        var graph = Model.Load(Basic(Direct("x"), Negate()));
        Assert.True(graph.Execute(Feed(true, 3, 8), true));
        graph.Nodes[0].Attributes!["then_branch"] = graph.Nodes[0].Attributes!["else_branch"];
        Assert.True(graph.Execute(Feed(true, 3, 8), true), graph.LastErrorMessage);
        Assert.Equal(new[] { -3f, -8f }, Values(graph, "y"));
    }

    [Fact]
    public void NodeExecutionBindsImplicitCaptures()
    {
        var graph = Model.Load(Basic(Direct("x"), Negate()));
        Assert.True(graph.ExecuteNode(Feed(false, 3, 8), "choose", true), graph.LastErrorMessage);
        Assert.Equal(new[] { -3f, -8f }, Values(graph, "y"));
    }

    [Fact]
    public void ExplicitInvalidationRebuildsNestedPreparedTransposes()
    {
        var then = new OnnxSubgraph
        {
            Initializers = { new OnnxTensor { Name = "w", ElementType = TensorElementType.Float,
                Dims = new[] { 1, 2 }, Data = new[] { 3f, 8f } } },
            Outputs = { Desc("out", TensorElementType.Float, 2, 1) },
            Nodes = { Op("transpose", "Transpose", new[] { "w" }, "out") },
        };
        var model = Basic(then, then); model.Outputs[0].Dims = new[] { 2, 1 };
        var graph = Model.Load(model);
        Assert.True(graph.Execute(Feed(true, 0, 0), true), graph.LastErrorMessage);
        var saved = (Tensor<float>)graph.Outputs["y"]!;
        var branch = (ComputationalGraph)graph.Nodes[0].Attributes!["then_branch"];
        var weights = (Tensor<float>)branch.Initializers["w"];
        weights.SetValue(0, 11f);
        graph.InvalidatePreparation();
        Assert.True(graph.Execute(Feed(true, 0, 0), true), graph.LastErrorMessage);
        Assert.Equal(new[] { 11f, 8f }, Values(graph, "y"));
        Assert.Equal(new[] { 3f, 8f }, saved.ToArray());
    }

    [Fact]
    public void MutatingNestedNodeInputsRefreshesParentCaptureReads()
    {
        var graph = Model.Load(Basic(Negate(), Negate()));
        Assert.True(graph.Execute(Feed(true, 3, 8), true));
        var branch = (ComputationalGraph)graph.Nodes[0].Attributes!["then_branch"];
        branch.Nodes[0].Inputs[0] = "unbound";
        Assert.False(graph.Execute(Feed(true, 3, 8), true));
        Assert.Contains("unbound", graph.LastErrorCause?.Message ?? graph.LastErrorMessage ?? "");
        branch.Nodes[0].Inputs[0] = "x";
        Assert.True(graph.Execute(Feed(true, 3, 8), true), graph.LastErrorMessage);
    }

    [Fact]
    public void FailedBranchRetainsPriorResultAndReportsChildFailure()
    {
        var failing = new OnnxSubgraph
        {
            Initializers = { new OnnxTensor { Name = "badshape", ElementType = TensorElementType.Int64, Dims = new[] { 1 }, Data = new long[] { 3 } } },
            Outputs = { Desc("bad", TensorElementType.Float, 2) },
            Nodes = { Op("bad-reshape", "Reshape", new[] { "x", "badshape" }, "bad") },
        };
        var graph = Model.Load(Basic(Direct("x"), failing));
        Assert.True(graph.Execute(Feed(true, 3, 8), true));
        var held = (Tensor<float>)graph.Outputs["y"]!;
        Assert.False(graph.Execute(Feed(false, 1, 2), true));
        Assert.Equal(new[] { 3f, 8f }, held.ToArray());
        var child = Assert.Single(graph.LastSubgraphExecutions);
        Assert.False(child.Succeeded); Assert.Equal("else_branch", child.Branch);
        Assert.NotNull(child.Error);
        Assert.True(graph.Execute(Feed(true, 5, 6), true));
        Assert.True(Assert.Single(graph.LastSubgraphExecutions).Succeeded);
    }

    [Fact]
    public void NestedWallProfilesKeepNodeIdsAndMemoryCountersScoped()
    {
        var inner = new OnnxSubgraph { Outputs = { Desc("z", TensorElementType.Float, 2) },
            Nodes = { Op("add", "Add", new[] { "x", "x" }, "temp"), Op("relu", "Relu", new[] { "temp" }, "z") } };
        var outer = new OnnxSubgraph { Outputs = { Desc("z", TensorElementType.Float, 2) },
            Nodes = { If("inner-if", "cond", inner, inner, "z") } };
        var graph = Model.Load(Basic(outer, outer), false);
        using var scope = Profiler.BeginWallExecution();
        Assert.True(graph.Execute(Feed(true, -3, 8), true), graph.LastErrorMessage);
        var wall = Assert.Single(graph.LastWallProfile!);
        Assert.Equal(OpType.If, wall.Op);
        var child = Assert.Single(graph.LastSubgraphExecutions);
        Assert.Single(child.WallProfile);
        var grandchild = Assert.Single(child.Children);
        Assert.Equal(2, grandchild.WallProfile.Count);
        foreach (var node in grandchild.WallProfile)
        {
            Assert.InRange(node.StartTicks, wall.StartTicks, wall.EndTicks);
            Assert.InRange(node.EndTicks, node.StartTicks, wall.EndTicks);
        }
        Assert.Equal(0, graph.LastPoolAllocatedNewBytes);
        Assert.Equal(0, child.PoolAllocatedNewBytes);
        Assert.True(grandchild.PoolAllocatedNewBytes > 0);
        Assert.True(grandchild.PeakLiveBytes >= 16); // captured x plus its output
        Assert.True(graph.LastAllocatedBytes >= child.AllocatedBytes);
    }

    [Theory]
    [InlineData(10, false)]
    [InlineData(11, true)]
    public void DifferentShapesRespectIfVersion(int version, bool accepted)
    {
        var small = new OnnxSubgraph
        {
            Initializers = { new OnnxTensor { Name = "small", ElementType = TensorElementType.Float, Dims = new[] { 1 }, Data = new[] { 9f } } },
            Outputs = { Desc("small", TensorElementType.Float, 1) },
        };
        var model = Basic(Direct("x"), small); model.Opset[""] = version; model.Outputs[0].Dims = new[] { -1 };
        var graph = Model.Load(model);
        Assert.Equal(accepted, graph.Execute(Feed(false, 3, 8), true));
        if (accepted) Assert.Equal(new[] { 9f }, Values(graph, "y"));
    }

    [Fact]
    public void NestedExternalInitializersHonorMetadataOnlyAndModelBaseDirectory()
    {
        static ValueInfoProto Info(string name, int type, params long[] dims)
        {
            var shape = new TensorShapeProto();
            foreach (long dim in dims) shape.Dim.Add(new TensorShapeProto.Types.Dimension { DimValue = dim });
            return new ValueInfoProto { Name = name, Type = new TypeProto { TensorType = new TypeProto.Types.Tensor { ElemType = type, Shape = shape } } };
        }
        var branch = new GraphProto { Name = "branch" };
        var weight = new TensorProto { Name = "weight", DataType = 1, DataLocation = TensorProto.Types.DataLocation.External };
        weight.Dims.Add(2); weight.ExternalData.Add(new StringStringEntryProto { Key = "location", Value = "weights.bin" });
        branch.Initializer.Add(weight);
        branch.Output.Add(Info("value", 1, 2));
        branch.Node.Add(new NodeProto { Name = "copy", OpType = "Identity", Input = { "weight" }, Output = { "value" } });
        var model = new ModelProto { IrVersion = 8, Graph = new GraphProto { Name = "external" }, OpsetImport = { new OperatorSetIdProto { Domain = "", Version = 14 } } };
        model.Graph.Input.Add(Info("cond", 9)); model.Graph.Output.Add(Info("y", 1, 2));
        var node = new NodeProto { Name = "choose", OpType = "If", Input = { "cond" }, Output = { "y" } };
        foreach (string key in new[] { "then_branch", "else_branch" })
            node.Attribute.Add(new AttributeProto { Name = key, Type = AttributeProto.Types.AttributeType.Graph, G = branch.Clone() });
        model.Graph.Node.Add(node);
        string directory = Path.Combine(Path.GetTempPath(), "lonnx-if-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            string file = Path.Combine(directory, "model.onnx"); File.WriteAllBytes(file, model.ToByteArray());
            var metadata = OnnxImport.ParseMetadata(file);
            Assert.Empty(((OnnxSubgraph)metadata.Nodes[0].Attributes["then_branch"]).Initializers[0].Data);
            Assert.Throws<FileNotFoundException>(() => OnnxImport.Parse(file));
            File.WriteAllBytes(Path.Combine(directory, "weights.bin"), new[] { 3f, 8f }.SelectMany(BitConverter.GetBytes).ToArray());
            var graph = OnnxImport.Load(file); Assert.NotNull(graph);
            Assert.True(graph.Execute(new Dictionary<string, ITensor> { ["cond"] = Feed(true)["cond"] }, true), graph.LastErrorMessage);
            Assert.Equal(new[] { 3f, 8f }, Values(graph, "y"));
        }
        finally { Directory.Delete(directory, true); }
    }

    [Fact]
    public void InvalidNestedIfInUnselectedBranchIsRejected()
    {
        var nested = new OnnxSubgraph { Outputs = { Desc("nested", TensorElementType.Float, 2) },
            Nodes = { If("bad-inner", "cond", Direct("x"), Negate(), "nested") } };
        nested.Nodes[0].Attributes.Remove("else_branch");
        var graph = Model.Load(Basic(Direct("x"), nested));
        Assert.False(graph.Execute(Feed(true, 3, 8), true));
        Assert.Contains("else_branch", graph.LastErrorMessage ?? "");
    }

    [Fact]
    public void SequenceOutputIsExplicitlyUnsupportedUntilElementTypeDescriptorsExist()
    {
        var then = new OnnxSubgraph { Outputs = { Desc("seq", TensorElementType.Sequence) },
            Nodes = { Op("sequence", "SequenceConstruct", new[] { "x" }, "seq") } };
        var model = Basic(then, then); model.Outputs[0] = Desc("y", TensorElementType.Sequence);
        var graph = Model.Load(model);
        Assert.False(graph.Execute(Feed(true, 3, 8), true));
        Assert.Contains("tensor outputs only", graph.LastErrorMessage ?? "");
    }
}
