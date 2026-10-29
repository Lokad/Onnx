using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using Lokad.Onnx.Optimization;

namespace Lokad.Onnx.Backend.Tests;

[Collection("SequentialLogSink")]
public class ConformerConvTests
{
    const int C = 1024;
    static ComputationalGraph Graph(int time, int masked = 0, bool denseWeights = false)
    {
        var graph = new ComputationalGraph(0);
        graph.Metadata["Name"] = "conformer-region";
        graph.Opset[""] = 18;
        graph.Inputs["x"] = new DenseTensor<float>(Enumerable.Range(0, time * C).Select(i => (i % 37 - 18) * .03125f).ToArray(), new[] { 1, time, C });
        graph.Inputs["mask"] = new DenseTensor<bool>(Enumerable.Range(0, time).Select(t => t >= time - masked).ToArray(), new[] { 1, 1, time });
        var w1 = new float[2 * C * C]; var w2 = new float[C * C]; var dw = new float[C * 9];
        for (int c = 0; c < C; c++)
        {
            w1[c * C + c] = .25f; w1[(C + c) * C + c] = -.5f; w2[c * C + c] = .75f;
            dw[c * 9 + 2] = -.125f; dw[c * 9 + 4] = .5f; dw[c * 9 + 7] = .25f;
        }
        if (denseWeights)
        {
            var random = new Random(2900);
            for (int i = 0; i < w1.Length; i++) w1[i] = random.NextSingle() * .25f - .125f;
            for (int i = 0; i < w2.Length; i++) w2[i] = random.NextSingle() * .25f - .125f;
        }
        graph.Initializers["w1"] = new DenseTensor<float>(w1, new[] { 2 * C, C, 1 });
        graph.Initializers["w2"] = new DenseTensor<float>(w2, new[] { C, C, 1 });
        graph.Initializers["dw"] = new DenseTensor<float>(dw, new[] { C, 1, 9 });
        graph.Initializers["bias"] = new DenseTensor<float>(Enumerable.Repeat(.0625f, C).ToArray(), new[] { C });
        graph.Outputs["y"] = new DenseTensor<float>(new[] { 1, time, C });
        Node Node(OpType op, string name, string[] inputs, string[] outputs, Dictionary<string, object>? attributes = null, bool fused = false)
            => new() { ID = graph.Nodes.Count + 1, Op = op, OpTypeName = op.ToString(), Name = name, Domain = "", OpsetVersion = 18, IsFused = fused, Inputs = inputs, Outputs = outputs, Attributes = attributes };
        Dictionary<string, object> Conv(int group, int kernel) => new() { ["group"] = group, ["kernel_shape"] = new[] { kernel }, ["pads"] = new[] { 0, 0 }, ["strides"] = new[] { 1 }, ["dilations"] = new[] { 1 } };
        graph.Nodes.Add(Node(OpType.Transpose, "start", new[] { "x" }, new[] { "tx" }, new() { ["perm"] = new[] { 0, 2, 1 } }));
        graph.Nodes.Add(Node(OpType.Conv, "pw1", new[] { "tx", "w1" }, new[] { "up" }, Conv(1, 1)));
        graph.Nodes.Add(Node(OpType.Split, "split", new[] { "up" }, new[] { "left", "right" }, new() { ["axis"] = 1 }));
        graph.Nodes.Add(Node(OpType.Sigmoid, "sigmoid", new[] { "right" }, new[] { "sigmoid" }));
        graph.Nodes.Add(Node(OpType.Mul, "gate", new[] { "left", "sigmoid" }, new[] { "gate" }));
        graph.Nodes.Add(Node(OpType.Constant, "zero", Array.Empty<string>(), new[] { "zero" }, new() { ["value"] = new DenseTensor<float>(new[] { 0f }, Array.Empty<int>()) }));
        graph.Nodes.Add(Node(OpType.Where, "mask", new[] { "mask", "zero", "gate" }, new[] { "masked" }));
        graph.Nodes.Add(Node(OpType.Constant, "pads", Array.Empty<string>(), new[] { "pads" }, new() { ["value"] = new DenseTensor<long>(new long[] { 0, 0, 4, 0, 0, 4 }, new[] { 6 }) }));
        graph.Nodes.Add(Node(OpType.Pad, "pad", new[] { "masked", "pads", "" }, new[] { "padded" }, new() { ["mode"] = "constant" }));
        graph.Nodes.Add(Node(OpType.Conv, "depthwise", new[] { "padded", "dw", "bias" }, new[] { "filtered" }, Conv(C, 9)));
        graph.Nodes.Add(Node(OpType.FusedSigmoidMul, "silu", new[] { "filtered" }, new[] { "activated" }, fused: true));
        graph.Nodes.Add(Node(OpType.Conv, "pw2", new[] { "activated", "w2" }, new[] { "down" }, Conv(1, 1)));
        graph.Nodes.Add(Node(OpType.Transpose, "/layers.0/conv/Transpose_1", new[] { "down" }, new[] { "y" }, new() { ["perm"] = new[] { 0, 2, 1 } }));
        return graph;
    }

    static Tensor<float> Reference(ComputationalGraph graph, ExecutionOptions options)
    {
        var result = CPUExecutionProvider.ConformerConv(graph.Inputs["x"], graph.Inputs["mask"], graph.Initializers["w1"],
            graph.Initializers["dw"], graph.Initializers["bias"], graph.Initializers["w2"],
            (ITensor)graph.Nodes.Single(n => n.Name == "pads").Attributes!["value"],
            (ITensor)graph.Nodes.Single(n => n.Name == "zero").Attributes!["value"], options);
        Assert.Equal(OpStatus.Success, result.Status);
        return (Tensor<float>)result.Outputs[0];
    }
    static void Exact(float[] expected, Tensor<float> actual) =>
        Assert.True(MemoryMarshal.AsBytes(expected.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(actual.ToArray().AsSpan())));

    [Fact]
    public void FusionKeepsLiveWeightsMaskAndOriginalFallback()
    {
        var graph = Graph(3, 1);
        var original = Reference(graph, ExecutionOptions.Memory).ToArray();
        Assert.Equal(1, GraphFusion.FuseConformerConv(graph, GraphFacts.Build(graph)));
        var fused = Assert.Single(graph.Nodes.Where(n => n.Op == OpType.FusedConformerConv));
        Assert.Equal(new[] { "x", "mask", "w1", "dw", "bias", "w2", "pads", "zero" }, fused.Inputs);
        var feeds = graph.Inputs.ToDictionary(p => p.Key, p => p.Value!);
        Assert.True(graph.Execute(feeds, true, ExecutionProvider.CPU, ExecutionOptions.Memory), graph.LastErrorMessage);
        Exact(original, (Tensor<float>)graph.Outputs["y"]);
        Assert.Equal(0, GraphFusion.FuseConformerConv(graph, GraphFacts.Build(graph)));
    }

    [Theory]
    [InlineData("visible")] [InlineData("shared")] [InlineData("captured")]
    [InlineData("padding")] [InlineData("stride")] [InlineData("attribute")]
    [InlineData("commuted")] [InlineData("intervening")] [InlineData("rebinding")]
    public void UnprovedRegionsStayUnchanged(string reason)
    {
        var graph = Graph(3);
        Node Extra(string input, string output) => new() { Op = OpType.Identity, OpTypeName = "Identity", Domain = "", OpsetVersion = 18, Name = output, Inputs = new[] { input }, Outputs = new[] { output } };
        switch (reason)
        {
            case "visible": graph.Outputs["up"] = new DenseTensor<float>(new[] { 1, 2 * C, 3 }); break;
            case "shared": graph.Nodes.Add(Extra("up", "held")); break;
            case "captured":
                var branch = new ComputationalGraph(); branch.Nodes.Add(Extra("up", "z")); branch.Outputs["z"] = new DenseTensor<float>(new[] { 1 });
                graph.Nodes.Add(new Node { Op = OpType.If, Inputs = new[] { "condition" }, Outputs = new[] { "branch" }, Attributes = new() { ["then_branch"] = branch, ["else_branch"] = branch } }); break;
            case "padding": graph.Nodes.Single(n => n.Name == "pad").Attributes!["mode"] = "reflect"; break;
            case "stride": graph.Nodes.Single(n => n.Name == "pw1").Attributes!["strides"] = new[] { 2 }; break;
            case "attribute": graph.Nodes.Single(n => n.Name == "pw2").Attributes!["unproved"] = 1; break;
            case "commuted": var mul = graph.Nodes[4]; mul.Inputs = new[] { "sigmoid", "left" }; graph.Nodes[4] = mul; break;
            case "intervening": graph.Nodes.Insert(4, Extra("x", "unrelated")); break;
            case "rebinding": graph.Nodes.Insert(4, Extra("x", "x")); break;
        }
        int count = graph.Nodes.Count;
        Assert.Equal(0, GraphFusion.FuseConformerConv(graph, GraphFacts.Build(graph)));
        Assert.Equal(count, graph.Nodes.Count);
    }

    [Fact]
    public void FusedFallbackRetainsLivePaddingAndFillValues()
    {
        var graph = Graph(3, 1);
        Assert.Equal(1, GraphFusion.FuseConformerConv(graph, GraphFacts.Build(graph)));
        var pads = (Tensor<long>)graph.Nodes.Single(n => n.Name == "pads").Attributes!["value"];
        var zero = (Tensor<float>)graph.Nodes.Single(n => n.Name == "zero").Attributes!["value"];
        pads.SetValue(2, 3); pads.SetValue(5, 5); zero.SetValue(0, .5f);
        var expected = Reference(graph, ExecutionOptions.Memory).ToArray();
        var feeds = graph.Inputs.ToDictionary(p => p.Key, p => p.Value!);
        Assert.True(graph.Execute(feeds, true, ExecutionProvider.CPU, ExecutionOptions.Memory), graph.LastErrorMessage);
        Exact(expected, (Tensor<float>)graph.Outputs["y"]);
    }

    [SkippableFact]
    public void ExplicitPreparationReplacesOnlyPrivatePointwisePayloads()
    {
        Skip.If(!Fma.IsSupported, "FMA required for owned preparation.");
        var graph = Graph(3, 1);
        var expected = Reference(graph, ExecutionOptions.Memory).ToArray();
        var original = Assert.IsType<DenseTensor<float>>(graph.Initializers["w1"]);
        Assert.Equal(1, GraphFusion.FuseConformerConv(graph, GraphFacts.Build(graph)));
        Assert.Equal(2, graph.PrepareOwnedMatMulWeights());
        var packed = Assert.IsType<OwnedPackedTensor>(graph.Initializers["w1"]);
        Exact(original.ToArray(), packed);
        Assert.Equal(12L * 1024 * 1024, graph.OwnedPackedWeightBytes);
        Assert.Equal(0, graph.RetainedPackedWeightBytes);
        Assert.Equal(0, graph.PrepareOwnedMatMulWeights());
        Exact(expected, Reference(graph, ExecutionOptions.Memory));
        original.Buffer.Span.Fill(37);
        Exact(expected, Reference(graph, ExecutionOptions.Memory));
        packed[0, 0, 0] = .5f;
        var next = Reference(graph, ExecutionOptions.Memory);
        var dense = packed.ToDenseTensor(); graph.Initializers["w1"] = dense;
        Exact(next.ToArray(), Reference(graph, ExecutionOptions.Memory));
    }

    [SkippableTheory]
    [InlineData(48, 0)] [InlineData(49, 7)] [InlineData(89, 7)] [InlineData(156, 0)]
    [InlineData(222, 0)] [InlineData(225, 7)]
    public void PreparedRegionPreservesBitsScratchAndHeldOutputs(int time, int masked)
    {
        Skip.If(!Avx512F.IsSupported || !Fma.IsSupported, "AVX512/FMA required for region execution.");
        var graph = Graph(time, masked, true);
        var expected = Reference(graph, ExecutionOptions.Memory).ToArray();
        Assert.Equal(1, GraphFusion.FuseConformerConv(graph, GraphFacts.Build(graph)));
        Assert.Equal(2, graph.PrepareOwnedMatMulWeights());
        var scratch = new ScratchAccountant();
        var options = new ExecutionOptions(OptimizationMode.Memory, TensorExecutionOptions.Intrinsics with { ScratchReporter = scratch });
        var held = Reference(graph, options);
        Exact(expected, held);
        Assert.InRange(scratch.TotalScratchBytes, 1, 17 * 1024 * 1024);
        var saved = held.ToArray();
        var failing = new ExecutionOptions(OptimizationMode.Memory, TensorExecutionOptions.Intrinsics with { ScratchReporter = new ThrowingScratch() });
        Assert.Throws<InvalidOperationException>(() => Reference(graph, failing));
        Exact(expected, Reference(graph, options));
        Exact(saved, held);
    }

    [SkippableTheory]
    [InlineData("shared")] [InlineData("visible")] [InlineData("alias")]
    public void PreparationProtectsSharedOrVisiblePointwiseWeights(string reason)
    {
        Skip.If(!Fma.IsSupported, "FMA required for owned preparation.");
        var graph = Graph(3);
        var source = Assert.IsType<DenseTensor<float>>(graph.Initializers["w1"]);
        Assert.Equal(1, GraphFusion.FuseConformerConv(graph, GraphFacts.Build(graph)));
        switch (reason)
        {
            case "shared": graph.Nodes.Add(new Node { Op = OpType.Identity, OpTypeName = "Identity", Domain = "", OpsetVersion = 18, Name = "held", Inputs = new[] { "w1" }, Outputs = new[] { "held" } }); break;
            case "visible": graph.Outputs["w1"] = source; break;
            case "alias": graph.Initializers["alias"] = new DenseTensor<float>(source.Buffer, source.Dimensions); break;
        }
        Assert.Equal(1, graph.PrepareOwnedMatMulWeights());
        Assert.Same(source, graph.Initializers["w1"]);
        Assert.IsType<OwnedPackedTensor>(graph.Initializers["w2"]);
    }

    sealed class ThrowingScratch : IScratchAccountant
    {
        int calls;
        public long TotalScratchBytes => 0;
        public void AddScratchBytes(long bytes) { if (++calls == 3) throw new InvalidOperationException("Injected scratch failure"); }
    }

    [SkippableFact]
    public void PreparedRegionKeepsExceptionalClassificationAndOriginalInputs()
    {
        Skip.If(!Avx512F.IsSupported || !Fma.IsSupported, "AVX512/FMA required for region execution.");
        var graph = Graph(49, 7, true);
        var input = (Tensor<float>)graph.Inputs["x"]!;
        input.SetValue(0, BitConverter.Int32BitsToSingle(unchecked((int)0xffc12345)));
        input.SetValue(1024, float.PositiveInfinity);
        var saved = input.ToArray();
        var expected = Reference(graph, ExecutionOptions.Memory).ToArray();
        Assert.Equal(2, graph.PrepareOwnedMatMulWeights());
        var actual = Reference(graph, ExecutionOptions.Memory).ToArray();
        for (int i = 0; i < expected.Length; i++)
            if (float.IsNaN(expected[i])) Assert.True(float.IsNaN(actual[i]));
            else Assert.Equal(BitConverter.SingleToInt32Bits(expected[i]), BitConverter.SingleToInt32Bits(actual[i]));
        Exact(saved, input);
    }

    [Theory]
    [InlineData("private")] [InlineData("shared")] [InlineData("visible")] [InlineData("alias")]
    public void RegionCollapseRequiresExplicitPreparationHardwareAndTwoAdmissibleWeights(string reason)
    {
        var graph = Graph(3);
        var source = Assert.IsType<DenseTensor<float>>(graph.Initializers["w1"]);
        switch (reason)
        {
            case "shared": graph.Nodes.Add(new Node { Op = OpType.Identity, OpTypeName = "Identity", Domain = "", OpsetVersion = 18, Name = "held", Inputs = new[] { "w1" }, Outputs = new[] { "held" } }); break;
            case "visible": graph.Outputs["w1"] = source; break;
            case "alias": graph.Initializers["alias"] = new DenseTensor<float>(source.Buffer, source.Dimensions); break;
        }
        graph.Prepare();
        Assert.DoesNotContain(graph.Nodes, n => n.Op == OpType.FusedConformerConv);
        bool admitted = Avx512F.IsSupported && Fma.IsSupported && reason == "private";
        Assert.Equal(admitted ? 2 : 0, graph.PrepareOwnedMatMulWeights());
        Assert.Equal(admitted ? 1 : 0, graph.Nodes.Count(n => n.Op == OpType.FusedConformerConv));
        if (!admitted) Assert.Same(source, graph.Initializers["w1"]);
    }
}
