namespace Lokad.Onnx.Backend.Tests;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using Xunit;
using Xunit.Abstractions;

public class PreparedConversionPrefixTests(ITestOutputHelper output)
{
    static ComputationalGraph Graph(long budget = long.MaxValue)
    {
        var graph = new ComputationalGraph(budget);
        graph.Metadata["Name"] = "prepared-conversion-prefix-test";
        graph.Opset[""] = 17;
        graph.Inputs["x"] = new DenseTensor<float>(new[] { 2f, 3f }, new[] { 1, 2 });
        graph.Initializers["q"] = new DenseTensor<sbyte>(new sbyte[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 });
        graph.Initializers["half_scale"] = new DenseTensor<Half>(new[] { (Half).5f }, Array.Empty<int>());
        graph.Initializers["zero"] = new DenseTensor<sbyte>(new sbyte[] { 0 }, Array.Empty<int>());
        graph.IntermediateOutputs["scale"] = null; graph.IntermediateOutputs["w"] = null;
        graph.Outputs["y"] = new DenseTensor<float>(new[] { 1, 3 });
        graph.Outputs["w"] = new DenseTensor<float>(new[] { 2, 3 });
        graph.Nodes.Add(new Node { Name = "scale", Op = OpType.Cast, Inputs = new[] { "half_scale" }, Outputs = new[] { "scale" }, OpsetVersion = 17,
            Attributes = new() { ["to"] = (int)TensorElementType.Float } });
        graph.Nodes.Add(new Node { Name = "dequantize", Op = OpType.DequantizeLinear, Inputs = new[] { "q", "scale", "zero" }, Outputs = new[] { "w" }, OpsetVersion = 17 });
        graph.Nodes.Add(new Node { Name = "multiply", Op = OpType.MatMul, Inputs = new[] { "x", "w" }, Outputs = new[] { "y" }, OpsetVersion = 17 });
        return graph;
    }

    static Tensor<float> Run(ComputationalGraph graph, ExecutionOptions? options = null)
    {
        graph.Reset();
        var x = new DenseTensor<float>(new[] { 2f, 3f }, new[] { 1, 2 });
        Assert.True(graph.Execute(new Dictionary<string, ITensor> { ["x"] = x }, true, ExecutionProvider.CPU, options), graph.LastErrorMessage);
        return Assert.IsAssignableFrom<Tensor<float>>(graph.Outputs["y"]);
    }

    [Fact]
    public void CompleteGraphUsesPreparedPrefixAndPreservesHeldValues()
    {
        var graph = Graph();
        graph.Prepare();
        var plan = PreparedConversionPrefix.Create(graph);
        if (Vector.IsHardwareAccelerated) Assert.NotNull(plan);
        else Assert.Null(plan);
        var held = Run(graph);
        Assert.Equal(new[] { 7f, 9.5f, 12f }, held.ToArray());
        var oldWeights = Assert.IsAssignableFrom<Tensor<float>>(graph.IntermediateOutputs["w"]);
        if (Vector.IsHardwareAccelerated) Assert.False(Assert.IsType<PreparedFloatTensor>(oldWeights).HasMaterialized);
        oldWeights.SetValue(0, 77f);
        Assert.Equal(new[] { 7f, 9.5f, 12f }, Run(graph).ToArray());
        Assert.Equal(77f, oldWeights.GetValue(0));
        Assert.Equal(new[] { 7f, 9.5f, 12f }, held.ToArray());
    }

    [Fact]
    public void SourceScaleZeroAndReplacementMutationsUseOriginalNodes()
    {
        var graph = Graph(); graph.Prepare();
        Assert.Equal(new[] { 7f, 9.5f, 12f }, Run(graph).ToArray());
        graph.Initializers["q"].SetValue(0, (sbyte)11);
        Assert.Equal(new[] { 17f, 9.5f, 12f }, Run(graph).ToArray());
        graph.Initializers["half_scale"].SetValue(0, (Half).25f);
        Assert.Equal(new[] { 8.5f, 4.75f, 6f }, Run(graph).ToArray());
        graph.Initializers["zero"].SetValue(0, (sbyte)1);
        Assert.Equal(new[] { 7.25f, 3.5f, 4.75f }, Run(graph).ToArray());
        graph.Initializers["q"] = new DenseTensor<sbyte>(new sbyte[] { 1, 1, 1, 1, 1, 1 }, new[] { 2, 3 });
        Assert.Equal(new[] { 0f, 0f, 0f }, Run(graph).ToArray());
    }

    [Fact]
    public void InvalidScaleFailsAtOriginalDequantizationNode()
    {
        var graph = Graph(); graph.Prepare();
        graph.Initializers["half_scale"].SetValue(0, (Half)(-1));
        graph.Reset();
        Assert.False(graph.Execute(new Dictionary<string, ITensor> { ["x"] = new DenseTensor<float>(new[] { 2f, 3f }, new[] { 1, 2 }) }, true));
        Assert.Equal("dequantize", graph.LastFailedNodeName);
        Assert.Contains("positive", graph.LastErrorMessage);
        Assert.False(graph.Outputs.TryGetValue("y", out var failedOutput) && failedOutput is not null);
    }

    [Fact]
    public void BudgetModeAndAttributeRefusalsPreserveExecution()
    {
        var noBudget = Graph(0);
        Assert.Null(PreparedConversionPrefix.Create(noBudget));
        Assert.Null(PreparedConversionPrefix.Create(Graph(36)));
        if (Vector.IsHardwareAccelerated) Assert.NotNull(PreparedConversionPrefix.Create(Graph(37)));
        Assert.Equal(new[] { 7f, 9.5f, 12f }, Run(noBudget).ToArray());
        var graph = Graph(); graph.Prepare();
        Assert.Equal(new[] { 7f, 9.5f, 12f }, Run(graph, ExecutionOptions.Scalar).ToArray());
        Assert.IsType<DenseTensor<float>>(graph.IntermediateOutputs["w"]);
        var custom = Graph();
        custom.Options = ExecutionOptions.Default with { Tensor = ExecutionOptions.Default.Tensor with { CopyReporter = new CopyAccountant() } };
        Assert.Null(PreparedConversionPrefix.Create(custom));
        Assert.Equal(new[] { 7f, 9.5f, 12f }, Run(custom).ToArray());
        Assert.Equal(new[] { 7f, 9.5f, 12f }, Run(Graph(), ExecutionOptions.Memory).ToArray());
        var odd = Graph();
        odd.Nodes[0].Attributes!["saturate"] = 0;
        Assert.Null(PreparedConversionPrefix.Create(odd));
        odd.Nodes[0].Attributes!["saturate"] = long.MaxValue;
        Assert.Null(PreparedConversionPrefix.Create(odd));
    }

    [Fact]
    public void LiveShapesAttributesFeedsAndProducerOrderInvalidatePrefix()
    {
        var graph = Graph(); var plan = PreparedConversionPrefix.Create(graph);
        if (!Vector.IsHardwareAccelerated) { Assert.Null(plan); return; }
        Assert.NotNull(plan); Assert.True(plan.Matches(graph, ExecutionOptions.Default));
        Assert.Equal(37, plan.RetainedBytes);
        var w = (Tensor<float>)plan.Bind(1).Outputs[0];
        Assert.Equal(new[] { .5f, 1f, 1.5f, 2f, 2.5f, 3f }, w.ToArray());
        graph.Inputs["scale"] = DenseTensor<float>.OfValues(new[] { 2f });
        Assert.False(plan.Matches(graph, ExecutionOptions.Default)); graph.Inputs.Remove("scale");
        graph.Nodes[0].Attributes!["saturate"] = 0;
        Assert.False(plan.Matches(graph, ExecutionOptions.Default)); graph.Nodes[0].Attributes!.Remove("saturate");
        graph.Initializers["q"] = graph.Initializers["q"] is Tensor<sbyte> q ? q.Reshape(new[] { 3, 2 }) : throw new Exception();
        Assert.False(plan.Matches(graph, ExecutionOptions.Default));
        var duplicate = Graph(); duplicate.Nodes.Add(duplicate.Nodes[0]);
        Assert.Null(PreparedConversionPrefix.Create(duplicate));
        var reordered = Graph(); (reordered.Nodes[0], reordered.Nodes[1]) = (reordered.Nodes[1], reordered.Nodes[0]);
        Assert.Null(PreparedConversionPrefix.Create(reordered));
    }

    [Fact]
    public void ConsumerErrorsKeepBorrowedWeightNames()
    {
        static string Fail(long budget)
        {
            var graph = Graph(budget);
            graph.Inputs["x"] = new DenseTensor<int>(new[] { 2, 3 }, new[] { 1, 2 });
            Assert.False(graph.Execute(new Dictionary<string, ITensor> { ["x"] = new DenseTensor<int>(new[] { 2, 3 }, new[] { 1, 2 }) }, true));
            Assert.Equal("multiply", graph.LastFailedNodeName);
            return graph.LastErrorMessage!;
        }
        string original = Fail(0);
        Assert.Contains("tensor w ", original);
        Assert.Equal(original, Fail(long.MaxValue));
    }

    [Fact]
    public void MutationsBetweenPrefixNodesAreCheckedAtUse()
    {
        var graph = Graph(); var plan = PreparedConversionPrefix.Create(graph);
        if (!Vector.IsHardwareAccelerated) { Assert.Null(plan); return; }
        Assert.NotNull(plan); Assert.True(plan.Matches(graph, ExecutionOptions.Default));
        Assert.True(plan.TryBind(0, graph.Nodes[0], graph, out var cast));
        graph.IntermediateOutputs["scale"] = cast.Outputs[0];
        Assert.True(plan.TryBind(1, graph.Nodes[1], graph, out _));
        cast.Outputs[0].SetValue(0, -1f);
        Assert.False(plan.TryBind(1, graph.Nodes[1], graph, out _));
        var original = graph.Nodes[1].Execute(graph, ExecutionProvider.CPU, ExecutionOptions.Default);
        Assert.Equal(OpStatus.Failure, original.Status);
        Assert.Contains("positive", original.Message);
        Assert.True(plan.TryBind(0, graph.Nodes[0], graph, out cast));
        graph.IntermediateOutputs["scale"] = cast.Outputs[0];
        graph.Initializers["q"].SetValue(0, (sbyte)99);
        Assert.False(plan.TryBind(1, graph.Nodes[1], graph, out _));
    }

    [SkippableFact]
    public void OriginalInt8OptimizedPrefixIsAdmittedWithoutRewritingModel()
    {
        var root = Environment.GetEnvironmentVariable("LONNX_PARADEE_FIXTURES");
        Skip.If(root is null, "Original Paradee assets are supplied only in the private model lane.");
        var graph = OnnxImport.Load(Path.Combine(root, "int8", "original.onnx"));
        Assert.NotNull(graph);
        // The import already prepared a plan; inspect another explicitly budgeted
        // copy of the source graph so this gate also records its actual census.
        var probe = new ComputationalGraph { Nodes = graph.Nodes, Initializers = graph.Initializers, Opset = graph.Opset };
        var plan = PreparedConversionPrefix.Create(probe);
        if (!Vector.IsHardwareAccelerated) { Assert.Null(plan); return; }
        Assert.NotNull(plan);
        Assert.True(plan.Matches(probe, ExecutionOptions.Default));
        output.WriteLine("Original optimized INT8 eligible prefix: {0} nodes; retained payload: {1} bytes", plan.Count, plan.RetainedBytes);
        Assert.InRange(plan.Count, 350, 418);
        Assert.InRange(plan.RetainedBytes, 31_084_544, 128 * 1024 * 1024);
    }
}
