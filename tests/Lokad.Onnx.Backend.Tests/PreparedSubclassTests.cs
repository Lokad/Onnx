namespace Lokad.Onnx.Backend.Tests;

using System;
using System.Collections.Generic;
using Xunit;

public class PreparedSubclassTests
{
    sealed class FailingQuantizedTensor() : DenseTensor<sbyte>(new sbyte[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 })
    {
        public override DenseTensor<sbyte> ToDenseTensor() => throw new InvalidOperationException("quantized storage conversion failed");
    }

    static ComputationalGraph Graph(long budget)
    {
        var g = new ComputationalGraph(budget);
        g.Metadata["Name"] = "prepared-subclass-public-contract";
        g.Opset[""] = 17;
        g.Inputs["x"] = new DenseTensor<float>(new[] { 2f, 3f }, new[] { 1, 2 });
        g.Initializers["q"] = new DenseTensor<sbyte>(new sbyte[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 });
        g.Initializers["half_scale"] = new DenseTensor<Half>(new[] { (Half).5f }, Array.Empty<int>());
        g.Initializers["zero"] = new DenseTensor<sbyte>(new sbyte[] { 0 }, Array.Empty<int>());
        g.IntermediateOutputs["scale"] = null; g.IntermediateOutputs["w"] = null;
        g.Outputs["y"] = new DenseTensor<float>(new[] { 1, 3 });
        g.Nodes.Add(new Node { Name = "scale", Op = OpType.Cast, Inputs = new[] { "half_scale" }, Outputs = new[] { "scale" }, OpsetVersion = 17,
            Attributes = new() { ["to"] = (int)TensorElementType.Float } });
        g.Nodes.Add(new Node { Name = "dequantize", Op = OpType.DequantizeLinear, Inputs = new[] { "q", "scale", "zero" }, Outputs = new[] { "w" }, OpsetVersion = 17 });
        g.Nodes.Add(new Node { Name = "multiply", Op = OpType.MatMul, Inputs = new[] { "x", "w" }, Outputs = new[] { "y" }, OpsetVersion = 17 });
        return g;
    }

    [Fact]
    public void ReplacedInitializerSubclassKeepsOriginalConversionFailure()
    {
        static string Fail(long budget)
        {
            var graph = Graph(budget);
            graph.Prepare();
            graph.Initializers["q"] = new FailingQuantizedTensor();
            graph.Reset();
            Assert.False(graph.Execute(new Dictionary<string, ITensor> { ["x"] = new DenseTensor<float>(new[] { 2f, 3f }, new[] { 1, 2 }) }, true));
            Assert.Equal("dequantize", graph.LastFailedNodeName);
            Assert.Contains("quantized storage conversion failed", graph.LastErrorMessage);
            return graph.LastErrorMessage!;
        }
        Assert.Equal(Fail(0), Fail(long.MaxValue));
    }

    [Fact]
    public void InitializerSubclassAtPreparationKeepsOriginalConversionFailure()
    {
        static string Fail(long budget)
        {
            var graph = Graph(budget);
            graph.Initializers["q"] = new FailingQuantizedTensor();
            graph.Prepare();
            graph.Reset();
            Assert.False(graph.Execute(new Dictionary<string, ITensor> { ["x"] = new DenseTensor<float>(new[] { 2f, 3f }, new[] { 1, 2 }) }, true));
            Assert.Equal("dequantize", graph.LastFailedNodeName);
            Assert.Contains("quantized storage conversion failed", graph.LastErrorMessage);
            return graph.LastErrorMessage!;
        }
        Assert.Equal(Fail(0), Fail(long.MaxValue));
    }
}
