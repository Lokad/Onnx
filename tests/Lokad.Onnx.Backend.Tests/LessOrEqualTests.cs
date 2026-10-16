using System;
using System.Collections.Generic;
using System.Linq;

namespace Lokad.Onnx.Backend.Tests;

public class LessOrEqualTests
{
    static ComputationalGraph Graph(ITensor a, ITensor b) => Graph(a, b, 12, "");

    static ComputationalGraph Graph(ITensor a, ITensor b, int version, string domain)
    {
        var model = new OnnxModel { Name = "ordered-comparison" };
        model.Opset[""] = version;
        model.Inputs.Add(new OnnxValueInfo { Name = "a", ElementType = a.ElementType, Dims = a.Dims });
        model.Inputs.Add(new OnnxValueInfo { Name = "b", ElementType = b.ElementType, Dims = b.Dims });
        model.Outputs.Add(new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Bool,
            Dims = Enumerable.Repeat(-1, Math.Max(a.Rank, b.Rank)).ToArray() });
        model.Nodes.Add(new OnnxNode { Name = "compare", OpType = "LessOrEqual", Domain = domain,
            Inputs = new[] { "a", "b" }, Outputs = new[] { "y" }, Attributes = new Dictionary<string, object>() });
        return Model.Load(model)!;
    }

    static bool[] Execute(ITensor a, ITensor b)
    {
        var graph = Graph(a, b);
        Assert.True(graph.Execute(new Dictionary<string, ITensor> { ["a"] = a, ["b"] = b }, true), graph.LastErrorMessage);
        return ((Tensor<bool>)graph.Outputs["y"]).ToArray();
    }

    [Fact]
    public void FloatComparison_IsOrderedAndPreservesInputs()
    {
        var values = new[] { float.NaN, float.NegativeInfinity, -0f, 0f, float.PositiveInfinity, 1f };
        var a = DenseTensor<float>.OfValues(values);
        var b = DenseTensor<float>.OfValues(new[] { 0f });
        Assert.Equal(new[] { false, true, true, true, false, false }, Execute(a, b));
        Assert.Equal(values, a.ToArray());
        Assert.Equal(new[] { false, false, false, false, false, false },
            Execute(a, DenseTensor<float>.OfValues(new[] { float.NaN })));
    }

    [Fact]
    public void DoubleComparison_HandlesNaNsAndInfinities()
    {
        var a = DenseTensor<double>.OfValues(new[] { double.NaN, double.PositiveInfinity, -0d, -1d });
        var b = DenseTensor<double>.OfValues(new[] { double.NaN, double.PositiveInfinity, 0d, -2d });
        Assert.Equal(new[] { false, true, true, false }, Execute(a, b));
    }

    [Fact]
    public void BroadcastAcrossBothInputs_UsesMultidirectionalShape()
    {
        var a = DenseTensor<float>.OfValues(new float[,] { { 1 }, { 3 } });
        var b = DenseTensor<float>.OfValues(new float[,] { { 0, 1, 4 } });
        Assert.Equal(new[] { false, true, true, false, false, true }, Execute(a, b));
    }

    [Fact]
    public void IntegerLimits_AvoidFloatingPointConversionsAndOverflow()
    {
        Assert.Equal(new[] { true, false, true }, Execute(
            DenseTensor<long>.OfValues(new[] { long.MinValue, long.MaxValue, long.MaxValue - 1 }),
            DenseTensor<long>.OfValues(new[] { 0L, long.MaxValue - 1, long.MaxValue })));
        Assert.Equal(new[] { true, false }, Execute(
            DenseTensor<ulong>.OfValues(new[] { 0UL, ulong.MaxValue }),
            DenseTensor<ulong>.OfValues(new[] { ulong.MaxValue, ulong.MaxValue - 1 })));
    }

    [Fact]
    public void MixedTypes_AndBooleanInputsAreRejected()
    {
        var a = DenseTensor<float>.OfValues(new[] { 1f });
        var b = DenseTensor<long>.OfValues(new[] { 1L });
        var graph = Graph(a, b);
        Assert.False(graph.Execute(new Dictionary<string, ITensor> { ["a"] = a, ["b"] = b }, true));
        var boolean = DenseTensor<bool>.OfValues(new[] { false });
        graph = Graph(boolean, boolean);
        Assert.False(graph.Execute(new Dictionary<string, ITensor> { ["a"] = boolean, ["b"] = boolean }, true));
    }

    [Theory]
    [InlineData(11, "", false)]
    [InlineData(12, "", true)]
    [InlineData(21, "ai.onnx", true)]
    [InlineData(21, "custom.domain", false)]
    public void Schema_EnforcesVersionAndDomain(int version, string domain, bool expected)
    {
        var a = DenseTensor<float>.OfValues(new[] { 1f });
        var graph = Graph(a, a, version, domain);
        Assert.Equal(expected, CPUExecutionProvider.SupportsNode(graph.Nodes[0]));
    }
}
