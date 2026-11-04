namespace Lokad.Onnx.Backend.Tests;

public class ParadeeOperatorTests
{
    static OpResult Run(OpType op, ITensor[] inputs, Dictionary<string, object>? attributes = null, int opset = 17)
    {
        var graph = new ComputationalGraph { Opset = new() { [""] = opset } };
        var names = Enumerable.Range(0, inputs.Length).Select(i => "input" + i).ToArray();
        for (int i = 0; i < inputs.Length; i++) graph.Inputs[names[i]] = inputs[i];
        var node = new Node { Op = op, OpTypeName = op.ToString(), Name = "test", Domain = "",
            OpsetVersion = opset, Inputs = names, Outputs = new[] { "output" }, Attributes = attributes ?? new() };
        return node.Execute(graph, ExecutionProvider.CPU, null);
    }

    static Tensor<T> Output<T>(OpResult result) where T : unmanaged
    {
        Assert.True(result.Status == OpStatus.Success, result.Message);
        return Assert.IsAssignableFrom<Tensor<T>>(Assert.Single(result.Outputs));
    }

    [Fact]
    public void RoundUsesEvenTiesAndPreservesSpecialValues()
    {
        var input = DenseTensor<float>.OfValues(new[] { -2.5f, -1.5f, -0.5f, 0.5f, 1.5f, 2.5f, float.NaN, float.PositiveInfinity });
        var values = Output<float>(Run(OpType.Round, new ITensor[] { input })).ToArray();
        Assert.Equal(new[] { -2f, -2f, -0f, 0f, 2f, 2f }, values.Take(6));
        Assert.True(float.IsNaN(values[6]));
        Assert.Equal(float.PositiveInfinity, values[7]);
        Assert.Equal(unchecked((int)0x80000000), BitConverter.SingleToInt32Bits(values[2]));
        var doubles = Output<double>(Run(OpType.Round, new ITensor[] { DenseTensor<double>.OfValues(new[] { 2.5, 3.5 }) }));
        Assert.Equal(new[] { 2.0, 4.0 }, doubles.ToArray());
    }

    [Fact]
    public void ExpPreservesDoublePrecisionAndIeeeValues()
    {
        var values = Output<double>(Run(OpType.Exp, new ITensor[] { DenseTensor<double>.OfValues(new[] { 0.0, -1.0, 1e-12, double.NegativeInfinity, double.NaN }) })).ToArray();
        Assert.Equal(1.0, values[0]);
        Assert.InRange(Math.Abs(values[1] - 0.36787944117144233), 0, 1e-16);
        Assert.True(values[2] > 1.0);
        Assert.Equal(0.0, values[3]);
        Assert.True(double.IsNaN(values[4]));
        Assert.Equal(OpStatus.Failure, Run(OpType.Exp, new ITensor[] { DenseTensor<int>.OfValues(new[] { 1 }) }).Status);
    }

    [Fact]
    public void NaNDetectionReturnsBoolWithoutMarkingInfinity()
    {
        var values = Output<bool>(Run(OpType.IsNaN, new ITensor[] { DenseTensor<float>.OfValues(new[] { float.NaN, float.PositiveInfinity, 0f, float.NegativeInfinity }) }));
        Assert.Equal(new[] { true, false, false, false }, values.ToArray());
        Assert.Equal(new[] { true, false }, Output<bool>(Run(OpType.IsNaN, new ITensor[] { DenseTensor<Half>.OfValues(new[] { Half.NaN, Half.PositiveInfinity }) })).ToArray());
    }

    [Theory]
    [InlineData(OpType.Greater)]
    [InlineData(OpType.GreaterOrEqual)]
    public void ComparisonsBroadcastAndKeepNanUnordered(OpType op)
    {
        var input = new DenseTensor<float>(new[] { float.NaN, 2f, 1f }, new[] { 3, 1 });
        var right = new DenseTensor<float>(new[] { 1f, 2f }, new[] { 1, 2 });
        var actual = Output<bool>(Run(op, new ITensor[] { input, right }));
        Assert.Equal(new[] { 3, 2 }, actual.Dimensions.ToArray());
        Assert.Equal(op == OpType.Greater ? new[] { false, false, true, false, false, false }
            : new[] { false, false, true, true, true, false }, actual.ToArray());
        var exact = Output<bool>(Run(op, new ITensor[] { DenseTensor<long>.OfValues(new[] { 9007199254740993L }), DenseTensor<long>.OfValues(new[] { 9007199254740992L }) }));
        Assert.True(exact.GetValue(0));
    }

    [Fact]
    public void VariadicMinBroadcastsAndReturnsOwnedValues()
    {
        var input = new DenseTensor<float>(new[] { 3f, 6f }, new[] { 2, 1 });
        var result = Output<float>(Run(OpType.Min, new ITensor[] { input,
            new DenseTensor<float>(new[] { 4f, 2f, 5f }, new[] { 1, 3 }), DenseTensor<float>.OfValues(new[] { 4f }) }));
        Assert.Equal(new[] { 2, 3 }, result.Dimensions.ToArray());
        Assert.Equal(new[] { 3f, 2f, 3f, 4f, 2f, 4f }, result.ToArray());
        var one = Output<float>(Run(OpType.Min, new ITensor[] { input }));
        input.SetValue(0, -100f);
        Assert.Equal(3f, one.GetValue(0));
        var special = Output<float>(Run(OpType.Min, new ITensor[] {
            DenseTensor<float>.OfValues(new[] { float.NaN, 1f, -0f, 0f }),
            DenseTensor<float>.OfValues(new[] { 1f, float.NaN, 0f, -0f }) })).ToArray();
        Assert.True(float.IsNaN(special[0]) && float.IsNaN(special[1]));
        Assert.Equal(0, BitConverter.SingleToInt32Bits(special[2]));
        Assert.Equal(unchecked((int)0x80000000), BitConverter.SingleToInt32Bits(special[3]));
    }

    [Theory]
    [InlineData(0, 1, 24)]
    [InlineData(1, 2, 12)]
    [InlineData(-1, 6, 4)]
    [InlineData(3, 24, 1)]
    public void FlattenNormalizesAxisAndPreservesValues(int axis, int left, int right)
    {
        var input = new DenseTensor<long>(Enumerable.Range(0, 24).Select(x => (long)x).ToArray(), new[] { 2, 3, 4 });
        var actual = Output<long>(Run(OpType.Flatten, new ITensor[] { input }, new() { ["axis"] = axis }));
        Assert.Equal(new[] { left, right }, actual.Dimensions.ToArray());
        Assert.Equal(input.ToArray(), actual.ToArray());
    }

    [Fact]
    public void FlattenHandlesEmptyExtentsAndScalarAxisZero()
    {
        var empty = new DenseTensor<float>(new[] { 2, 0, 3 });
        var result = Output<float>(Run(OpType.Flatten, new ITensor[] { empty }, new() { ["axis"] = 2 }));
        Assert.Equal(new[] { 0, 3 }, result.Dimensions.ToArray());
        var scalar = new DenseTensor<int>(new[] { 7 }, Array.Empty<int>());
        Assert.Equal(new[] { 1, 1 }, Output<int>(Run(OpType.Flatten, new ITensor[] { scalar }, new() { ["axis"] = 0 })).Dimensions.ToArray());
        Assert.Equal(OpStatus.Failure, Run(OpType.Flatten, new ITensor[] { scalar }).Status);
    }

    [Fact]
    public void GatherElementsUsesIndicesCoordinatesAndNegativeAxisIndices()
    {
        var input = new DenseTensor<int>(new[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 });
        var indices = new DenseTensor<long>(new long[] { 2, 0, -1, 1 }, new[] { 2, 2 });
        var result = Output<int>(Run(OpType.GatherElements, new ITensor[] { input, indices }, new() { ["axis"] = -1 }));
        Assert.Equal(new[] { 2, 2 }, result.Dimensions.ToArray());
        Assert.Equal(new[] { 3, 1, 6, 5 }, result.ToArray());
        var transposed = Output<int>(CPUExecutionProvider.Transpose(input, new[] { 1, 0 }, null, null));
        var rowIndices = new DenseTensor<int>(new[] { 2, 0, 1, 1 }, new[] { 2, 2 });
        Assert.Equal(new[] { 3, 4, 2, 5 }, Output<int>(Run(OpType.GatherElements, new ITensor[] { transposed, rowIndices })).ToArray());
    }

    [Theory]
    [InlineData(-4)]
    [InlineData(3)]
    [InlineData(long.MinValue)]
    public void GatherElementsRejectsIndicesOutsideExtent(long index)
    {
        var result = Run(OpType.GatherElements, new ITensor[] { DenseTensor<float>.OfValues(new[] { 1f, 2f, 3f }), DenseTensor<long>.OfValues(new[] { index }) });
        Assert.Equal(OpStatus.Failure, result.Status);
        Assert.Contains("index", result.Message);
    }

    [Fact]
    public void GatherElementsValidatesShapeAndAllowsEmptyIndices()
    {
        var input = new DenseTensor<float>(new[] { 2, 3 });
        Assert.Equal(OpStatus.Failure, Run(OpType.GatherElements, new ITensor[] { input, new DenseTensor<long>(new[] { 3, 1 }) }, new() { ["axis"] = 1 }).Status);
        Assert.Equal(OpStatus.Failure, Run(OpType.GatherElements, new ITensor[] { input, DenseTensor<long>.OfValues(new long[] { 0 }) }).Status);
        Assert.Equal(0, Output<float>(Run(OpType.GatherElements, new ITensor[] { input, new DenseTensor<long>(new[] { 0, 3 }) })).Length);
    }

    [Theory]
    [InlineData(false, false, new long[] { 1, 3, 6, 4, 9, 15 })]
    [InlineData(false, true, new long[] { 6, 5, 3, 15, 11, 6 })]
    [InlineData(true, false, new long[] { 0, 1, 3, 0, 4, 9 })]
    [InlineData(true, true, new long[] { 5, 3, 0, 11, 6, 0 })]
    public void CumSumSupportsReverseAndExclusiveRows(bool exclusive, bool reverse, long[] expected)
    {
        var input = new DenseTensor<long>(new long[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 });
        var axis = new DenseTensor<long>(new long[] { -1 }, Array.Empty<int>());
        var result = Run(OpType.CumSum, new ITensor[] { input, axis }, new() { ["exclusive"] = exclusive ? 1 : 0, ["reverse"] = reverse ? 1 : 0 });
        Assert.Equal(expected, Output<long>(result).ToArray());
    }

    [Fact]
    public void CumSumPreservesLargeIntegerPayloadsAndDoublePrecision()
    {
        var axis = new DenseTensor<int>(new[] { 0 }, Array.Empty<int>());
        var integers = DenseTensor<long>.OfValues(new[] { 9007199254740993L, 1, -2 });
        Assert.Equal(new[] { 9007199254740993L, 9007199254740994L, 9007199254740992L }, Output<long>(Run(OpType.CumSum, new ITensor[] { integers, axis })).ToArray());
        var values = Output<double>(Run(OpType.CumSum, new ITensor[] { DenseTensor<double>.OfValues(new[] { 1.0, 1e-12, 1e-12 }), axis })).ToArray();
        Assert.InRange(values[2] - 1.0, 1.9e-12, 2.1e-12);
        Assert.Equal(0, Output<float>(Run(OpType.CumSum, new ITensor[] { new DenseTensor<float>(new[] { 0, 2 }), axis })).Length);
    }

    [Fact]
    public void CumSumValidatesAxisAndBooleanAttributes()
    {
        var input = DenseTensor<float>.OfValues(new[] { 1f, 2f });
        Assert.Equal(OpStatus.Failure, Run(OpType.CumSum, new ITensor[] { input, DenseTensor<long>.OfValues(new long[] { 0 }) }).Status);
        Assert.Equal(OpStatus.Failure, Run(OpType.CumSum, new ITensor[] { input, new DenseTensor<long>(new long[] { 1 }, Array.Empty<int>()) }).Status);
        Assert.Equal(OpStatus.Failure, Run(OpType.CumSum, new ITensor[] { input, new DenseTensor<long>(new long[] { 0 }, Array.Empty<int>()) }, new() { ["exclusive"] = 2 }).Status);
    }

    [Theory]
    [InlineData(OpType.Round, 10)]
    [InlineData(OpType.IsNaN, 8)]
    [InlineData(OpType.GatherElements, 10)]
    [InlineData(OpType.CumSum, 10)]
    [InlineData(OpType.GreaterOrEqual, 11)]
    public void SchemasRejectOpsBeforeTheirIntroduction(OpType op, int version)
    {
        Assert.False(OperatorSchemas.IsSupported(op, "", version, false));
        Assert.True(OperatorSchemas.IsSupported(op, "", 17, false));
        Assert.False(OperatorSchemas.IsSupported(op, "custom", 17, false));
    }
}
