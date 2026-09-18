namespace Lokad.Onnx.Backend.Tests;

public class EncoderFoundationEdgeTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void BooleanAndUsesLogicalCoordinatesForReversedAndOffsetInputs(bool broadcast)
    {
        var left = new DenseTensor<bool>(new[] { 2, 3 }, true);
        var values = new[] { true, false, true, false, true, false };
        for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++) left[i, j] = values[i * 3 + j];
        int rows = broadcast ? 1 : 2;
        var data = new[] { false, true, true, false, true, false, true, false };
        var right = new DenseTensor<bool>(data.AsMemory(1, rows * 3), new[] { rows, 3 });
        var result = CPUExecutionProvider.And(left, right, null);
        Assert.Equal(OpStatus.Success, result.Status);
        var output = (Tensor<bool>)result.Outputs[0];
        for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++)
            Assert.Equal(values[i * 3 + j] && right[broadcast ? 0 : i, j], output[i, j]);
        var inverse = CPUExecutionProvider.Not(left, null);
        for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++)
            Assert.Equal(!values[i * 3 + j], ((Tensor<bool>)inverse.Outputs[0])[i, j]);
    }

    [Theory]
    [InlineData(4294967297L, 0L)]
    [InlineData(-4294967295L, 0L)]
    [InlineData(2147483647L, 2147483647L)]
    [InlineData(-3L, 0L)]
    public void InvalidPaddingCannotWrapIntoAValidOutput(long begin, long end)
    {
        var data = DenseTensor<float>.OfValues(new[] { 1f, 2f });
        var result = CPUExecutionProvider.Pad(data, DenseTensor<long>.OfValues(new[] { begin, end }), null, null, null, null, null);
        Assert.Equal(OpStatus.Failure, result.Status);
        Assert.Equal(new[] { 1f, 2f }, data.ToArray());
    }

    [Fact]
    public void PadsRequireRankOneAndOutputVolumeMustFitAnArray()
    {
        var data = new DenseTensor<float>(new[] { 2, 2 });
        var matrix = new DenseTensor<long>(new long[4], new[] { 2, 2 });
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Pad(data, matrix, null, null, null, null, null).Status);
        var enormous = DenseTensor<long>.OfValues(new long[] { 50000, 50000, 0, 0 });
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Pad(data, enormous, null, null, null, null, null).Status);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void PadCropsAndFillsUsingCoordinatesWithoutChangingInt64Bits(int layout)
    {
        var raw = new long[12];
        var data = new DenseTensor<long>(raw.AsMemory(2, 8), new[] { 2, 4 }, layout == 2);
        const long high = 9007199254740993;
        for (int row = 0; row < 2; row++) for (int column = 0; column < 4; column++) data[row, column] = high + row * 4 + column;
        var before = (long[])raw.Clone();
        var pads = DenseTensor<long>.OfValues(new long[] { 1, -1, 0, 2 });
        var fill = new DenseTensor<long>(new[] { -high }, Array.Empty<int>());
        var opts = layout == 0 ? ExecutionOptions.Scalar : layout == 1 ? ExecutionOptions.Simd : ExecutionOptions.Intrinsics;
        var result = CPUExecutionProvider.Pad(data, pads, fill, "constant", null, null, opts);
        Assert.Equal(OpStatus.Success, result.Status);
        var output = (Tensor<long>)result.Outputs[0];
        Assert.Equal(new[] { 3, 5 }, output.Dimensions.ToArray());
        Assert.Equal(new long[] { -high, -high, -high, -high, -high,
            high + 1, high + 2, high + 3, -high, -high,
            high + 5, high + 6, high + 7, -high, -high }, output.ToArray());
        Assert.Equal(before, raw);
    }

    [Fact]
    public void EmptyPadInputCanBecomeFilledAndFullCroppingProducesEmptyOutput()
    {
        var empty = new DenseTensor<double>(new[] { 0, 2 });
        var fill = DenseTensor<double>.OfValues(new[] { -2.5 });
        var padded = CPUExecutionProvider.Pad(empty, DenseTensor<long>.OfValues(new long[] { 1, 0, 1, 0 }), fill, null, null, null, null);
        Assert.Equal(OpStatus.Success, padded.Status);
        Assert.Equal(new[] { -2.5, -2.5, -2.5, -2.5 }, ((Tensor<double>)padded.Outputs[0]).ToArray());
        var cropped = CPUExecutionProvider.Pad(DenseTensor<float>.OfValues(new[] { 1f, 2f }),
            DenseTensor<long>.OfValues(new long[] { -1, -1 }), null, null, null, null, null);
        Assert.Equal(OpStatus.Success, cropped.Status);
        Assert.Empty(((Tensor<float>)cropped.Outputs[0]).ToArray());
    }

    [Fact]
    public void UnaryOperatorsPreserveSpecialValuesAndHandleEmptyAndScalarShapes()
    {
        var values = new[] { float.NegativeInfinity, -0f, float.PositiveInfinity, float.NaN, -1.25f, 2.75f };
        var input = new DenseTensor<float>(new[] { 2, 3 }, true);
        for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++) input[i, j] = values[i * 3 + j];
        var sigmoid = (Tensor<float>)CPUExecutionProvider.Sigmoid(input, null).Outputs[0];
        var floor = (Tensor<float>)CPUExecutionProvider.Floor(input, null).Outputs[0];
        Assert.Equal(0f, sigmoid[0, 0]);
        Assert.Equal(.5f, sigmoid[0, 1]);
        Assert.Equal(1f, sigmoid[0, 2]);
        Assert.True(float.IsNaN(sigmoid[1, 0]));
        Assert.Equal(float.NegativeInfinity, floor[0, 0]);
        Assert.Equal(unchecked((int)0x80000000), BitConverter.SingleToInt32Bits(floor[0, 1]));
        Assert.Equal(float.PositiveInfinity, floor[0, 2]);
        Assert.True(float.IsNaN(floor[1, 0]));
        Assert.Equal(-2f, floor[1, 1]);
        Assert.Equal(2f, floor[1, 2]);
        var empty = new DenseTensor<float>(new[] { 0, 2 });
        Assert.Equal(0, CPUExecutionProvider.Sigmoid(empty, null).Outputs[0].Length);
        Assert.Equal(0, CPUExecutionProvider.Floor(empty, null).Outputs[0].Length);
        var scalar = new DenseTensor<bool>(new[] { true }, Array.Empty<int>());
        var mask = new DenseTensor<bool>(new[] { 0, 2 });
        Assert.Equal(0, CPUExecutionProvider.And(mask, scalar, null).Outputs[0].Length);
        Assert.Equal(0, CPUExecutionProvider.Not(scalar, null).Outputs[0].Rank);
        Assert.False(((Tensor<bool>)CPUExecutionProvider.Not(scalar, null).Outputs[0]).GetValue(0));
    }
}
