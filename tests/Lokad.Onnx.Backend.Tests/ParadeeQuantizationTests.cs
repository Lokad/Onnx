namespace Lokad.Onnx.Backend.Tests;

public class ParadeeQuantizationTests
{
    static Tensor<T> Output<T>(OpResult result) where T : unmanaged
    {
        Assert.True(result.Status == OpStatus.Success, result.Message);
        return Assert.IsAssignableFrom<Tensor<T>>(Assert.Single(result.Outputs));
    }

    [Fact]
    public void HalfStorageWideningPreservesSubnormalsSignedZeroAndSpecialValues()
    {
        var values = new[] { BitConverter.UInt16BitsToHalf(0), BitConverter.UInt16BitsToHalf(0x8000),
            BitConverter.UInt16BitsToHalf(1), (Half)1.5f, Half.MaxValue, Half.PositiveInfinity, Half.NaN };
        var source = DenseTensor<Half>.OfValues(values);
        var actual = Output<float>(CPUExecutionProvider.Cast(source, TensorElementType.Float, null)).ToArray();
        Assert.Equal(new[] { 0f, -0f, 5.960464477539063e-8f, 1.5f, 65504f, float.PositiveInfinity }, actual.Take(6));
        Assert.Equal(unchecked((int)0x80000000), BitConverter.SingleToInt32Bits(actual[1]));
        Assert.True(float.IsNaN(actual[6]));
        var clone = Output<Half>(CPUExecutionProvider.Cast(source, TensorElementType.Float16, null));
        source.SetValue(3, (Half)7f);
        Assert.Equal((Half)1.5f, clone.GetValue(3));
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Cast(source, TensorElementType.Int32, null).Status);
    }

    [Fact]
    public void HalfWideningReadsLogicalStridedValues()
    {
        var source = new DenseTensor<Half>(new Half[] { (Half)1, (Half)2, (Half)3, (Half)4, (Half)5, (Half)6 }, new[] { 2, 3 });
        var view = Output<Half>(CPUExecutionProvider.Transpose(source, new[] { 1, 0 }, null, null));
        var widened = Output<double>(CPUExecutionProvider.Cast(view, TensorElementType.Double, null));
        Assert.Equal(new[] { 3, 2 }, widened.Dimensions.ToArray());
        Assert.Equal(new double[] { 1, 4, 2, 5, 3, 6 }, widened.ToArray());
    }

    [Fact]
    public void PerAxisDequantizationSubtractsZeroPointsWithoutByteOverflow()
    {
        var input = new DenseTensor<sbyte>(new sbyte[] { -128, -1, 127, 2, -20, 30 }, new[] { 2, 3 });
        var scale = DenseTensor<float>.OfValues(new[] { .5f, 2f });
        var zeros = DenseTensor<sbyte>.OfValues(new sbyte[] { -5, 10 });
        var result = Output<float>(CPUExecutionProvider.DequantizeLinear(input, scale, zeros, 0, null));
        Assert.Equal(new[] { 2, 3 }, result.Dimensions.ToArray());
        Assert.Equal(new[] { -61.5f, 2f, 66f, -16f, -60f, 40f }, result.ToArray());
        var unsigned = DenseTensor<byte>.OfValues(new byte[] { 0, 128, 255 });
        var scalar = new DenseTensor<float>(new[] { .5f }, Array.Empty<int>());
        var point = new DenseTensor<byte>(new byte[] { 128 }, Array.Empty<int>());
        Assert.Equal(new[] { -64f, 0f, 63.5f }, Output<float>(CPUExecutionProvider.DequantizeLinear(unsigned, scalar, point, null, null)).ToArray());
    }

    [Fact]
    public void DequantizationSupportsNegativeAxisAndRejectsInvalidParameters()
    {
        var input = new DenseTensor<sbyte>(new sbyte[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 });
        var scales = DenseTensor<float>.OfValues(new[] { 1f, 2f, 3f });
        Assert.Equal(new[] { 1f, 4f, 9f, 4f, 10f, 18f }, Output<float>(CPUExecutionProvider.DequantizeLinear(input, scales, null, -1, null)).ToArray());
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.DequantizeLinear(input, scales, null, 0, null).Status);
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.DequantizeLinear(input, DenseTensor<float>.OfValues(new[] { 0f }), null, 0, null).Status);
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.DequantizeLinear(input, scales, DenseTensor<sbyte>.OfValues(new sbyte[] { 0 }), 1, null).Status);
        var empty = new DenseTensor<sbyte>(new[] { 2, 0 });
        Assert.Empty(Output<float>(CPUExecutionProvider.DequantizeLinear(empty, DenseTensor<float>.OfValues(new[] { 1f, 2f }), null, 0, null)).ToArray());
    }

    [Theory]
    [InlineData(0, new long[] { 2, -1, 1, -2 })]
    [InlineData(1, new long[] { -1, -1, 1, 1 })]
    public void ModBroadcastsWithFloorOrTruncatingRemainders(int fmod, long[] expected)
    {
        var input = new DenseTensor<long>(new long[] { -7, 7 }, new[] { 2, 1 });
        var divisor = new DenseTensor<long>(new long[] { 3, -3 }, new[] { 1, 2 });
        Assert.Equal(expected, Output<long>(CPUExecutionProvider.Mod(input, divisor, fmod, null)).ToArray());
        Assert.Equal(new long[] { 0 }, Output<long>(CPUExecutionProvider.Mod(DenseTensor<long>.OfValues(new[] { long.MinValue }), DenseTensor<long>.OfValues(new long[] { -1 }), fmod, null)).ToArray());
    }

    [Fact]
    public void ModValidatesZeroAndFloatingPointRemainderFlag()
    {
        var input = DenseTensor<float>.OfValues(new[] { -7.5f, 7.5f });
        var divisor = DenseTensor<float>.OfValues(new[] { 3f });
        Assert.Equal(new[] { -1.5f, 1.5f }, Output<float>(CPUExecutionProvider.Mod(input, divisor, 1, null)).ToArray());
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Mod(input, divisor, 0, null).Status);
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.Mod(DenseTensor<int>.OfValues(new[] { 1 }), DenseTensor<int>.OfValues(new[] { 0 }), null, null).Status);
    }
}
