using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// First ConstantOfShape coverage anywhere: default and explicit fills,
/// zero extents, scalar output, and descriptive rejections, against ORT
/// 1.29 probe values.
/// </summary>
public class ConstantOfShapeBoundaryTests
{
    static Tensor<float> Run(long[] dims, float[]? value)
    {
        DenseTensor<float>? v = value is null ? null : DenseTensor<float>.OfValues(value);
        var result = CPU.ConstantOfShape(DenseTensor<long>.OfValues(dims), v, null);
        Assert.Equal(OpStatus.Success, result.Status);
        return (Tensor<float>)result.Outputs![0];
    }

    [Fact]
    public void DefaultValue_FillsZeros()
    {
        // ORT 1.29: zeros(2, 2).
        var y = Run(new long[] { 2L, 2L }, null);
        Assert.Equal(new int[] { 2, 2 }, y.Dimensions.ToArray());
        Assert.Equal(new float[] { 0f, 0f, 0f, 0f }, y.ToArray());
    }

    [Fact]
    public void ExplicitValue_FillsConstant()
    {
        // ORT 1.29: [7, 7, 7].
        Assert.Equal(new float[] { 7f, 7f, 7f }, Run(new long[] { 3L }, new float[] { 7f }).ToArray());
    }

    [Fact]
    public void DoubleValue_FillsConstant()
    {
        // ORT 1.29: 2x2 of 3.25.
        var result = CPU.ConstantOfShape(DenseTensor<long>.OfValues(new long[] { 2L, 2L }), DenseTensor<double>.OfValues(new double[] { 3.25 }), null);
        Assert.Equal(OpStatus.Success, result.Status);
        var y = (Tensor<double>)result.Outputs![0];
        Assert.Equal(new int[] { 2, 2 }, y.Dimensions.ToArray());
        Assert.Equal(new double[] { 3.25, 3.25, 3.25, 3.25 }, y.ToArray());
    }

    [Fact]
    public void ZeroDim_ReturnsEmpty()
    {
        // ORT 1.29: shape (2, 0), no elements.
        var y = Run(new long[] { 2L, 0L }, null);
        Assert.Equal(new int[] { 2, 0 }, y.Dimensions.ToArray());
        Assert.Empty(y.ToArray());
    }

    [Fact]
    public void EmptyDims_ReturnsScalar()
    {
        // ORT 1.29: no dims at all -> rank-0 zero.
        var result = CPU.ConstantOfShape(DenseTensor<long>.OfShape(0), null, null);
        Assert.Equal(OpStatus.Success, result.Status);
        var y = (Tensor<float>)result.Outputs![0];
        Assert.Equal(new int[0], y.Dimensions.ToArray());
        Assert.Equal(new float[] { 0f }, y.ToArray());
    }

    [Fact]
    public void NegativeDims_FailsCleanly()
    {
        // ORT 1.29 fails the run; Lokad returns a descriptive Failure.
        var r = CPU.ConstantOfShape(DenseTensor<long>.OfValues(new long[] { 2L, -1L }), null, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("non-negative", r.Message ?? "");
    }

    [Fact]
    public void MultiElementValue_FailsCleanly()
    {
        var r = CPU.ConstantOfShape(
            DenseTensor<long>.OfValues(new long[] { 2L }),
            DenseTensor<float>.OfValues(new float[] { 1f, 2f }), null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("single element", r.Message ?? "");
    }

    [Fact]
    public void UnsignedValues_FillConstant()
    {
        // ORT 1.29: 2x2 of max-u32 and 1x2 of max-u64.
        var r32 = CPU.ConstantOfShape(DenseTensor<long>.OfValues(new long[] { 2L, 2L }), DenseTensor<uint>.OfValues(new uint[] { 4294967295u }), null);
        Assert.Equal(OpStatus.Success, r32.Status);
        Assert.Equal(new uint[] { 4294967295u, 4294967295u, 4294967295u, 4294967295u }, ((Tensor<uint>)r32.Outputs[0]).ToArray());
        var r64 = CPU.ConstantOfShape(DenseTensor<long>.OfValues(new long[] { 1L, 2L }), DenseTensor<ulong>.OfValues(new ulong[] { 18446744073709551615ul }), null);
        Assert.Equal(OpStatus.Success, r64.Status);
        Assert.Equal(new ulong[] { 18446744073709551615ul, 18446744073709551615ul }, ((Tensor<ulong>)r64.Outputs[0]).ToArray());
    }

    [Fact]
    public void IntBoolValues_FillConstant()
    {
        // ORT 1.29: [2] of int64 7 and [1, 2] of true.
        var ri = CPU.ConstantOfShape(DenseTensor<long>.OfValues(new long[] { 2L }), DenseTensor<long>.OfValues(new long[] { 7L }), null);
        Assert.Equal(OpStatus.Success, ri.Status);
        Assert.Equal(new long[] { 7L, 7L }, ((Tensor<long>)ri.Outputs[0]).ToArray());
        var rb = CPU.ConstantOfShape(DenseTensor<long>.OfValues(new long[] { 1L, 2L }), DenseTensor<bool>.OfValues(new bool[] { true }), null);
        Assert.Equal(OpStatus.Success, rb.Status);
        Assert.Equal(new bool[] { true, true }, ((Tensor<bool>)rb.Outputs[0]).ToArray());
    }

    [Fact]
    public void Sub32Values_FillConstant()
    {
        // ORT 1.29: [2] of int8 -5, uint8 200, int16 -30000 and [1, 2] of uint16 60000.
        var sh2 = DenseTensor<long>.OfValues(new long[] { 2L });
        var i8 = CPU.ConstantOfShape(sh2, DenseTensor<sbyte>.OfValues(new sbyte[] { -5 }), null);
        Assert.Equal(OpStatus.Success, i8.Status);
        Assert.Equal(new sbyte[] { -5, -5 }, ((Tensor<sbyte>)i8.Outputs[0]).ToArray());
        var u8 = CPU.ConstantOfShape(sh2, DenseTensor<byte>.OfValues(new byte[] { 200 }), null);
        Assert.Equal(OpStatus.Success, u8.Status);
        Assert.Equal(new byte[] { 200, 200 }, ((Tensor<byte>)u8.Outputs[0]).ToArray());
        var i16 = CPU.ConstantOfShape(sh2, DenseTensor<short>.OfValues(new short[] { -30000 }), null);
        Assert.Equal(OpStatus.Success, i16.Status);
        Assert.Equal(new short[] { -30000, -30000 }, ((Tensor<short>)i16.Outputs[0]).ToArray());
        var u16 = CPU.ConstantOfShape(DenseTensor<long>.OfValues(new long[] { 1L, 2L }), DenseTensor<ushort>.OfValues(new ushort[] { 60000 }), null);
        Assert.Equal(OpStatus.Success, u16.Status);
        Assert.Equal(new ushort[] { 60000, 60000 }, ((Tensor<ushort>)u16.Outputs[0]).ToArray());
    }
}
