using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Pins the documented CastOps policy against ORT 1.29 probe values:
/// fractional floats truncate toward zero, integer narrowing wraps modulo
/// the target width, float to signed-integer overflow and NaN saturate to
/// the minimum value, out-of-int64-range floats saturate unsigned targets,
/// and bools convert by nonzero testing.
/// </summary>
public class CastBoundaryTests
{
    static ITensor CastTo(ITensor input, TensorElementType target)
    {
        var result = CPU.Cast(input, target, null);
        Assert.Equal(OpStatus.Success, result.Status);
        return result.Outputs![0];
    }

    [Fact]
    public void HalfSource_RejectedCleanly()
    {
        // ORT computes half casts, but half kernels are out of scope, so
        // the provider fails descriptively instead of throwing from
        // CastOps (which has no half source handler).
        var h = DenseTensor<Half>.OfValues(new Half[] { (Half)1.5f });
        Assert.Equal(OpStatus.Failure, CPU.Cast(h, TensorElementType.Float, null).Status);
    }

    [Fact]
    public void EmptyInput_YieldsEmpty()
    {
        // ORT 1.29: casting zero elements yields zero elements, typed.
        var y = (Tensor<int>)CastTo(DenseTensor<float>.OfShape(0), TensorElementType.Int32);
        Assert.Equal(new int[] { 0 }, y.Dimensions.ToArray());
        Assert.Empty(y.ToArray());
    }

    [Fact]
    public void Float_TruncatesTowardZero()
    {
        var output = (Tensor<int>)CastTo(DenseTensor<float>.OfValues(new float[] { 1.9f, -1.9f, 2.5f, -2.5f, 0f, -0.0f }), TensorElementType.Int32);
        Assert.Equal(new int[] { 1, -1, 2, -2, 0, 0 }, output.ToArray());
    }

    [Fact]
    public void FloatOverflowAndNaN_SaturateSignedMin()
    {
        var i32 = (Tensor<int>)CastTo(DenseTensor<float>.OfValues(new float[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity, 1e20f, -1e20f, 2147483648f, -2147483649f }), TensorElementType.Int32);
        Assert.Equal(new int[] { int.MinValue, int.MinValue, int.MinValue, int.MinValue, int.MinValue, int.MinValue, int.MinValue }, i32.ToArray());
        var i64 = (Tensor<long>)CastTo(DenseTensor<double>.OfValues(new double[] { double.NaN, double.PositiveInfinity, 9.3e18, -9.3e18 }), TensorElementType.Int64);
        Assert.Equal(new long[] { long.MinValue, long.MinValue, long.MinValue, long.MinValue }, i64.ToArray());
        var i64edge = (Tensor<long>)CastTo(DenseTensor<double>.OfValues(new double[] { 9223372036854775808.0, -9223372036854775808.0, 9223372036854774784.0 }), TensorElementType.Int64);
        Assert.Equal(new long[] { long.MinValue, long.MinValue, 9223372036854774784L }, i64edge.ToArray());
        var i32edge = (Tensor<int>)CastTo(DenseTensor<double>.OfValues(new double[] { 2147483648.0, 2147483647.5 }), TensorElementType.Int32);
        Assert.Equal(new int[] { int.MinValue, 2147483647 }, i32edge.ToArray());
    }

    [Fact]
    public void FloatOutsideInt64Range_SaturatesUnsigned()
    {
        var u32 = (Tensor<uint>)CastTo(DenseTensor<float>.OfValues(new float[] { float.NaN, float.PositiveInfinity, 1e20f, -1e20f, 9223372036854775808f, -9223372036854775808f }), TensorElementType.UInt32);
        Assert.Equal(new uint[] { 0u, 0u, 0u, 0u, 0u, 0u }, u32.ToArray());
        var u64out = (Tensor<ulong>)CastTo(DenseTensor<float>.OfValues(new float[] { float.NaN, float.PositiveInfinity, 1e20f, -1e20f }), TensorElementType.UInt64);
        Assert.Equal(new ulong[] { 9223372036854775808UL, 9223372036854775808UL, 9223372036854775808UL, 9223372036854775808UL }, u64out.ToArray());
        var u64big = (Tensor<ulong>)CastTo(DenseTensor<double>.OfValues(new double[] { 18446744073709551616.0 }), TensorElementType.UInt64);
        Assert.Equal(new ulong[] { 9223372036854775808UL }, u64big.ToArray());
    }

    [Fact]
    public void DoubleOutsideRange_SaturatesUnsigned32()
    {
        // ORT 1.29: like the float twin, out-of-range doubles and NaN
        // saturate uint32 to 0 ([nan, inf, 1e300, -1e300, 1.9] ->
        // [0, 0, 0, 0, 1]).
        var u32 = (Tensor<uint>)CastTo(DenseTensor<double>.OfValues(new double[] { double.NaN, double.PositiveInfinity, 1e300, -1e300, 1.9 }), TensorElementType.UInt32);
        Assert.Equal(new uint[] { 0u, 0u, 0u, 0u, 1u }, u32.ToArray());
    }

    [Fact]
    public void FloatInsideInt64Range_WrapsUnsigned()
    {
        var u32 = (Tensor<uint>)CastTo(DenseTensor<float>.OfValues(new float[] { -1.5f, 1.9f, 5000000000f, -5000000000f, 4294967296f, -0.5f }), TensorElementType.UInt32);
        Assert.Equal(new uint[] { 4294967295u, 1u, 705032704u, 3589934592u, 0u, 0u }, u32.ToArray());
        var u64 = (Tensor<ulong>)CastTo(DenseTensor<float>.OfValues(new float[] { -1.5f, 1.9f, -2.0f }), TensorElementType.UInt64);
        Assert.Equal(new ulong[] { 18446744073709551615UL, 1UL, 18446744073709551614UL }, u64.ToArray());
    }

    [Fact]
    public void Float_ConvertsBoolByNonzeroTest()
    {
        var output = (Tensor<bool>)CastTo(DenseTensor<float>.OfValues(new float[] { 0f, -0.0f, 1.5f, -2.5f, float.NaN }), TensorElementType.Bool);
        Assert.Equal(new bool[] { false, false, true, true, true }, output.ToArray());
    }

    [Fact]
    public void Int64ToInt32_WrapsModuloWidth()
    {
        var output = (Tensor<int>)CastTo(DenseTensor<long>.OfValues(new long[] { 2147483647L, 2147483648L, -2147483649L, 4294967297L }), TensorElementType.Int32);
        Assert.Equal(new int[] { 2147483647, -2147483648, 2147483647, 1 }, output.ToArray());
    }

    [Fact]
    public void UnsignedSource_Converts()
    {
        // ORT 1.29: max-u32 rounds up to 2^32 in float, same-width
        // uint-to-signed casts wrap (not saturate), nonzero tests true,
        // max-u64 is exactly 2^64 in double, and u32 widens to u64.
        var f32 = (Tensor<float>)CastTo(DenseTensor<uint>.OfValues(new uint[] { 4294967295u, 7u }), TensorElementType.Float);
        Assert.Equal(new float[] { 4294967296f, 7f }, f32.ToArray());
        var i32 = (Tensor<int>)CastTo(DenseTensor<uint>.OfValues(new uint[] { 4294967295u, 7u }), TensorElementType.Int32);
        Assert.Equal(new int[] { -1, 7 }, i32.ToArray());
        var b32 = (Tensor<bool>)CastTo(DenseTensor<uint>.OfValues(new uint[] { 0u, 5u }), TensorElementType.Bool);
        Assert.Equal(new bool[] { false, true }, b32.ToArray());
        var f64 = (Tensor<double>)CastTo(DenseTensor<ulong>.OfValues(new ulong[] { 18446744073709551615ul, 7ul }), TensorElementType.Double);
        Assert.Equal(new double[] { 1.8446744073709552e19, 7.0 }, f64.ToArray());
        var i64 = (Tensor<long>)CastTo(DenseTensor<ulong>.OfValues(new ulong[] { 18446744073709551615ul, 7ul }), TensorElementType.Int64);
        Assert.Equal(new long[] { -1L, 7L }, i64.ToArray());
        var w64 = (Tensor<ulong>)CastTo(DenseTensor<uint>.OfValues(new uint[] { 4294967295u }), TensorElementType.UInt64);
        Assert.Equal(new ulong[] { 4294967295ul }, w64.ToArray());
    }

    [Fact]
    public void Sub32BoolSources_Widen()
    {
        // ORT 1.29: sub-32 and bool sources widen exactly (no rounding).
        var i8 = (Tensor<int>)CastTo(DenseTensor<sbyte>.OfValues(new sbyte[] { -128, -1, 0, 127 }), TensorElementType.Int32);
        Assert.Equal(new int[] { -128, -1, 0, 127 }, i8.ToArray());
        var u8 = (Tensor<int>)CastTo(DenseTensor<byte>.OfValues(new byte[] { 0, 255 }), TensorElementType.Int32);
        Assert.Equal(new int[] { 0, 255 }, u8.ToArray());
        var i16 = (Tensor<int>)CastTo(DenseTensor<short>.OfValues(new short[] { -32768, 30000 }), TensorElementType.Int32);
        Assert.Equal(new int[] { -32768, 30000 }, i16.ToArray());
        var u16 = (Tensor<int>)CastTo(DenseTensor<ushort>.OfValues(new ushort[] { 0, 65535 }), TensorElementType.Int32);
        Assert.Equal(new int[] { 0, 65535 }, u16.ToArray());
        var b = (Tensor<int>)CastTo(DenseTensor<bool>.OfValues(new bool[] { false, true }), TensorElementType.Int32);
        Assert.Equal(new int[] { 0, 1 }, b.ToArray());
        var f = (Tensor<float>)CastTo(DenseTensor<sbyte>.OfValues(new sbyte[] { -128, 100 }), TensorElementType.Float);
        Assert.Equal(new float[] { -128f, 100f }, f.ToArray());
    }
}
