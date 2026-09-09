namespace Lokad.Onnx;

using System;

/// <summary>
/// Typed numeric conversions implementing ONNX Cast semantics without boxing.
/// </summary>
/// <remarks>
/// Policy, verified against the native ORT engine: fractional floats truncate
/// toward zero, integer narrowing wraps modulo the target width, float to
/// signed-integer overflow and NaN saturate to the minimum value, and bools
/// convert by nonzero testing. Unsigned 64-bit targets use an explicit
/// truncate-and-wrap helper because runtime float conversion saturates there.
/// </remarks>
internal static class CastOps
{
    public static ITensor Cast(ITensor source, TensorElementType target)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        if (source.ElementType == target) return source.Clone();
        switch (source.ElementType)
        {
            case TensorElementType.Bool: return FromBool((Tensor<bool>)source, target);
            case TensorElementType.Int8: return FromSByte((Tensor<sbyte>)source, target);
            case TensorElementType.UInt8: return FromByte((Tensor<byte>)source, target);
            case TensorElementType.Int16: return FromInt16((Tensor<short>)source, target);
            case TensorElementType.UInt16: return FromUInt16((Tensor<ushort>)source, target);
            case TensorElementType.Int32: return FromInt32((Tensor<int>)source, target);
            case TensorElementType.UInt32: return FromUInt32((Tensor<uint>)source, target);
            case TensorElementType.Int64: return FromInt64((Tensor<long>)source, target);
            case TensorElementType.UInt64: return FromUInt64((Tensor<ulong>)source, target);
            case TensorElementType.Float: return FromFloat((Tensor<float>)source, target);
            case TensorElementType.Double: return FromDouble((Tensor<double>)source, target);
            default: throw new NotSupportedException($"Cast from {source.ElementType} is not supported.");
        }
    }

    static DenseTensor<U> Convert<S, U>(Tensor<S> src, Func<S, U> f) where S : unmanaged where U : unmanaged
    {
        if (src is DenseTensor<S> d && IsStandardDense(d))
        {
            var dst = new DenseTensor<U>(src.Dimensions);
            var s = d.Buffer.Span;
            var o = dst.Buffer.Span;
            for (int i = 0; i < s.Length; i++) o[i] = f(s[i]);
            return dst;
        }
        var dims = src.Dimensions.ToArray();
        var dd = new DenseTensor<U>(dims);
        if (src.Length == 0) return dd;
        int rank = src.Rank;
        var coords = new int[rank];
        for (long linear = 0; linear < src.Length; linear++)
        {
            long rest = linear;
            for (int dim = rank - 1; dim >= 0; dim--) { int size = dims[dim]; coords[dim] = (int)(rest % size); rest /= size; }
            dd[coords] = f(src[coords]);
        }
        return dd;
    }

    static bool IsStandardDense<S>(DenseTensor<S> d) where S : unmanaged
    {
        if (d.IsReversedStride) return false;
        var expected = ArrayUtilities.GetStrides(d.Dimensions);
        var actual = d.Strides;
        if (actual.Length != expected.Length) return false;
        for (int i = 0; i < actual.Length; i++) if (actual[i] != expected[i]) return false;
        return (long)d.Buffer.Length == d.Length;
    }

    static uint ToUInt32(float v) => ToUInt32((double)v);

    static uint ToUInt32(double v)
    {
        if (double.IsNaN(v) || double.IsInfinity(v)) return 0u;
        // Outside int64 range the reference yields zero instead of fmod
        // garbage (ORT 1.29); truncation cannot cross these bounds.
        if (v >= 9223372036854775808.0 || v <= -9223372036854775808.0) return 0u;
        double m = Math.Truncate(v) % 4294967296.0;
        if (m < 0.0) m += 4294967296.0;
        if (!(m < 4294967296.0)) return 0u;
        return (uint)m;
    }

    static ulong ToUInt64(float v) => ToUInt64((double)v);

    static ulong ToUInt64(double v)
    {
        if (double.IsNaN(v) || double.IsInfinity(v)) return 9223372036854775808UL;
        // Finite values at or beyond 2^64 saturate like the x86 conversion
        // (ORT 1.29); the negative bound is stated explicitly instead of
        // relying on overflow conversion.
        if (v >= 18446744073709551616.0 || v <= -9223372036854775808.0) return 9223372036854775808UL;
        double t = Math.Truncate(v);
        if (t < 9223372036854775808.0) return unchecked((ulong)unchecked((long)t));
        if (t < 18446744073709551616.0) return unchecked((ulong)unchecked((long)(t - 9223372036854775808.0))) + 9223372036854775808UL;
        double m = t % 18446744073709551616.0;
        if (m < 0.0) m += 18446744073709551616.0;
        if (!(m < 18446744073709551616.0)) return 0UL;
        return (ulong)m;
    }

    // The bound is 2^31, not (float)int.MaxValue which rounds up to 2^31: exactly 2^31
    // must saturate (ORT 1.29) instead of falling into platform overflow conversion.
    static int ToInt32(float v) => !(v < 2147483648f) || v < -2147483648f ? int.MinValue : unchecked((int)v);

    static int ToInt32(double v) => !(v < 2147483648.0) || v < -2147483648.0 ? int.MinValue : unchecked((int)v);

    static long ToInt64(float v) => !(v < 9223372036854775808f) || v < -9223372036854775808f ? long.MinValue : unchecked((long)v);

    static long ToInt64(double v) => !(v < 9223372036854775808.0) || v < -9223372036854775808.0 ? long.MinValue : unchecked((long)v);

    static NotSupportedException NoTarget(TensorElementType target) =>
        new NotSupportedException($"Cast to {target} is not supported.");

    static ITensor FromBool(Tensor<bool> src, TensorElementType target)
    {
        switch (target)
        {
            case TensorElementType.Int8: return Convert<bool, sbyte>(src, v => v ? (sbyte)1 : (sbyte)0);
            case TensorElementType.UInt8: return Convert<bool, byte>(src, v => v ? (byte)1 : (byte)0);
            case TensorElementType.Int16: return Convert<bool, short>(src, v => v ? (short)1 : (short)0);
            case TensorElementType.UInt16: return Convert<bool, ushort>(src, v => v ? (ushort)1 : (ushort)0);
            case TensorElementType.Int32: return Convert<bool, int>(src, v => v ? 1 : 0);
            case TensorElementType.UInt32: return Convert<bool, uint>(src, v => v ? 1u : 0u);
            case TensorElementType.Int64: return Convert<bool, long>(src, v => v ? 1L : 0L);
            case TensorElementType.UInt64: return Convert<bool, ulong>(src, v => v ? 1UL : 0UL);
            case TensorElementType.Float: return Convert<bool, float>(src, v => v ? 1f : 0f);
            case TensorElementType.Double: return Convert<bool, double>(src, v => v ? 1.0 : 0.0);
            default: throw NoTarget(target);
        }
    }

    static ITensor FromSByte(Tensor<sbyte> src, TensorElementType target)
    {
        switch (target)
        {
            case TensorElementType.Bool: return Convert<sbyte, bool>(src, v => v != 0);
            case TensorElementType.UInt8: return Convert<sbyte, byte>(src, v => unchecked((byte)v));
            case TensorElementType.Int16: return Convert<sbyte, short>(src, v => v);
            case TensorElementType.UInt16: return Convert<sbyte, ushort>(src, v => unchecked((ushort)v));
            case TensorElementType.Int32: return Convert<sbyte, int>(src, v => v);
            case TensorElementType.UInt32: return Convert<sbyte, uint>(src, v => unchecked((uint)v));
            case TensorElementType.Int64: return Convert<sbyte, long>(src, v => v);
            case TensorElementType.UInt64: return Convert<sbyte, ulong>(src, v => unchecked((ulong)v));
            case TensorElementType.Float: return Convert<sbyte, float>(src, v => v);
            case TensorElementType.Double: return Convert<sbyte, double>(src, v => v);
            default: throw NoTarget(target);
        }
    }

    static ITensor FromByte(Tensor<byte> src, TensorElementType target)
    {
        switch (target)
        {
            case TensorElementType.Bool: return Convert<byte, bool>(src, v => v != 0);
            case TensorElementType.Int8: return Convert<byte, sbyte>(src, v => unchecked((sbyte)v));
            case TensorElementType.Int16: return Convert<byte, short>(src, v => v);
            case TensorElementType.UInt16: return Convert<byte, ushort>(src, v => v);
            case TensorElementType.Int32: return Convert<byte, int>(src, v => v);
            case TensorElementType.UInt32: return Convert<byte, uint>(src, v => v);
            case TensorElementType.Int64: return Convert<byte, long>(src, v => v);
            case TensorElementType.UInt64: return Convert<byte, ulong>(src, v => v);
            case TensorElementType.Float: return Convert<byte, float>(src, v => v);
            case TensorElementType.Double: return Convert<byte, double>(src, v => v);
            default: throw NoTarget(target);
        }
    }

    static ITensor FromInt16(Tensor<short> src, TensorElementType target)
    {
        switch (target)
        {
            case TensorElementType.Bool: return Convert<short, bool>(src, v => v != 0);
            case TensorElementType.Int8: return Convert<short, sbyte>(src, v => unchecked((sbyte)v));
            case TensorElementType.UInt8: return Convert<short, byte>(src, v => unchecked((byte)v));
            case TensorElementType.UInt16: return Convert<short, ushort>(src, v => unchecked((ushort)v));
            case TensorElementType.Int32: return Convert<short, int>(src, v => v);
            case TensorElementType.UInt32: return Convert<short, uint>(src, v => unchecked((uint)v));
            case TensorElementType.Int64: return Convert<short, long>(src, v => v);
            case TensorElementType.UInt64: return Convert<short, ulong>(src, v => unchecked((ulong)v));
            case TensorElementType.Float: return Convert<short, float>(src, v => v);
            case TensorElementType.Double: return Convert<short, double>(src, v => v);
            default: throw NoTarget(target);
        }
    }

    static ITensor FromUInt16(Tensor<ushort> src, TensorElementType target)
    {
        switch (target)
        {
            case TensorElementType.Bool: return Convert<ushort, bool>(src, v => v != 0);
            case TensorElementType.Int8: return Convert<ushort, sbyte>(src, v => unchecked((sbyte)v));
            case TensorElementType.UInt8: return Convert<ushort, byte>(src, v => unchecked((byte)v));
            case TensorElementType.Int16: return Convert<ushort, short>(src, v => unchecked((short)v));
            case TensorElementType.Int32: return Convert<ushort, int>(src, v => v);
            case TensorElementType.UInt32: return Convert<ushort, uint>(src, v => v);
            case TensorElementType.Int64: return Convert<ushort, long>(src, v => v);
            case TensorElementType.UInt64: return Convert<ushort, ulong>(src, v => v);
            case TensorElementType.Float: return Convert<ushort, float>(src, v => v);
            case TensorElementType.Double: return Convert<ushort, double>(src, v => v);
            default: throw NoTarget(target);
        }
    }

    static ITensor FromInt32(Tensor<int> src, TensorElementType target)
    {
        switch (target)
        {
            case TensorElementType.Bool: return Convert<int, bool>(src, v => v != 0);
            case TensorElementType.Int8: return Convert<int, sbyte>(src, v => unchecked((sbyte)v));
            case TensorElementType.UInt8: return Convert<int, byte>(src, v => unchecked((byte)v));
            case TensorElementType.Int16: return Convert<int, short>(src, v => unchecked((short)v));
            case TensorElementType.UInt16: return Convert<int, ushort>(src, v => unchecked((ushort)v));
            case TensorElementType.UInt32: return Convert<int, uint>(src, v => unchecked((uint)v));
            case TensorElementType.Int64: return Convert<int, long>(src, v => v);
            case TensorElementType.UInt64: return Convert<int, ulong>(src, v => unchecked((ulong)v));
            case TensorElementType.Float: return Convert<int, float>(src, v => v);
            case TensorElementType.Double: return Convert<int, double>(src, v => v);
            default: throw NoTarget(target);
        }
    }

    static ITensor FromUInt32(Tensor<uint> src, TensorElementType target)
    {
        switch (target)
        {
            case TensorElementType.Bool: return Convert<uint, bool>(src, v => v != 0);
            case TensorElementType.Int8: return Convert<uint, sbyte>(src, v => unchecked((sbyte)v));
            case TensorElementType.UInt8: return Convert<uint, byte>(src, v => unchecked((byte)v));
            case TensorElementType.Int16: return Convert<uint, short>(src, v => unchecked((short)v));
            case TensorElementType.UInt16: return Convert<uint, ushort>(src, v => unchecked((ushort)v));
            case TensorElementType.Int32: return Convert<uint, int>(src, v => unchecked((int)v));
            case TensorElementType.Int64: return Convert<uint, long>(src, v => v);
            case TensorElementType.UInt64: return Convert<uint, ulong>(src, v => v);
            case TensorElementType.Float: return Convert<uint, float>(src, v => v);
            case TensorElementType.Double: return Convert<uint, double>(src, v => v);
            default: throw NoTarget(target);
        }
    }

    static ITensor FromInt64(Tensor<long> src, TensorElementType target)
    {
        switch (target)
        {
            case TensorElementType.Bool: return Convert<long, bool>(src, v => v != 0L);
            case TensorElementType.Int8: return Convert<long, sbyte>(src, v => unchecked((sbyte)v));
            case TensorElementType.UInt8: return Convert<long, byte>(src, v => unchecked((byte)v));
            case TensorElementType.Int16: return Convert<long, short>(src, v => unchecked((short)v));
            case TensorElementType.UInt16: return Convert<long, ushort>(src, v => unchecked((ushort)v));
            case TensorElementType.Int32: return Convert<long, int>(src, v => unchecked((int)v));
            case TensorElementType.UInt32: return Convert<long, uint>(src, v => unchecked((uint)v));
            case TensorElementType.UInt64: return Convert<long, ulong>(src, v => unchecked((ulong)v));
            case TensorElementType.Float: return Convert<long, float>(src, v => v);
            case TensorElementType.Double: return Convert<long, double>(src, v => v);
            default: throw NoTarget(target);
        }
    }

    static ITensor FromUInt64(Tensor<ulong> src, TensorElementType target)
    {
        switch (target)
        {
            case TensorElementType.Bool: return Convert<ulong, bool>(src, v => v != 0UL);
            case TensorElementType.Int8: return Convert<ulong, sbyte>(src, v => unchecked((sbyte)v));
            case TensorElementType.UInt8: return Convert<ulong, byte>(src, v => unchecked((byte)v));
            case TensorElementType.Int16: return Convert<ulong, short>(src, v => unchecked((short)v));
            case TensorElementType.UInt16: return Convert<ulong, ushort>(src, v => unchecked((ushort)v));
            case TensorElementType.Int32: return Convert<ulong, int>(src, v => unchecked((int)v));
            case TensorElementType.UInt32: return Convert<ulong, uint>(src, v => unchecked((uint)v));
            case TensorElementType.Int64: return Convert<ulong, long>(src, v => unchecked((long)v));
            case TensorElementType.Float: return Convert<ulong, float>(src, v => v);
            case TensorElementType.Double: return Convert<ulong, double>(src, v => v);
            default: throw NoTarget(target);
        }
    }

    static ITensor FromFloat(Tensor<float> src, TensorElementType target)
    {
        switch (target)
        {
            case TensorElementType.Bool: return Convert<float, bool>(src, v => v != 0f);
            case TensorElementType.Int8: return Convert<float, sbyte>(src, v => unchecked((sbyte)ToInt32(v)));
            case TensorElementType.UInt8: return Convert<float, byte>(src, v => unchecked((byte)ToUInt32(v)));
            case TensorElementType.Int16: return Convert<float, short>(src, v => unchecked((short)ToInt32(v)));
            case TensorElementType.UInt16: return Convert<float, ushort>(src, v => unchecked((ushort)ToUInt32(v)));
            case TensorElementType.Int32: return Convert<float, int>(src, v => ToInt32(v));
            case TensorElementType.UInt32: return Convert<float, uint>(src, v => ToUInt32(v));
            case TensorElementType.Int64: return Convert<float, long>(src, v => ToInt64(v));
            case TensorElementType.UInt64: return Convert<float, ulong>(src, v => ToUInt64(v));
            case TensorElementType.Double: return Convert<float, double>(src, v => (double)v);
            default: throw NoTarget(target);
        }
    }

    static ITensor FromDouble(Tensor<double> src, TensorElementType target)
    {
        switch (target)
        {
            case TensorElementType.Bool: return Convert<double, bool>(src, v => v != 0.0);
            case TensorElementType.Int8: return Convert<double, sbyte>(src, v => unchecked((sbyte)ToInt32(v)));
            case TensorElementType.UInt8: return Convert<double, byte>(src, v => unchecked((byte)ToUInt32(v)));
            case TensorElementType.Int16: return Convert<double, short>(src, v => unchecked((short)ToInt32(v)));
            case TensorElementType.UInt16: return Convert<double, ushort>(src, v => unchecked((ushort)ToUInt32(v)));
            case TensorElementType.Int32: return Convert<double, int>(src, v => ToInt32(v));
            case TensorElementType.UInt32: return Convert<double, uint>(src, v => ToUInt32(v));
            case TensorElementType.Int64: return Convert<double, long>(src, v => ToInt64(v));
            case TensorElementType.UInt64: return Convert<double, ulong>(src, v => ToUInt64(v));
            case TensorElementType.Float: return Convert<double, float>(src, v => unchecked((float)v));
            default: throw NoTarget(target);
        }
    }
}
