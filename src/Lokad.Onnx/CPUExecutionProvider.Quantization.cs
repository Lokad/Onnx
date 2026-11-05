namespace Lokad.Onnx;

using System;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using static OpResult;

public partial class CPUExecutionProvider
{
    public static OpResult DequantizeLinear(ITensor? input, ITensor? scale, ITensor? zeroPoint, int? axis, ExecutionOptions? options)
        => DequantizeLinear(input, scale, zeroPoint, axis, options, null);

    internal static OpResult DequantizeLinear(ITensor? input, ITensor? scale, ITensor? zeroPoint, int? axis, ExecutionOptions? options, TensorBufferPool? pool)
    {
        const OpType op = OpType.DequantizeLinear;
        if (input is null) return MissingInput(op, nameof(input));
        if (scale is null) return MissingInput(op, nameof(scale));
        var opts = (options ?? ExecutionOptions.Default).Validated();
        if (input.ElementType is not TensorElementType.Int8 and not TensorElementType.UInt8)
            return InputTypeNotSupported(op, nameof(input), input);
        if (scale.ElementType != TensorElementType.Float) return WrongInputType(op, nameof(scale), TensorElementType.Float, scale);
        if (scale.Rank > 1 || scale.Length == 0) return WrongInputShape(op, nameof(scale), scale, "Scale must be a scalar or nonempty channel vector.");
        var factors = ((Tensor<float>)scale).ToArray();
        foreach (float value in factors)
            if (!float.IsFinite(value) || value <= 0f) return Failure(op, "Dequantization scales must be finite and positive.");
        if (zeroPoint is not null)
        {
            if (zeroPoint.ElementType != input.ElementType) return WrongInputType(op, nameof(zeroPoint), input.ElementType, zeroPoint);
            if (zeroPoint.Rank > 1 || zeroPoint.Length != scale.Length)
                return WrongInputShape(op, nameof(zeroPoint), zeroPoint, "Zero point must match the scalar or channel scale.");
        }
        int dim = axis ?? 1, inner = 1;
        if (factors.Length > 1)
        {
            if (dim < 0) dim += input.Rank;
            if (dim < 0 || dim >= input.Rank) return Failure(op, "DequantizeLinear axis is outside the input rank.");
            if (factors.Length != input.Dims[dim]) return WrongInputShape(op, nameof(scale), scale, "Scale length must equal the quantized axis extent.");
            // Empty output requires no indexing, even when other extents are large.
            if (input.Length != 0)
                for (int d = dim + 1; d < input.Rank; d++) inner = checked(inner * input.Dims[d]);
        }
        return input.ElementType == TensorElementType.Int8
            ? Success(op, DequantizeValues((Tensor<sbyte>)input, factors, (Tensor<sbyte>?)zeroPoint, inner, opts.Tensor.UseSimd, pool))
            : Success(op, DequantizeValues((Tensor<byte>)input, factors, (Tensor<byte>?)zeroPoint, inner, opts.Tensor.UseSimd, pool));
    }

    static DenseTensor<float> DequantizeValues<T>(Tensor<T> input, float[] factors, Tensor<T>? zeroPoint, int inner, bool useSimd, TensorBufferPool? pool) where T : unmanaged, IBinaryInteger<T>
    {
        var dense = input.ToRowMajor();
        var values = dense.Buffer.Span.Slice(0, checked((int)input.Length));
        var zeros = zeroPoint?.ToArray();
        var dimensions = input.Dimensions.ToArray();
        var result = pool is null ? new DenseTensor<float>(dimensions)
            : new DenseTensor<float>(pool.Rent<float>(values.Length), dimensions);
        var target = result.Buffer.Span;
        int block = factors.Length == 1 ? values.Length : inner;
        int channel = 0;
        for (int start = 0; start < values.Length; start += block)
        {
            int zero = zeros is null ? 0 : int.CreateChecked(zeros[channel]);
            DequantizeBlock(values.Slice(start, block), target.Slice(start, block), factors[channel], zero, useSimd);
            if (++channel == factors.Length) channel = 0;
        }
        return result;
    }

    static void DequantizeBlock<T>(ReadOnlySpan<T> values, Span<float> target, float factor, int zero, bool useSimd)
        where T : unmanaged, IBinaryInteger<T>
    {
        int i = 0;
        if (useSimd && Vector.IsHardwareAccelerated && values.Length >= Vector<byte>.Count)
        {
            int width = Vector<byte>.Count;
            var scale = new Vector<float>(factor);
            var offset = new Vector<float>(zero);
            if (typeof(T) == typeof(sbyte))
            {
                var bytes = MemoryMarshal.Cast<T, sbyte>(values);
                for (; i <= values.Length - width; i += width)
                {
                    Vector.Widen(new Vector<sbyte>(bytes.Slice(i, width)), out var low, out var high);
                    Vector.Widen(low, out var a, out var b);
                    Vector.Widen(high, out var c, out var d);
                    StoreDequantized(Vector.ConvertToSingle(a), Vector.ConvertToSingle(b), Vector.ConvertToSingle(c), Vector.ConvertToSingle(d), target.Slice(i, width), scale, offset);
                }
            }
            else
            {
                var bytes = MemoryMarshal.Cast<T, byte>(values);
                for (; i <= values.Length - width; i += width)
                {
                    Vector.Widen(new Vector<byte>(bytes.Slice(i, width)), out var low, out var high);
                    Vector.Widen(low, out var a, out var b);
                    Vector.Widen(high, out var c, out var d);
                    StoreDequantized(Vector.ConvertToSingle(a), Vector.ConvertToSingle(b), Vector.ConvertToSingle(c), Vector.ConvertToSingle(d), target.Slice(i, width), scale, offset);
                }
            }
        }
        for (; i < values.Length; i++) target[i] = (int.CreateChecked(values[i]) - zero) * factor;
    }

    static void StoreDequantized(Vector<float> a, Vector<float> b, Vector<float> c, Vector<float> d,
        Span<float> target, Vector<float> scale, Vector<float> offset)
    {
        // Byte values and zero points are exact in float, including their signed difference.
        ((a - offset) * scale).CopyTo(target);
        ((b - offset) * scale).CopyTo(target.Slice(Vector<float>.Count));
        ((c - offset) * scale).CopyTo(target.Slice(2 * Vector<float>.Count));
        ((d - offset) * scale).CopyTo(target.Slice(3 * Vector<float>.Count));
    }

    public static OpResult Mod(ITensor? a, ITensor? b, int? fmod, ExecutionOptions? options)
    {
        const OpType op = OpType.Mod;
        if (a is null) return MissingInput(op, nameof(a));
        if (b is null) return MissingInput(op, nameof(b));
        if (a.ElementType != b.ElementType) return WrongInputType(op, nameof(b), a.ElementType, b);
        if (fmod is not null && fmod != 0 && fmod != 1) return AttributeNotSupported(op, nameof(fmod), fmod.ToString()!, "Expected 0 or 1.");
        var opts = (options ?? ExecutionOptions.Default).Validated();
        if ((a.ElementType is TensorElementType.Float or TensorElementType.Double) && fmod != 1)
            return AttributeNotSupported(op, nameof(fmod), (fmod ?? 0).ToString(), "Floating-point Mod requires fmod=1.");
        if (b is Tensor<int> integers && integers.Any(value => value == 0)
            || b is Tensor<long> longs && longs.Any(value => value == 0))
            return Failure(op, "Mod divisor cannot be zero.");
        return a switch
        {
            Tensor<int> t => Success(op, IntegerMod(t, (Tensor<int>)b, fmod == 1, opts)),
            Tensor<long> t => Success(op, IntegerMod(t, (Tensor<long>)b, fmod == 1, opts)),
            Tensor<float> t => Success(op, t.BroadcastApply((Tensor<float>)b, static (x, y) => x % y, opts.Tensor)),
            Tensor<double> t => Success(op, t.BroadcastApply((Tensor<double>)b, static (x, y) => x % y, opts.Tensor)),
            _ => InputTypeNotSupported(op, nameof(a), a),
        };
    }

    static Tensor<T> IntegerMod<T>(Tensor<T> a, Tensor<T> b, bool fmod, ExecutionOptions options) where T : unmanaged, IBinaryInteger<T>, ISignedNumber<T>
    {
        return a.BroadcastApply(b, (x, y) =>
        {
            T result = y == T.NegativeOne ? T.Zero : x % y;
            return !fmod && result != T.Zero && (result < T.Zero) != (y < T.Zero) ? result + y : result;
        }, options.Tensor);
    }
}
