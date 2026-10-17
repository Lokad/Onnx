namespace Lokad.Onnx;

using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

// Preserve the six-operator arithmetic while processing independent rows together.
internal static class RmsNormalizationKernel
{
    internal static bool TryNormalize(ReadOnlySpan<float> input, ReadOnlySpan<float> scale, Span<float> output,
        int columns, float epsilon, float exponent, bool useSimd)
    {
        if (columns <= 0 || input.Length == 0 || input.Length % columns != 0 || scale.Length != columns
            || output.Length != input.Length || !float.IsFinite(epsilon) || epsilon <= 0 || exponent != -.5f
            || input.Overlaps(output) || scale.Overlaps(output)) return false;
        if (!AllFinite(input, useSimd) || !AllFinite(scale, useSimd)) return false;
        int rows = input.Length / columns, row = 0;
        if (useSimd && Sse.IsSupported && columns >= 32)
        {
            ref float head = ref MemoryMarshal.GetReference(input);
            for (; row <= rows - 16; row += 16)
            {
                int start0 = row * columns, start1 = (row + 4) * columns;
                int start2 = (row + 8) * columns, start3 = (row + 12) * columns;
                var sums0 = Vector128<float>.Zero;
                var sums1 = Vector128<float>.Zero;
                var sums2 = Vector128<float>.Zero;
                var sums3 = Vector128<float>.Zero;
                int column = 0;
                for (; column <= columns - 4; column += 4)
                {
                    sums0 = AccumulateFour(ref head, start0 + column, columns, sums0);
                    sums1 = AccumulateFour(ref head, start1 + column, columns, sums1);
                    sums2 = AccumulateFour(ref head, start2 + column, columns, sums2);
                    sums3 = AccumulateFour(ref head, start3 + column, columns, sums3);
                }
                for (; column < columns; column++)
                {
                    sums0 = AccumulateTail(input, start0 + column, columns, sums0);
                    sums1 = AccumulateTail(input, start1 + column, columns, sums1);
                    sums2 = AccumulateTail(input, start2 + column, columns, sums2);
                    sums3 = AccumulateTail(input, start3 + column, columns, sums3);
                }
                ApplyFour(input, scale, output, start0, columns, sums0.GetElement(0), sums0.GetElement(1), sums0.GetElement(2), sums0.GetElement(3), epsilon, exponent);
                ApplyFour(input, scale, output, start1, columns, sums1.GetElement(0), sums1.GetElement(1), sums1.GetElement(2), sums1.GetElement(3), epsilon, exponent);
                ApplyFour(input, scale, output, start2, columns, sums2.GetElement(0), sums2.GetElement(1), sums2.GetElement(2), sums2.GetElement(3), epsilon, exponent);
                ApplyFour(input, scale, output, start3, columns, sums3.GetElement(0), sums3.GetElement(1), sums3.GetElement(2), sums3.GetElement(3), epsilon, exponent);
            }
        }
        if (useSimd && columns >= 32)
        {
            for (; row <= rows - 4; row += 4)
            {
                var x0 = input.Slice(row * columns, columns);
                var x1 = input.Slice((row + 1) * columns, columns);
                var x2 = input.Slice((row + 2) * columns, columns);
                var x3 = input.Slice((row + 3) * columns, columns);
                float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
                int column = 0;
                if (Sse.IsSupported)
                {
                    // Each SIMD lane belongs to a different row. Transpose four
                    // rounded squares so each row still adds columns in order.
                    var sums = Vector128<float>.Zero;
                    for (; column <= columns - 4; column += 4)
                        sums = AccumulateFour(ref MemoryMarshal.GetReference(input), row * columns + column, columns, sums);
                    sum0 = sums.GetElement(0);
                    sum1 = sums.GetElement(1);
                    sum2 = sums.GetElement(2);
                    sum3 = sums.GetElement(3);
                }
                for (; column < columns; column++)
                {
                    sum0 += MathF.FusedMultiplyAdd(x0[column], x0[column], 0f);
                    sum1 += MathF.FusedMultiplyAdd(x1[column], x1[column], 0f);
                    sum2 += MathF.FusedMultiplyAdd(x2[column], x2[column], 0f);
                    sum3 += MathF.FusedMultiplyAdd(x3[column], x3[column], 0f);
                }
                Apply(x0, scale, output.Slice(row * columns, columns), MathF.Pow(sum0 / columns + epsilon, exponent), useSimd);
                Apply(x1, scale, output.Slice((row + 1) * columns, columns), MathF.Pow(sum1 / columns + epsilon, exponent), useSimd);
                Apply(x2, scale, output.Slice((row + 2) * columns, columns), MathF.Pow(sum2 / columns + epsilon, exponent), useSimd);
                Apply(x3, scale, output.Slice((row + 3) * columns, columns), MathF.Pow(sum3 / columns + epsilon, exponent), useSimd);
            }
        }
        for (; row < rows; row++)
        {
            var values = input.Slice(row * columns, columns);
            float sum = 0;
            foreach (float value in values) sum += MathF.FusedMultiplyAdd(value, value, 0f);
            Apply(values, scale, output.Slice(row * columns, columns), MathF.Pow(sum / columns + epsilon, exponent), useSimd);
        }
        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector128<float> AccumulateFour(ref float input, int offset, int columns, Vector128<float> sums)
    {
        var a = Vector128.LoadUnsafe(ref input, (nuint)offset);
        var b = Vector128.LoadUnsafe(ref input, (nuint)(offset + columns));
        var c = Vector128.LoadUnsafe(ref input, (nuint)(offset + 2 * columns));
        var d = Vector128.LoadUnsafe(ref input, (nuint)(offset + 3 * columns));
        a = Sse.Multiply(a, a);
        b = Sse.Multiply(b, b);
        c = Sse.Multiply(c, c);
        d = Sse.Multiply(d, d);
        var ab0 = Sse.UnpackLow(a, b);
        var cd0 = Sse.UnpackLow(c, d);
        var ab1 = Sse.UnpackHigh(a, b);
        var cd1 = Sse.UnpackHigh(c, d);
        sums = Sse.Add(sums, Sse.MoveLowToHigh(ab0, cd0));
        sums = Sse.Add(sums, Sse.MoveHighToLow(cd0, ab0));
        sums = Sse.Add(sums, Sse.MoveLowToHigh(ab1, cd1));
        return Sse.Add(sums, Sse.MoveHighToLow(cd1, ab1));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector128<float> AccumulateTail(ReadOnlySpan<float> input, int offset, int columns, Vector128<float> sums)
    {
        var values = Vector128.Create(input[offset], input[offset + columns],
            input[offset + 2 * columns], input[offset + 3 * columns]);
        return Sse.Add(sums, Sse.Multiply(values, values));
    }

    static void ApplyFour(ReadOnlySpan<float> input, ReadOnlySpan<float> scale, Span<float> output,
        int start, int columns, float sum0, float sum1, float sum2, float sum3, float epsilon, float exponent)
    {
        Apply(input.Slice(start, columns), scale, output.Slice(start, columns), MathF.Pow(sum0 / columns + epsilon, exponent), true);
        start += columns;
        Apply(input.Slice(start, columns), scale, output.Slice(start, columns), MathF.Pow(sum1 / columns + epsilon, exponent), true);
        start += columns;
        Apply(input.Slice(start, columns), scale, output.Slice(start, columns), MathF.Pow(sum2 / columns + epsilon, exponent), true);
        start += columns;
        Apply(input.Slice(start, columns), scale, output.Slice(start, columns), MathF.Pow(sum3 / columns + epsilon, exponent), true);
    }

    static bool AllFinite(ReadOnlySpan<float> values, bool useSimd)
    {
        int i = 0, width = Vector<float>.Count;
        if (useSimd)
        {
            if (Avx512F.IsSupported)
            {
                var wide = MemoryMarshal.Cast<float, Vector512<float>>(values);
                var maximumWide = Vector512.Create(float.MaxValue);
                for (int j = 0; j < wide.Length; j++)
                    if (!Vector512.LessThanOrEqualAll(Vector512.Abs(wide[j]), maximumWide)) return false;
                i = wide.Length * 16;
            }
            var vectors = MemoryMarshal.Cast<float, Vector<float>>(values.Slice(i));
            var maximum = new Vector<float>(float.MaxValue);
            for (int j = 0; j < vectors.Length; j++)
                if (!Vector.LessThanOrEqualAll(Vector.Abs(vectors[j]), maximum)) return false;
            i += vectors.Length * width;
        }
        for (; i < values.Length; i++) if (!float.IsFinite(values[i])) return false;
        return true;
    }

    static void Apply(ReadOnlySpan<float> input, ReadOnlySpan<float> scale, Span<float> output, float inverse, bool useSimd)
    {
        int i = 0, width = Vector<float>.Count;
        if (useSimd)
        {
            if (Avx512F.IsSupported)
            {
                var wideInput = MemoryMarshal.Cast<float, Vector512<float>>(input);
                var wideScale = MemoryMarshal.Cast<float, Vector512<float>>(scale);
                var wideOutput = MemoryMarshal.Cast<float, Vector512<float>>(output);
                var wideFactor = Vector512.Create(inverse);
                for (int j = 0; j < wideInput.Length; j++)
                {
                    var normalized = Avx512F.Multiply(wideInput[j], wideFactor);
                    wideOutput[j] = Avx512F.Multiply(normalized, wideScale[j]);
                }
                i = wideInput.Length * 16;
            }
            var vectors = MemoryMarshal.Cast<float, Vector<float>>(input.Slice(i));
            var gamma = MemoryMarshal.Cast<float, Vector<float>>(scale.Slice(i));
            var destination = MemoryMarshal.Cast<float, Vector<float>>(output.Slice(i));
            var factor = new Vector<float>(inverse);
            for (int j = 0; j < vectors.Length; j++)
            {
                var normalized = vectors[j] * factor;
                destination[j] = normalized * gamma[j];
            }
            i += vectors.Length * width;
        }
        for (; i < input.Length; i++)
        {
            float normalized = input[i] * inverse;
            output[i] = normalized * scale[i];
        }
    }
}
