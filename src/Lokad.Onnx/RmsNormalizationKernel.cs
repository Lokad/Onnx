namespace Lokad.Onnx;

using System.Numerics;

// Internal prototype: dispatch is added only after its exact arithmetic is qualified.
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
        if (useSimd && columns >= 32)
        {
            for (; row <= rows - 4; row += 4)
            {
                var x0 = input.Slice(row * columns, columns);
                var x1 = input.Slice((row + 1) * columns, columns);
                var x2 = input.Slice((row + 2) * columns, columns);
                var x3 = input.Slice((row + 3) * columns, columns);
                float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
                for (int column = 0; column < columns; column++)
                {
                    // Explicit +0 rounds the square before the separate addition.
                    // It prevents contraction of square-plus-sum into one FMA.
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

    static bool AllFinite(ReadOnlySpan<float> values, bool useSimd)
    {
        int i = 0, width = Vector<float>.Count;
        if (useSimd)
        {
            var maximum = new Vector<float>(float.MaxValue);
            for (; i <= values.Length - width; i += width)
                if (!Vector.LessThanOrEqualAll(Vector.Abs(new Vector<float>(values.Slice(i, width))), maximum)) return false;
        }
        for (; i < values.Length; i++) if (!float.IsFinite(values[i])) return false;
        return true;
    }

    static void Apply(ReadOnlySpan<float> input, ReadOnlySpan<float> scale, Span<float> output, float inverse, bool useSimd)
    {
        int i = 0, width = Vector<float>.Count;
        if (useSimd)
        {
            var factor = new Vector<float>(inverse);
            for (; i <= input.Length - width; i += width)
            {
                var normalized = new Vector<float>(input.Slice(i, width)) * factor;
                (normalized * new Vector<float>(scale.Slice(i, width))).CopyTo(output.Slice(i, width));
            }
        }
        for (; i < input.Length; i++)
        {
            float normalized = input[i] * inverse;
            output[i] = normalized * scale[i];
        }
    }
}
