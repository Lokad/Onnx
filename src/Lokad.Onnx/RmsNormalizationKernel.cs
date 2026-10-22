namespace Lokad.Onnx;

using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

// Preserve the six-operator arithmetic, including the qualified float-mean tree.
internal static class RmsNormalizationKernel
{
    internal static bool TryNormalize(ReadOnlySpan<float> input, ReadOnlySpan<float> scale, Span<float> output,
        int columns, float epsilon, float exponent, bool useSimd)
    {
        if (columns <= 0 || input.Length == 0 || input.Length % columns != 0 || scale.Length != columns
            || output.Length != input.Length || !float.IsFinite(epsilon) || epsilon <= 0 || exponent != -.5f
            || input.Overlaps(output) || scale.Overlaps(output)) return false;
        if (!AllFinite(input, useSimd) || !AllFinite(scale, useSimd)) return false;
        int rows = input.Length / columns;
        for (int row = 0; row < rows; row++)
        {
            int offset = row * columns;
            var values = input.Slice(offset, columns);
            float sum = FloatReduction.Sum(values, Math.Min((-offset) & 3, columns), 0f, true);
            Apply(values, scale, output.Slice(offset, columns), MathF.Pow(sum / columns + epsilon, exponent), useSimd);
        }
        return true;
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
