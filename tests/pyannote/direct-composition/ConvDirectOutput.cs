using System;
using System.Buffers;

namespace Lokad.Onnx;

public abstract partial class Tensor<T> where T : unmanaged
{
    internal static unsafe bool TryConvDirectOutput(ReadOnlySpan<float> weights,
        ReadOnlySpan<float> patch, Span<float> temporary, Span<float> destination,
        ReadOnlySpan<float> bias, bool hasBias, int rows, int reduction,
        int columns, int outputStride, TensorExecutionOptions options)
    {
        if (!CanUseConvPortableRows(rows, reduction, columns, options)) return false;
        long extent = (long)(rows - 1) * outputStride + columns;
        if ((long)rows * reduction > weights.Length || (long)reduction * columns > patch.Length
            || (long)rows * columns > temporary.Length || outputStride < columns
            || extent > destination.Length || (hasBias && bias.Length < rows))
            throw new ArgumentException("Convolution tile buffers do not match their dimensions.");
        if (weights.Overlaps(temporary) || weights.Overlaps(destination)
            || patch.Overlaps(temporary) || patch.Overlaps(destination)
            || temporary.Overlaps(destination)
            || (hasBias && (bias.Overlaps(temporary) || bias.Overlaps(destination))))
            throw new ArgumentException("Convolution output and scratch must not overlap inputs or each other.");
        var packed = RentScratch<float>(reduction * columns, options);
        try
        {
            int grouped = rows / 6 * 6;
            int remainder = rows - grouped;
            temporary.Slice(0, remainder * columns).Clear();
            fixed (float* a = weights)
            fixed (float* b = patch)
            fixed (float* t = temporary)
            fixed (float* c = destination)
            fixed (float* p = packed)
            fixed (float* bi = bias)
            {
                MathOps.PackPanelsB(reduction, columns, b, p);
                ConvDirectOutput.Multiply(grouped, reduction, columns, a, p, c,
                    outputStride, bi, hasBias);
                MathOps.mm_unsafe_vectorized_intrinsics_2x4packed_bump(remainder,
                    reduction, columns, a + grouped * reduction, p, t);
                for (int row = 0; row < remainder; row++)
                {
                    float value = hasBias ? bi[grouped + row] : 0f;
                    for (int col = 0; col < columns; col++)
                        c[(grouped + row) * outputStride + col] = hasBias
                            ? t[row * columns + col] + value : t[row * columns + col];
                }
            }
        }
        finally { ArrayPool<float>.Shared.Return(packed); }
        return true;
    }
}
