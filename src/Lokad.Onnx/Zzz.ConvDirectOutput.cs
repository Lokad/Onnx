using System;
using System.Buffers;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx;

public abstract partial class Tensor<T> where T : unmanaged
{
    internal static unsafe bool TryConvDirectOutput(ReadOnlySpan<float> weights,
        ReadOnlySpan<float> patch, Span<float> temporary, Span<float> destination,
        ReadOnlySpan<float> bias, bool hasBias, int rows, int reduction,
        int columns, int outputStride, TensorExecutionOptions options)
    {
        bool directWide = options.UseSimd && options.UseIntrinsics && Fma.IsSupported
            && AblationSwitches.EnablePackedAvx512Rows && Avx512F.IsSupported
            && rows is >= 64 and <= 1024 && (rows & 7) == 0
            && reduction is >= 64 and <= 4096 && columns is >= 16 and <= 128;
        if (!directWide && !CanUseConvPortableRows(rows, reduction, columns, options)) return false;
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
        // At most one panel has exactly the patch layout; borrow its pinned input.
        var packed = columns > 32 ? RentScratch<float>(reduction * columns, options) : null;
        try
        {
            int grouped = rows / 6 * 6;
            int remainder = rows - grouped;
            fixed (float* a = weights)
            fixed (float* b = patch)
            fixed (float* t = temporary)
            fixed (float* c = destination)
            fixed (float* p = packed)
            fixed (float* bi = bias)
            {
                float* panels = packed is null ? b : p;
                if (packed is not null) MathOps.PackPanelsB(reduction, columns, b, panels);
                // Only complete 12/8-row groups enter the strided zero-seeded
                // route. Logical packed columns and final row stride differ.
                if (directWide && MathOps.TryPackedAvx512Rows(rows, reduction, columns,
                    a, panels, c, outputStride, zeroSeed: true))
                {
                    if (hasBias)
                        for (int row = 0; row < rows; row++)
                        {
                            int col = 0;
                            var offset = Vector512.Create(bi[row]);
                            for (; col <= columns - 16; col += 16)
                            {
                                float* cell = c + row * outputStride + col;
                                *(Vector512<float>*)cell = Avx512F.Add(*(Vector512<float>*)cell, offset);
                            }
                            for (; col < columns; col++) c[row * outputStride + col] += bi[row];
                        }
                    return true;
                }
                temporary.Slice(0, remainder * columns).Clear();
                // Reuse the matrix dispatcher's wider consumer for the bounded
                // large-row, medium-reduction territory. Its accumulation order
                // is unchanged; the existing temporary provides contiguous rows.
                if (AblationSwitches.EnablePackedAvx512Rows && Avx512F.IsSupported
                    && rows >= 256 && reduction >= 256 && reduction <= 1024 && columns >= 32)
                {
                    temporary.Slice(0, rows * columns).Clear();
                    if (MathOps.TryPackedAvx512Rows(rows, reduction, columns, a, panels, t))
                    {
                        for (int row = 0; row < rows; row++)
                        for (int col = 0; col < columns; col++)
                            c[row * outputStride + col] = hasBias
                                ? t[row * columns + col] + bi[row] : t[row * columns + col];
                        return true;
                    }
                }
                ConvDirectOutput.Multiply(grouped, reduction, columns, a, panels, c,
                    outputStride, bi, hasBias);
                MathOps.mm_unsafe_vectorized_intrinsics_2x4packed_bump(remainder,
                    reduction, columns, a + grouped * reduction, panels, t);
                for (int row = 0; row < remainder; row++)
                {
                    float value = hasBias ? bi[grouped + row] : 0f;
                    for (int col = 0; col < columns; col++)
                        c[(grouped + row) * outputStride + col] = hasBias
                            ? t[row * columns + col] + value : t[row * columns + col];
                }
            }
        }
        finally { if (packed is not null) ArrayPool<float>.Shared.Return(packed); }
        return true;
    }
}
