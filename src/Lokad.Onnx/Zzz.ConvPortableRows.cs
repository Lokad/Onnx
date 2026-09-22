using System;
using System.Buffers;
using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx;

public abstract partial class Tensor<T> where T : unmanaged
{
    // Keep existing generic dispatch, short reductions and experimental AVX-512
    // precedence. These are the same packing size limits as the generic route.
    internal static bool CanUseConvPortableRows(int rows, int reduction, int columns, TensorExecutionOptions options) =>
        options.UseSimd && options.UseIntrinsics && Fma.IsSupported
        && !AblationSwitches.EnablePackedAvx512Dynamic
        && rows >= 32 && (rows & 1) == 0 && rows % 3 != 0
        && reduction >= 64 && columns > 0
        && (long)reduction * columns <= 67108864
        && (rows >= 64 || (long)reduction * columns <= 65536);

    internal static unsafe bool TryConvPortableRows(ReadOnlySpan<float> weights,
        ReadOnlySpan<float> patch, Span<float> destination, int rows, int reduction,
        int columns, TensorExecutionOptions options)
    {
        if (!CanUseConvPortableRows(rows, reduction, columns, options)) return false;
        if ((long)rows * reduction > weights.Length || (long)reduction * columns > patch.Length
            || (long)rows * columns > destination.Length)
            throw new ArgumentException("Convolution tile buffers do not match their dimensions.");
        var packed = RentScratch<float>(reduction * columns, options);
        try
        {
            fixed (float* a = weights)
            fixed (float* b = patch)
            fixed (float* c = destination)
            fixed (float* p = packed)
            {
                destination.Slice(0, rows * columns).Clear();
                MathOps.PackPanelsB(reduction, columns, b, p);
                int grouped = rows / 6 * 6;
                MathOps.mm_unsafe_vectorized_intrinsics_3x4packed(grouped, reduction, columns, a, p, c);
                MathOps.mm_unsafe_vectorized_intrinsics_2x4packed_bump(rows - grouped, reduction, columns,
                    a + grouped * reduction, p, c + grouped * columns);
            }
        }
        finally { ArrayPool<float>.Shared.Return(packed); }
        return true;
    }
}
