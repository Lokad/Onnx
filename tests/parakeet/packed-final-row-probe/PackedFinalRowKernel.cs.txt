using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx;

internal static unsafe class PackedFinalRowKernel
{
    // Same reduction/FMA order as ShortWideMultiplyRemainder, with packed B addressing.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void Multiply(int n, int k, float* a, float* packed, float* c)
    {
        for (int column = 0; column < k; column += 32)
        {
            int width = Math.Min(32, k - column);
            float* panel = packed + column * n;
            float* output = c + column;
            for (int j = 0; j < n; j++)
            {
                float value = a[j];
                float* row = panel + j * width;
                var av = Vector256.Create(value);
                int ceiling = width / Vector256<float>.Count * Vector256<float>.Count;
                var bv = MemoryMarshal.Cast<float, Vector256<float>>(new Span<float>(row, width));
                var cv = MemoryMarshal.Cast<float, Vector256<float>>(new Span<float>(output, width));
                for (int v = 0; v < bv.Length; v++)
                    cv[v] = Fma.MultiplyAdd(bv[v], av, cv[v]);
                for (int tail = ceiling; tail < width; tail++)
                    output[tail] += value * row[tail];
            }
        }
    }
}
