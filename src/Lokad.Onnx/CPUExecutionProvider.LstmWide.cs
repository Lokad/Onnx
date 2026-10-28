using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx;

public partial class CPUExecutionProvider
{
    // The small-LSTM panel is output-major within each reduction step. Keep
    // separate multiply/add and ascending reduction order, including on AVX-512.
    // Return the processed extent so the existing portable path owns all tails.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    static int LstmProjectOrderedWide(ReadOnlySpan<float> input, ReadOnlySpan<float> panel, Span<float> output)
    {
        if (!Avx512F.IsSupported) return 0;
        ref float weights = ref MemoryMarshal.GetReference(panel);
        ref float target = ref MemoryMarshal.GetReference(output);
        int o = 0;
        for (; o <= output.Length - 64; o += 64)
        {
            var a = Vector512<float>.Zero; var b = a; var c = a; var d = a;
            for (int k = 0; k < input.Length; k++)
            {
                var x = Vector512.Create(input[k]);
                nuint row = (nuint)(k * output.Length + o);
                a = Vector512.Add(a, Vector512.Multiply(x, Vector512.LoadUnsafe(ref weights, row)));
                b = Vector512.Add(b, Vector512.Multiply(x, Vector512.LoadUnsafe(ref weights, row + 16)));
                c = Vector512.Add(c, Vector512.Multiply(x, Vector512.LoadUnsafe(ref weights, row + 32)));
                d = Vector512.Add(d, Vector512.Multiply(x, Vector512.LoadUnsafe(ref weights, row + 48)));
            }
            a.StoreUnsafe(ref target, (nuint)o);
            b.StoreUnsafe(ref target, (nuint)(o + 16));
            c.StoreUnsafe(ref target, (nuint)(o + 32));
            d.StoreUnsafe(ref target, (nuint)(o + 48));
        }
        return o;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    static int LstmProjectOrderedRowsWide(ReadOnlySpan<float> input, int s0, int s1, int s2, int s3, int inputSize,
        ReadOnlySpan<float> panel, Span<float> output, int outputs)
    {
        if (!Avx512F.IsSupported) return 0;
        ref float weights = ref MemoryMarshal.GetReference(panel);
        ref float target = ref MemoryMarshal.GetReference(output);
        int o = 0;
        for (; o <= outputs - 32; o += 32)
        {
            var a0 = Vector512<float>.Zero; var b0 = a0;
            var a1 = a0; var b1 = a0; var a2 = a0; var b2 = a0; var a3 = a0; var b3 = a0;
            for (int k = 0; k < inputSize; k++)
            {
                nuint row = (nuint)(k * outputs + o);
                var w0 = Vector512.LoadUnsafe(ref weights, row);
                var w1 = Vector512.LoadUnsafe(ref weights, row + 16);
                var x0 = Vector512.Create(input[s0 + k]);
                a0 = Vector512.Add(a0, Vector512.Multiply(x0, w0));
                b0 = Vector512.Add(b0, Vector512.Multiply(x0, w1));
                var x1 = Vector512.Create(input[s1 + k]);
                a1 = Vector512.Add(a1, Vector512.Multiply(x1, w0));
                b1 = Vector512.Add(b1, Vector512.Multiply(x1, w1));
                var x2 = Vector512.Create(input[s2 + k]);
                a2 = Vector512.Add(a2, Vector512.Multiply(x2, w0));
                b2 = Vector512.Add(b2, Vector512.Multiply(x2, w1));
                var x3 = Vector512.Create(input[s3 + k]);
                a3 = Vector512.Add(a3, Vector512.Multiply(x3, w0));
                b3 = Vector512.Add(b3, Vector512.Multiply(x3, w1));
            }
            a0.StoreUnsafe(ref target, (nuint)o); b0.StoreUnsafe(ref target, (nuint)(o + 16));
            a1.StoreUnsafe(ref target, (nuint)(outputs + o)); b1.StoreUnsafe(ref target, (nuint)(outputs + o + 16));
            a2.StoreUnsafe(ref target, (nuint)(2 * outputs + o)); b2.StoreUnsafe(ref target, (nuint)(2 * outputs + o + 16));
            a3.StoreUnsafe(ref target, (nuint)(3 * outputs + o)); b3.StoreUnsafe(ref target, (nuint)(3 * outputs + o + 16));
        }
        return o;
    }
}
