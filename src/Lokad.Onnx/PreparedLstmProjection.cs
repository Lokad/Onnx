namespace Lokad.Onnx;

using System;
using System.Numerics;
using System.Runtime.InteropServices;

internal static class PreparedLstmProjection
{
    // Prepared arrays stay within one process; Vector.Count is fixed for its lifetime.
    // Keep the original reader's four accumulators and arithmetic.
    internal static int ColumnsPerBlock => 4 * Vector<float>.Count;

    internal static void Multiply(ReadOnlySpan<float> input, ReadOnlySpan<float> panel, Span<float> output)
    {
        if (panel.Length != checked(input.Length * output.Length))
            throw new ArgumentException("The projection panel does not match its input and output dimensions.");
        int width = Vector<float>.Count, block = ColumnsPerBlock, o = 0;
        ref float weights = ref MemoryMarshal.GetReference(panel);
        ref float destination = ref MemoryMarshal.GetReference(output);
        if (Vector.IsHardwareAccelerated)
        {
            for (; o <= output.Length - block; o += block)
            {
                var a = Vector<float>.Zero; var b = a; var c = a; var d = a;
                for (int k = 0; k < input.Length; k++)
                {
                    var x = new Vector<float>(input[k]);
                    nuint row = (nuint)(o * input.Length + k * block);
                    a = Vector.Add(a, Vector.Multiply(x, Vector.LoadUnsafe(ref weights, row)));
                    b = Vector.Add(b, Vector.Multiply(x, Vector.LoadUnsafe(ref weights, row + (nuint)width)));
                    c = Vector.Add(c, Vector.Multiply(x, Vector.LoadUnsafe(ref weights, row + (nuint)(2 * width))));
                    d = Vector.Add(d, Vector.Multiply(x, Vector.LoadUnsafe(ref weights, row + (nuint)(3 * width))));
                }
                a.StoreUnsafe(ref destination, (nuint)o);
                b.StoreUnsafe(ref destination, (nuint)(o + width));
                c.StoreUnsafe(ref destination, (nuint)(o + 2 * width));
                d.StoreUnsafe(ref destination, (nuint)(o + 3 * width));
            }
        }
        for (; o < output.Length; o++)
        {
            int group = o / block * block;
            int columns = Math.Min(block, output.Length - group);
            float value = 0f;
            for (int k = 0; k < input.Length; k++)
                value += input[k] * panel[group * input.Length + k * columns + o - group];
            output[o] = value;
        }
    }
}
