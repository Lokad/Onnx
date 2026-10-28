namespace Lokad.Onnx;

using System;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

internal static class PreparedLstmProjection
{
    // Prepared arrays stay within one process; Vector.Count is fixed for its lifetime.
    // Keep the original reader's four accumulators and arithmetic.
    internal static int ColumnsPerBlock => 4 * Vector<float>.Count;

    internal static void Pack(ReadOnlySpan<float> source, Span<float> target, int inputs, int outputs, int directions)
    {
        if (inputs < 0 || outputs < 0 || directions < 0
            || source.Length != checked(inputs * outputs * directions) || target.Length != source.Length)
            throw new ArgumentException("Projection packing dimensions do not match the supplied spans.");
        int block = ColumnsPerBlock, length = checked(inputs * outputs);
        for (int d = 0; d < directions; d++)
        for (int o = 0; o < outputs; o += block)
        {
            int columns = Math.Min(block, outputs - o);
            for (int k = 0; k < inputs; k++)
            for (int lane = 0; lane < columns; lane++)
                target[d * length + o * inputs + k * columns + lane] = source[d * length + (o + lane) * inputs + k];
        }
    }

    internal static void Multiply(ReadOnlySpan<float> input, ReadOnlySpan<float> panel, Span<float> output, bool useIntrinsics = true)
    {
        if (panel.Length != checked(input.Length * output.Length))
            throw new ArgumentException("The projection panel does not match its input and output dimensions.");
        int width = Vector<float>.Count, block = ColumnsPerBlock, o = 0;
        ref float weights = ref MemoryMarshal.GetReference(panel);
        ref float destination = ref MemoryMarshal.GetReference(output);
        if (useIntrinsics && Vector512.IsHardwareAccelerated && block == 32)
        {
            // Consume two existing panels together; arithmetic within each
            // output keeps the ascending, separate multiply/add reduction.
            for (; o <= output.Length - 2 * block; o += 2 * block)
            {
                var a = Vector512<float>.Zero; var b = a; var c = a; var d = a;
                for (int k = 0; k < input.Length; k++)
                {
                    var x = Vector512.Create(input[k]);
                    nuint first = (nuint)(o * input.Length + k * block);
                    nuint second = first + (nuint)(block * input.Length);
                    a = Vector512.Add(a, Vector512.Multiply(x, Vector512.LoadUnsafe(ref weights, first)));
                    b = Vector512.Add(b, Vector512.Multiply(x, Vector512.LoadUnsafe(ref weights, first + 16)));
                    c = Vector512.Add(c, Vector512.Multiply(x, Vector512.LoadUnsafe(ref weights, second)));
                    d = Vector512.Add(d, Vector512.Multiply(x, Vector512.LoadUnsafe(ref weights, second + 16)));
                }
                a.StoreUnsafe(ref destination, (nuint)o); b.StoreUnsafe(ref destination, (nuint)(o + 16));
                c.StoreUnsafe(ref destination, (nuint)(o + block)); d.StoreUnsafe(ref destination, (nuint)(o + block + 16));
            }
            for (; o <= output.Length - block; o += block)
            {
                var a = Vector512<float>.Zero; var b = a;
                for (int k = 0; k < input.Length; k++)
                {
                    var x = Vector512.Create(input[k]);
                    nuint row = (nuint)(o * input.Length + k * block);
                    a = Vector512.Add(a, Vector512.Multiply(x, Vector512.LoadUnsafe(ref weights, row)));
                    b = Vector512.Add(b, Vector512.Multiply(x, Vector512.LoadUnsafe(ref weights, row + 16)));
                }
                a.StoreUnsafe(ref destination, (nuint)o);
                b.StoreUnsafe(ref destination, (nuint)(o + 16));
            }
        }
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
