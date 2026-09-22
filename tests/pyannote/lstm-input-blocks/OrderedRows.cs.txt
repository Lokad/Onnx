using System;
using System.Numerics;
using System.Runtime.InteropServices;

namespace Lokad.Onnx;

public partial class CPUExecutionProvider
{
    // Four independent time rows share weights; each output retains its original reduction order.
    internal static void LstmProjectOrderedRows(ReadOnlySpan<float> input, int rowStart, int rowStride, int inputSize,
        ReadOnlySpan<float> panel, Span<float> output, int rows)
    {
        if ((uint)(rows - 1) >= 4u || inputSize < 0 || output.Length % rows != 0)
            throw new ArgumentException("Invalid ordered projection row dimensions.");
        int outputs = output.Length / rows;
        if (panel.Length != checked(inputSize * outputs))
            throw new ArgumentException("The projection panel does not match its input and output dimensions.");
        long last = (long)rowStart + (rows - 1L) * rowStride;
        if (Math.Min(rowStart, last) < 0 || Math.Max(rowStart, last) + inputSize > input.Length)
            throw new ArgumentException("Projection input rows exceed the supplied span.");
        if (input.Overlaps(output) || panel.Overlaps(output))
            throw new ArgumentException("Projection destinations must not alias operands.");
        if (rows < 4)
        {
            for (int r = 0; r < rows; r++)
                LstmProjectOrdered(input.Slice(rowStart + r * rowStride, inputSize), panel, output.Slice(r * outputs, outputs));
            return;
        }
        int width = Vector<float>.Count, block = 2 * width, o = 0;
        int vectorEnd = outputs / (4 * width) * (4 * width);
        int s0 = rowStart, s1 = rowStart + rowStride, s2 = rowStart + 2 * rowStride, s3 = rowStart + 3 * rowStride;
        ref float weights = ref MemoryMarshal.GetReference(panel);
        ref float destination = ref MemoryMarshal.GetReference(output);
        if (Vector.IsHardwareAccelerated)
        {
            for (; o <= vectorEnd - block; o += block)
            {
                var a0 = Vector<float>.Zero; var b0 = a0;
                var a1 = a0; var b1 = a0; var a2 = a0; var b2 = a0; var a3 = a0; var b3 = a0;
                for (int k = 0; k < inputSize; k++)
                {
                    nuint offset = (nuint)(k * outputs + o);
                    var w0 = Vector.LoadUnsafe(ref weights, offset);
                    var w1 = Vector.LoadUnsafe(ref weights, offset + (nuint)width);
                    var x0 = new Vector<float>(input[s0 + k]);
                    a0 = Vector.Add(a0, Vector.Multiply(x0, w0));
                    b0 = Vector.Add(b0, Vector.Multiply(x0, w1));
                    var x1 = new Vector<float>(input[s1 + k]);
                    a1 = Vector.Add(a1, Vector.Multiply(x1, w0));
                    b1 = Vector.Add(b1, Vector.Multiply(x1, w1));
                    var x2 = new Vector<float>(input[s2 + k]);
                    a2 = Vector.Add(a2, Vector.Multiply(x2, w0));
                    b2 = Vector.Add(b2, Vector.Multiply(x2, w1));
                    var x3 = new Vector<float>(input[s3 + k]);
                    a3 = Vector.Add(a3, Vector.Multiply(x3, w0));
                    b3 = Vector.Add(b3, Vector.Multiply(x3, w1));
                }
                a0.StoreUnsafe(ref destination, (nuint)o); b0.StoreUnsafe(ref destination, (nuint)(o + width));
                a1.StoreUnsafe(ref destination, (nuint)(outputs + o)); b1.StoreUnsafe(ref destination, (nuint)(outputs + o + width));
                a2.StoreUnsafe(ref destination, (nuint)(2 * outputs + o)); b2.StoreUnsafe(ref destination, (nuint)(2 * outputs + o + width));
                a3.StoreUnsafe(ref destination, (nuint)(3 * outputs + o)); b3.StoreUnsafe(ref destination, (nuint)(3 * outputs + o + width));
            }
        }
        for (; o < outputs; o++)
        {
            float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            for (int k = 0; k < inputSize; k++)
            {
                float weight = panel[k * outputs + o];
                a0 += input[s0 + k] * weight; a1 += input[s1 + k] * weight;
                a2 += input[s2 + k] * weight; a3 += input[s3 + k] * weight;
            }
            output[o] = a0; output[outputs + o] = a1; output[2 * outputs + o] = a2; output[3 * outputs + o] = a3;
        }
    }
}
