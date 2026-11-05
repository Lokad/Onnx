using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace Lokad.Onnx;

public partial class CPUExecutionProvider
{
    // Ordinary standard activations share dispatch for omitted/explicit triples.
    // Every path retains separate ascending bias additions and cell products.
    static void LstmDefaultCell(ReadOnlySpan<float> input, ReadOnlySpan<float> recurrent,
        ReadOnlySpan<float> bias, Span<float> cell, Span<float> hidden, Span<float> output, bool vectorize)
    {
        int size = cell.Length;
        int h = vectorize ? LstmVectorCell(input, recurrent, bias, cell, hidden, output) : 0;
        for (; h < size; h++)
        {
            float iPre = input[h] + recurrent[h] + bias[h] + bias[4 * size + h];
            float oPre = input[size + h] + recurrent[size + h] + bias[size + h] + bias[5 * size + h];
            float fPre = input[2 * size + h] + recurrent[2 * size + h] + bias[2 * size + h] + bias[6 * size + h];
            float gPre = input[3 * size + h] + recurrent[3 * size + h] + bias[3 * size + h] + bias[7 * size + h];
            float fv = 1f / (1f + MathF.Exp(-fPre));
            float gv = MathF.Tanh(gPre);
            float iv = 1f / (1f + MathF.Exp(-iPre));
            float cNew = fv * cell[h] + iv * gv;
            float hNew = (1f / (1f + MathF.Exp(-oPre))) * MathF.Tanh(cNew);
            cell[h] = cNew; hidden[h] = hNew; output[h] = hNew;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    static int LstmVectorCell(ReadOnlySpan<float> input, ReadOnlySpan<float> recurrent, ReadOnlySpan<float> bias,
        Span<float> cell, Span<float> hidden, Span<float> output)
    {
        int size = cell.Length, width = Vector<float>.Count, h = 0;
        var one = Vector<float>.One;
        for (; h <= size - width; h += width)
        {
            var iv = one / (one + Vector.Exp(-LstmGatePre(input, recurrent, bias, h, size)));
            var ov = one / (one + Vector.Exp(-LstmGatePre(input, recurrent, bias, size + h, size)));
            var fv = one / (one + Vector.Exp(-LstmGatePre(input, recurrent, bias, 2 * size + h, size)));
            var gv = LstmVectorTanh(LstmGatePre(input, recurrent, bias, 3 * size + h, size));
            var forgotten = fv * new Vector<float>(cell.Slice(h, width));
            var entered = iv * gv;
            var next = forgotten + entered;
            var result = ov * LstmVectorTanh(next);
            next.CopyTo(cell.Slice(h, width)); result.CopyTo(hidden.Slice(h, width)); result.CopyTo(output.Slice(h, width));
        }
        return h;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector<float> LstmGatePre(ReadOnlySpan<float> input, ReadOnlySpan<float> recurrent,
        ReadOnlySpan<float> bias, int offset, int size)
    {
        int width = Vector<float>.Count;
        var value = new Vector<float>(input.Slice(offset, width)) + new Vector<float>(recurrent.Slice(offset, width));
        value += new Vector<float>(bias.Slice(offset, width));
        return value + new Vector<float>(bias.Slice(4 * size + offset, width));
    }

    // Float rational tanh with separate multiply/add operations.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector<float> LstmVectorTanh(Vector<float> value)
    {
        var lower = new Vector<float>(-9f); var upper = new Vector<float>(9f);
        value = Vector.ConditionalSelect(Vector.LessThan(value, lower), lower, value);
        value = Vector.ConditionalSelect(Vector.GreaterThan(value, upper), upper, value);
        var squared = value * value;
        var numerator = squared * new Vector<float>(-2.76076847742355e-16f) + new Vector<float>(2.00018790482477e-13f);
        numerator = numerator * squared + new Vector<float>(-8.60467152213735e-11f);
        numerator = numerator * squared + new Vector<float>(5.12229709037114e-08f);
        numerator = numerator * squared + new Vector<float>(1.48572235717979e-05f);
        numerator = numerator * squared + new Vector<float>(6.37261928875436e-04f);
        numerator = numerator * squared + new Vector<float>(4.89352455891786e-03f);
        numerator *= value;
        var denominator = squared * new Vector<float>(1.19825839466702e-06f) + new Vector<float>(1.18534705686654e-04f);
        denominator = denominator * squared + new Vector<float>(2.26843463243900e-03f);
        denominator = denominator * squared + new Vector<float>(4.89352518554385e-03f);
        return numerator / denominator;
    }
}
