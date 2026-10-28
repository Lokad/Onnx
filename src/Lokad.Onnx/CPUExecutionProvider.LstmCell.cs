using System;

namespace Lokad.Onnx;

public partial class CPUExecutionProvider
{
    // The ordinary omitted-activation cell, with the same scalar functions and
    // separate ascending bias additions as the fully general epilogue.
    static void LstmDefaultCell(ReadOnlySpan<float> input, ReadOnlySpan<float> recurrent,
        ReadOnlySpan<float> bias, Span<float> cell, Span<float> hidden, Span<float> output)
    {
        int size = cell.Length;
        for (int h = 0; h < size; h++)
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
}
