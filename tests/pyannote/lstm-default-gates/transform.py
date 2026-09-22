"""Specialize only the validated default state update; retain the generic loop."""
import difflib

RECURRENT='src/Lokad.Onnx/CPUExecutionProvider.Recurrent.cs'
HELPER='src/Lokad.Onnx/Zzz.LstmDefaultGates.cs'

def replace_once(source,before,after):
    assert source.count(before)==1,before
    return source.replace(before,after)

def transform(source):
    original=source
    source=replace_once(source,'            var hAct = gates[3 * d + 2];', '''            var hAct = gates[3 * d + 2];
            bool defaultGates = bd is not null && pd is null && !clip.HasValue && !inputForget
                && acts[3 * d].ToLowerInvariant() == "sigmoid"
                && acts[3 * d + 1].ToLowerInvariant() == "tanh"
                && acts[3 * d + 2].ToLowerInvariant() == "tanh";''')
    source=replace_once(source,'                    for (int h = 0; h < H; h++)', '''                    if (defaultGates)
                    {
                        LstmUpdateDefaultGates(xw, hr, bd!.Buffer.Span.Slice(bDir, 8 * H),
                            cv, hv, yArr.AsSpan(yOff, H));
                        continue;
                    }
                    for (int h = 0; h < H; h++)''')
    loop=original[original.index('                    for (int h = 0; h < H; h++)'):original.index('                if (yhArr is not null)')]
    assert loop in source
    helper='''using System;

namespace Lokad.Onnx;

public partial class CPUExecutionProvider
{
    // Caller has validated shapes and default sigmoid/tanh/tanh, bias present,
    // no peepholes or clipping, and uncoupled input/forget. Preserve every
    // selected float operation; only remove generic dispatch and buffer access.
    static void LstmUpdateDefaultGates(ReadOnlySpan<float> xw, ReadOnlySpan<float> hr,
        ReadOnlySpan<float> bias, Span<float> cv, Span<float> hv, Span<float> output)
    {
        int H = hv.Length;
        for (int h = 0; h < H; h++)
        {
            float iPre = xw[h] + hr[h] + bias[h] + bias[4 * H + h];
            float oPre = xw[H + h] + hr[H + h] + bias[H + h] + bias[5 * H + h];
            float fPre = xw[2 * H + h] + hr[2 * H + h] + bias[2 * H + h] + bias[6 * H + h];
            float gPre = xw[3 * H + h] + hr[3 * H + h] + bias[3 * H + h] + bias[7 * H + h];
            float fv = 1f / (1f + MathF.Exp(-fPre));
            float gv = MathF.Tanh(gPre);
            float iv = 1f / (1f + MathF.Exp(-iPre));
            float cNew = fv * cv[h] + iv * gv;
            float hNew = (1f / (1f + MathF.Exp(-oPre))) * MathF.Tanh(cNew);
            cv[h] = cNew;
            hv[h] = hNew;
            output[h] = hNew;
        }
    }
}
'''
    changed={RECURRENT:source,HELPER:helper}
    patch=''.join(difflib.unified_diff(original.splitlines(True),source.splitlines(True),fromfile=RECURRENT,tofile=RECURRENT))
    patch+=''.join(difflib.unified_diff([],helper.splitlines(True),fromfile='/dev/null',tofile=HELPER))
    return changed,patch
