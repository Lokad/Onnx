"""Replace only final masked stores; keep every original shape and reduction."""
from pathlib import Path
import sys

ORIGINAL=Path(__file__).resolve().parents[1] / 'direct-output'
sys.path.insert(0,str(ORIGINAL))
from generate_v4 import generate as previous_generate


def generate(source):
    generated,original=previous_generate(source)
    for row in range(1,4):
        cast=f'                    Vector256<int> co{row} = Unsafe.As<Vector256<float>, Vector256<int>>(ref c{row});\n'
        old=f'Avx2.MaskStore((int*)Cp{row}, tmask, co{row});'
        assert generated.count(cast)==1 and generated.count(old)==1
        generated=generated.replace(cast,'').replace(old,f'StoreNarrow(Cp{row}, c{row}, tail);')
    assert generated.endswith('}\n') and 'Avx2.MaskStore(' not in generated
    helper='''
    // Store exactly the valid floats, without reading or writing padding.
    // The 64-bit stores reinterpret bits; no double arithmetic is performed.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static unsafe void StoreNarrow(float* destination, Vector256<float> value, int count)
    {
        switch (count)
        {
            case 1:
                Sse.StoreScalar(destination, value.GetLower());
                return;
            case 2:
                Sse2.StoreScalar((double*)destination, value.GetLower().AsDouble());
                return;
            case 3:
                Sse2.StoreScalar((double*)destination, value.GetLower().AsDouble());
                destination[2] = value.GetElement(2);
                return;
            case 4:
                Sse.Store(destination, value.GetLower());
                return;
            case 5:
                Sse.Store(destination, value.GetLower());
                destination[4] = value.GetElement(4);
                return;
            case 6:
                Sse.Store(destination, value.GetLower());
                Sse2.StoreScalar((double*)(destination + 4), value.GetUpper().AsDouble());
                return;
            case 7:
                Sse.Store(destination, value.GetLower());
                Sse2.StoreScalar((double*)(destination + 4), value.GetUpper().AsDouble());
                destination[6] = value.GetElement(6);
                return;
            default: throw new ArgumentOutOfRangeException(nameof(count));
        }
    }
'''
    return generated[:-2]+helper+'}\n',original
