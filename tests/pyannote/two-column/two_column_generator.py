"""Keep the two-column reduction separate from generic output bookkeeping."""
from pathlib import Path
import sys

ORIGINAL=Path(__file__).resolve().parents[1] / 'direct-output'
sys.path.insert(0,str(ORIGINAL))
from generate_v4 import generate as prior_generate


def generate(source):
    result,original=prior_generate(source)
    start=result.index('public unsafe static void Multiply(')
    brace=result.index('{',start)
    result=result[:brace+1]+'''
        if (K == 2)
        {
            if (Avx2.IsSupported)
            {
                MultiplyTwoColumns(M, N, A, P, C, outputStride, Bias, hasBias);
                return;
            }
        }
'''+result[brace+1:]
    helper='''
    // Keep generic panel/output state out of this long two-column reduction.
    [MethodImpl(MethodImplOptions.NoInlining)]
    static unsafe void MultiplyTwoColumns(int M, int N, float* A, float* P,
        float* C, int outputStride, float* Bias, bool hasBias)
    {
        var mask = Vector256.Create(-1, -1, 0, 0, 0, 0, 0, 0);
        for (int i = 0; i < M; i += 3)
        {
            float* a1 = A + i * N;
            float* a2 = a1 + N;
            float* a3 = a2 + N;
            float* bp = P;
            var c1 = Vector256<float>.Zero;
            var c2 = Vector256<float>.Zero;
            var c3 = Vector256<float>.Zero;
            for (int j = 0; j < N; ++j)
            {
                var bv = Avx2.MaskLoad((int*)bp, mask).AsSingle();
                var av1 = Vector256.Create(*a1);
                var av2 = Vector256.Create(*a2);
                var av3 = Vector256.Create(*a3);
                c1 = c1 + av1 * bv;
                c2 = c2 + av2 * bv;
                c3 = c3 + av3 * bv;
                ++a1; ++a2; ++a3; bp += 2;
            }
            if (hasBias)
            {
                c1 = AddBiasVector(c1, Vector256.Create(Bias[i]));
                c2 = AddBiasVector(c2, Vector256.Create(Bias[i + 1]));
                c3 = AddBiasVector(c3, Vector256.Create(Bias[i + 2]));
            }
            float* cp = C + i * outputStride;
            Avx2.MaskStore((int*)cp, mask, c1.AsInt32());
            Avx2.MaskStore((int*)(cp + outputStride), mask, c2.AsInt32());
            Avx2.MaskStore((int*)(cp + 2 * outputStride), mask, c3.AsInt32());
        }
    }
'''
    assert result.endswith('}\n')
    return result[:-2]+helper+'}\n',original
