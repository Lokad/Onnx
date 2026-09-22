"""Derive direct final-row stores while preserving the original reduction loops."""
import re


def generate(source):
    signature='public unsafe static void mm_unsafe_vectorized_intrinsics_3x4packed('
    assert source.count(signature)==1
    start=source.index(signature); brace=source.index('{',start); end=brace+1; depth=1
    while depth:
        depth+=(source[end]=='{')-(source[end]=='}'); end+=1
    original=source[start:end]; method=original
    def replace(old,new,count):
        nonlocal method
        assert method.count(old)==count,(old,method.count(old),count)
        method=method.replace(old,new)
    replace(signature,'public unsafe static void Multiply(',1)
    replace('float* C)','float* C, int outputStride, float* Bias, bool hasBias)',1)
    for old,new,count in [('C + i * K','C + i * outputStride',3),
        ('C + (i + 1) * K','C + (i + 1) * outputStride',2),
        ('C + (i + 2) * K','C + (i + 2) * outputStride',2),
        ('var Cp2 = Cp1 + K;','var Cp2 = Cp1 + outputStride;',1),
        ('var Cp3 = Cp2 + K;','var Cp3 = Cp2 + outputStride;',1)]: replace(old,new,count)
    method,count=re.subn(r'(Vector256<float> c[012][0123] = )Cpv[123]\[[0123]\];',r'\1Vector256<float>.Zero;',method)
    assert count==12
    method,count=re.subn(r'(Vector256<float> c[123] = )rC[123]\[tt\];',r'\1Vector256<float>.Zero;',method)
    assert count==3
    method,count=re.subn(r'(Vector256<int> ci[123] = )Avx2.MaskLoad\(\(int\*\)Cp[123], tmask\);',r'\1Vector256<int>.Zero;',method)
    assert count==3
    additions=['                if (hasBias)','                {','                    var bias = Vector256.Create(Bias[i]);']
    for row in range(3):
        if row: additions.append(f'                    bias = Vector256.Create(Bias[i + {row}]);')
        for col in range(4): additions.append(f'                    c{row}{col} = Avx.Add(c{row}{col}, bias);')
    additions.append('                }')
    replace('                Cpv1[0] = c00;', '\n'.join(additions)+'\n                Cpv1[0] = c00;',1)
    bias3='''                    if (hasBias)
                    {
                        c1 = Avx.Add(c1, Vector256.Create(Bias[i]));
                        c2 = Avx.Add(c2, Vector256.Create(Bias[i + 1]));
                        c3 = Avx.Add(c3, Vector256.Create(Bias[i + 2]));
                    }
'''
    replace('                    rC1[tt] = c1;',bias3+'                    rC1[tt] = c1;',1)
    replace('                    Vector256<int> co1 =',bias3+'                    Vector256<int> co1 =',1)
    old='''                else
                {
                    for (int j = 0; j < N; ++j)'''
    new='''                else
                {
                    for (int k = 0; k < tail; k++) { Cp1[k] = 0f; Cp2[k] = 0f; Cp3[k] = 0f; }
                    for (int j = 0; j < N; ++j)'''
    replace(old,new,1)
    last='''                        for (int k = 0; k < tail; k++) { Cp1[k] += a1 * t[k]; Cp2[k] += a2 * t[k]; Cp3[k] += a3 * t[k]; }
                    }'''
    replace(last,last+'''
                    if (hasBias)
                        for (int k = 0; k < tail; k++) { Cp1[k] += Bias[i]; Cp2[k] += Bias[i + 1]; Cp3[k] += Bias[i + 2]; }''',1)
    # Every original reduction statement remains in the original order.
    def arithmetic(text):
        return [line.strip() for line in text.splitlines() if 'Fma.MultiplyAdd(' in line or 'c1 = c1 + av1 * bv;' in line
            or 'c2 = c2 + av2 * bv;' in line or 'c3 = c3 + av3 * bv;' in line or 'Cp1[k] += a1 * t[k]' in line]
    assert arithmetic(method)==arithmetic(original)
    assert method.count('for (int j = 0; j < N; ++j)')==4
    header='''// Generated isolated zero-seeded three-row consumer. Full reduction first,
// optional bias second, stores directly to the supplied final row stride.
using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

static class DirectOutput
{
    '''
    return header+method+'\n}\n',original
