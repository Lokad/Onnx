"""Preserve bias NaN payloads independently of scalar/SIMD register allocation."""
from generate_v3 import generate as previous_generate
import re


def generate(source):
    generated,original=previous_generate(source)
    generated,count=re.subn(r'Avx\.Add\((c[012][0123]), bias\)',r'AddBiasVector(\1, bias)',generated)
    assert count==12
    generated,count=re.subn(r'Avx\.Add\((c[123]), (Vector256\.Create\(Bias\[i(?: \+ [12])?\]\))\)',r'AddBiasVector(\1, \2)',generated)
    assert count==6
    for row,offset in [(1,''),(2,' + 1'),(3,' + 2')]:
        old=f'Sse.AddScalar(Vector128.CreateScalar(Bias[i{offset}]), Vector128.CreateScalar(Cp{row}[k])).ToScalar()'
        assert generated.count(old)==1
        generated=generated.replace(old,f'AddBiasScalar(Cp{row}[k], Bias[i{offset}])')
    assert generated.endswith('\n}\n')
    helpers='''
    // The original scalar epilogue selects the bias NaN when both operands
    // are NaN. SIMD register allocation may otherwise reverse that choice.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector256<float> AddBiasVector(Vector256<float> value, Vector256<float> bias)
        => float.IsNaN(bias.GetElement(0)) ? Avx.Add(bias, Vector256<float>.Zero) : Avx.Add(value, bias);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static float AddBiasScalar(float value, float bias)
    {
        var b = Vector128.CreateScalar(bias);
        return float.IsNaN(bias) ? Sse.AddScalar(b, Vector128<float>.Zero).ToScalar()
            : Sse.AddScalar(Vector128.CreateScalar(value), b).ToScalar();
    }
'''
    return generated[:-3]+helpers+'}\n',original
