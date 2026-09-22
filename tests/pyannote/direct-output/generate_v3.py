"""Preserve the existing scalar epilogue's NaN operand preference explicitly."""
from generate import generate as original_generate


def generate(source):
    generated,original=original_generate(source)
    old='''                    if (hasBias)
                        for (int k = 0; k < tail; k++) { Cp1[k] += Bias[i]; Cp2[k] += Bias[i + 1]; Cp3[k] += Bias[i + 2]; }'''
    new='''                    if (hasBias)
                        for (int k = 0; k < tail; k++)
                        {
                            // Match the existing scalar bias/copy epilogue when
                            // both operands are NaN: the bias payload wins.
                            Cp1[k] = Sse.AddScalar(Vector128.CreateScalar(Bias[i]), Vector128.CreateScalar(Cp1[k])).ToScalar();
                            Cp2[k] = Sse.AddScalar(Vector128.CreateScalar(Bias[i + 1]), Vector128.CreateScalar(Cp2[k])).ToScalar();
                            Cp3[k] = Sse.AddScalar(Vector128.CreateScalar(Bias[i + 2]), Vector128.CreateScalar(Cp3[k])).ToScalar();
                        }'''
    assert generated.count(old)==1
    return generated.replace(old,new),original
