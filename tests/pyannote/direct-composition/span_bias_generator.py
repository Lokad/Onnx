"""Match the observed actual span epilogue's accumulator NaN precedence."""
from two_column_generator import generate as prior_generate


def generate(source):
    result,original=prior_generate(source)
    old='''    static Vector256<float> AddBiasVector(Vector256<float> value, Vector256<float> bias)
        => float.IsNaN(bias.GetElement(0)) ? Avx.Add(bias, Vector256<float>.Zero) : Avx.Add(value, bias);'''
    new='''    static Vector256<float> AddBiasVector(Vector256<float> value, Vector256<float> bias)
    {
        var accumulatorNaN = Avx.Compare(value, value, FloatComparisonMode.UnorderedNonSignaling);
        var quietAccumulator = Avx.Add(value, Vector256<float>.Zero);
        return Avx.BlendVariable(Avx.Add(value, bias), quietAccumulator, accumulatorNaN);
    }'''
    assert result.count(old)==1;result=result.replace(old,new)
    old='''    static float AddBiasScalar(float value, float bias)
    {
        var b = Vector128.CreateScalar(bias);
        return float.IsNaN(bias) ? Sse.AddScalar(b, Vector128<float>.Zero).ToScalar()
            : Sse.AddScalar(Vector128.CreateScalar(value), b).ToScalar();
    }'''
    new='''    internal static float AddBiasScalar(float value, float bias)
    {
        var v = Vector128.CreateScalar(value);
        return float.IsNaN(value) ? Sse.AddScalar(v, Vector128<float>.Zero).ToScalar()
            : Sse.AddScalar(v, Vector128.CreateScalar(bias)).ToScalar();
    }'''
    assert result.count(old)==1;result=result.replace(old,new)
    # Update comments that described the component pointer epilogue contract.
    result=result.replace('the bias payload wins.','the accumulator payload wins.')
    result=result.replace('The original scalar epilogue selects the bias NaN when both operands',
        'The actual span epilogue selects the accumulator NaN when both operands')
    return result,original
