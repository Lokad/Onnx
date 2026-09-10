using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Freezes exceptional floating-point propagation against ORT 1.29 probe
/// values: NaN and infinite Softmax rows, negative Sqrt, zero-by-zero
/// division, and infinite Erf. The differential corpus cannot carry these
/// because its generator refuses non-finite outputs by policy.
/// </summary>
public class ExceptionalFloatTests
{
    static Tensor<float> RunSoftmax(float[,] values)
    {
        var result = CPU.Softmax(DenseTensor<float>.OfValues(values), -1, null, null, 13);
        Assert.Equal(OpStatus.Success, result.Status);
        return (Tensor<float>)result.Outputs![0];
    }

    [Fact]
    public void SoftmaxNaNRow_PropagatesAcrossRow()
    {
        var y = RunSoftmax(new float[,] { { float.NaN, 1f }, { 1f, 2f } });
        var values = y.ToArray();
        Assert.True(float.IsNaN(values[0]));
        Assert.True(float.IsNaN(values[1]));
        Assert.Equal(0.26894f, values[2], 5);
        Assert.Equal(0.73106f, values[3], 5);
    }

    [Fact]
    public void SoftmaxInfiniteRows_YieldNaN()
    {
        var y = RunSoftmax(new float[,] { { float.NegativeInfinity, float.NegativeInfinity }, { float.PositiveInfinity, float.PositiveInfinity } });
        foreach (var v in y.ToArray()) Assert.True(float.IsNaN(v));
    }

    static Tensor<double> RunSoftmaxDouble(double[,] values)
    {
        var result = CPU.Softmax(DenseTensor<double>.OfValues(values), -1, null, null, 13);
        Assert.Equal(OpStatus.Success, result.Status);
        return (Tensor<double>)result.Outputs![0];
    }

    [Fact]
    public void SoftmaxDoubleNaNRow_PropagatesAcrossRow()
    {
        // ORT 1.29 double: NaN row -> [nan, nan]; [1, 2] -> [0.26894..., 0.73105...].
        var y = RunSoftmaxDouble(new double[,] { { double.NaN, 1.0 }, { 1.0, 2.0 } });
        var values = y.ToArray();
        Assert.True(double.IsNaN(values[0]));
        Assert.True(double.IsNaN(values[1]));
        Assert.Equal(0.26894142137, values[2], 12);
        Assert.Equal(0.73105857863, values[3], 12);
    }

    [Fact]
    public void SoftmaxDoubleInfiniteRows_YieldNaN()
    {
        // ORT 1.29 double: all-infinite rows -> [nan, nan] each.
        var y = RunSoftmaxDouble(new double[,] { { double.NegativeInfinity, double.NegativeInfinity }, { double.PositiveInfinity, double.PositiveInfinity } });
        foreach (var v in y.ToArray()) Assert.True(double.IsNaN(v));
    }

    [Fact]
    public void SqrtNegative_YieldsNaN()
    {
        var result = CPU.Sqrt(DenseTensor<float>.OfValues(new float[] { -1f, 0f }), null);
        Assert.Equal(OpStatus.Success, result.Status);
        var values = ((Tensor<float>)result.Outputs![0]).ToArray();
        Assert.True(float.IsNaN(values[0]));
        Assert.Equal(0f, values[1]);
    }

    [Fact]
    public void DivZeroByZero_YieldsNaN()
    {
        var result = CPU.Div(DenseTensor<float>.OfValues(new float[] { 0f }), DenseTensor<float>.OfValues(new float[] { 0f }), null, null);
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.True(float.IsNaN(((Tensor<float>)result.Outputs![0])[0]));
    }


    [Fact]
    public void GeluExceptional_MatchesOrt()
    {
        // ORT 1.29: [nan, inf, nan].
        var result = CPU.Gelu(DenseTensor<float>.OfValues(new float[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity }), null, null, null);
        Assert.Equal(OpStatus.Success, result.Status);
        var values = ((Tensor<float>)result.Outputs![0]).ToArray();
        Assert.True(float.IsNaN(values[0]));
        Assert.Equal(float.PositiveInfinity, values[1]);
        Assert.True(float.IsNaN(values[2]));
    }

    [Fact]
    public void TanhExceptional_MatchesOrt()
    {
        // ORT 1.29: [1, -1, nan].
        var result = CPU.Tanh(DenseTensor<float>.OfValues(new float[] { float.PositiveInfinity, float.NegativeInfinity, float.NaN }), null);
        Assert.Equal(OpStatus.Success, result.Status);
        var values = ((Tensor<float>)result.Outputs![0]).ToArray();
        Assert.Equal(1f, values[0]);
        Assert.Equal(-1f, values[1]);
        Assert.True(float.IsNaN(values[2]));
    }

    [Fact]
    public void TanhDoubleExceptional_MatchesOrt()
    {
        // ORT 1.29 double: [1, -1, nan].
        var result = CPU.Tanh(DenseTensor<double>.OfValues(new double[] { double.PositiveInfinity, double.NegativeInfinity, double.NaN }), null);
        Assert.Equal(OpStatus.Success, result.Status);
        var values = ((Tensor<double>)result.Outputs![0]).ToArray();
        Assert.Equal(1.0, values[0]);
        Assert.Equal(-1.0, values[1]);
        Assert.True(double.IsNaN(values[2]));
    }

    [Fact]
    public void CosExceptional_MatchesOrt()
    {
        // ORT 1.29: [nan, nan, nan, 1].
        var result = CPU.Cos(DenseTensor<float>.OfValues(new float[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity, 0f }), null);
        Assert.Equal(OpStatus.Success, result.Status);
        var values = ((Tensor<float>)result.Outputs![0]).ToArray();
        Assert.True(float.IsNaN(values[0]));
        Assert.True(float.IsNaN(values[1]));
        Assert.True(float.IsNaN(values[2]));
        Assert.Equal(1f, values[3]);
    }

    [Fact]
    public void SinExceptional_MatchesOrt()
    {
        // ORT 1.29: [nan, nan, nan, 0].
        var result = CPU.Sin(DenseTensor<float>.OfValues(new float[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity, 0f }), null);
        Assert.Equal(OpStatus.Success, result.Status);
        var values = ((Tensor<float>)result.Outputs![0]).ToArray();
        Assert.True(float.IsNaN(values[0]));
        Assert.True(float.IsNaN(values[1]));
        Assert.True(float.IsNaN(values[2]));
        Assert.Equal(0f, values[3]);
    }

    [Fact]
    public void CosDoubleExceptional_MatchesOrt()
    {
        // ORT 1.29 double: [nan, nan, nan, 1].
        var result = CPU.Cos(DenseTensor<double>.OfValues(new double[] { double.NaN, double.PositiveInfinity, double.NegativeInfinity, 0.0 }), null);
        Assert.Equal(OpStatus.Success, result.Status);
        var values = ((Tensor<double>)result.Outputs![0]).ToArray();
        Assert.True(double.IsNaN(values[0]));
        Assert.True(double.IsNaN(values[1]));
        Assert.True(double.IsNaN(values[2]));
        Assert.Equal(1.0, values[3]);
    }

    [Fact]
    public void SinDoubleExceptional_MatchesOrt()
    {
        // ORT 1.29 double: [nan, nan, nan, 0].
        var result = CPU.Sin(DenseTensor<double>.OfValues(new double[] { double.NaN, double.PositiveInfinity, double.NegativeInfinity, 0.0 }), null);
        Assert.Equal(OpStatus.Success, result.Status);
        var values = ((Tensor<double>)result.Outputs![0]).ToArray();
        Assert.True(double.IsNaN(values[0]));
        Assert.True(double.IsNaN(values[1]));
        Assert.True(double.IsNaN(values[2]));
        Assert.Equal(0.0, values[3]);
    }

    [Fact]
    public void NegDoubleExceptional_MatchesOrt()
    {
        // ORT 1.29 double: [nan, -inf, inf, -0].
        var result = CPU.Neg(DenseTensor<double>.OfValues(new double[] { double.NaN, double.PositiveInfinity, double.NegativeInfinity, 0.0 }), null);
        Assert.Equal(OpStatus.Success, result.Status);
        var values = ((Tensor<double>)result.Outputs![0]).ToArray();
        Assert.True(double.IsNaN(values[0]));
        Assert.Equal(double.NegativeInfinity, values[1]);
        Assert.Equal(double.PositiveInfinity, values[2]);
        Assert.Equal(System.BitConverter.DoubleToInt64Bits(-0.0), System.BitConverter.DoubleToInt64Bits(values[3]));
    }

    [Fact]
    public void ReluDoubleExceptional_MatchesOrt()
    {
        // ORT 1.29 double: [nan, 0, inf].
        var result = CPU.Relu(DenseTensor<double>.OfValues(new double[] { double.NaN, double.NegativeInfinity, double.PositiveInfinity }), null);
        Assert.Equal(OpStatus.Success, result.Status);
        var values = ((Tensor<double>)result.Outputs![0]).ToArray();
        Assert.True(double.IsNaN(values[0]));
        Assert.Equal(0.0, values[1]);
        Assert.Equal(double.PositiveInfinity, values[2]);
    }

    [Fact]
    public void GeluDoubleExceptional_MatchesExactFormula()
    {
        // No ORT CPU reference exists: double Gelu decomposes to double Erf,
        // which ORT 1.29 CPU rejects as NOT_IMPLEMENTED. Values follow the
        // exact formula 0.5*x*(1+erf(x/sqrt(2))), matching the float pattern.
        var result = CPU.Gelu(DenseTensor<double>.OfValues(new double[] { double.NaN, double.PositiveInfinity, double.NegativeInfinity }), null, null, null);
        Assert.Equal(OpStatus.Success, result.Status);
        var values = ((Tensor<double>)result.Outputs![0]).ToArray();
        Assert.True(double.IsNaN(values[0]));
        Assert.Equal(double.PositiveInfinity, values[1]);
        Assert.True(double.IsNaN(values[2]));
    }

    [Fact]
    public void ErfDoubleInfinite_MatchesExactFormula()
    {
        // No ORT CPU reference exists: ORT 1.29 CPU rejects double Erf as
        // NOT_IMPLEMENTED. erf(+/-inf) = +/-1 is mathematically forced.
        var result = CPU.Erf(DenseTensor<double>.OfValues(new double[] { double.PositiveInfinity, double.NegativeInfinity, 0.0 }), null, null);
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.Equal(new double[] { 1.0, -1.0, 0.0 }, ((Tensor<double>)result.Outputs![0]).ToArray());
    }

    [Fact]
    public void SqrtDoubleExceptional_MatchesOrt()
    {
        // ORT 1.29 double: [-1, 0, 4] -> [nan, 0, 2].
        var result = CPU.Sqrt(DenseTensor<double>.OfValues(new double[] { -1, 0, 4 }), null);
        Assert.Equal(OpStatus.Success, result.Status);
        var values = ((Tensor<double>)result.Outputs![0]).ToArray();
        Assert.True(double.IsNaN(values[0]));
        Assert.Equal(0.0, values[1]);
        Assert.Equal(2.0, values[2]);
    }

    [Fact]
    public void DivDoubleZeroByZero_MatchesOrt()
    {
        // ORT 1.29 double: [0/0, 1/0] -> [nan, inf].
        var result = CPU.Div(DenseTensor<double>.OfValues(new double[] { 0, 1 }), DenseTensor<double>.OfValues(new double[] { 0, 0 }), null, null);
        Assert.Equal(OpStatus.Success, result.Status);
        var values = ((Tensor<double>)result.Outputs![0]).ToArray();
        Assert.True(double.IsNaN(values[0]));
        Assert.Equal(double.PositiveInfinity, values[1]);
    }

    [Fact]
    public void SqrtNegativeInfinity_YieldsNaN()
    {
        var result = CPU.Sqrt(DenseTensor<float>.OfValues(new float[] { float.NegativeInfinity }), null);
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.True(float.IsNaN(((Tensor<float>)result.Outputs![0])[0]));
    }

    [Fact]
    public void ReluExceptional_MatchesOrt()
    {
        // ORT 1.29: [nan, 0, inf].
        var result = CPU.Relu(DenseTensor<float>.OfValues(new float[] { float.NaN, float.NegativeInfinity, float.PositiveInfinity }), null);
        Assert.Equal(OpStatus.Success, result.Status);
        var values = ((Tensor<float>)result.Outputs![0]).ToArray();
        Assert.True(float.IsNaN(values[0]));
        Assert.Equal(0f, values[1]);
        Assert.Equal(float.PositiveInfinity, values[2]);
    }

    [Fact]
    public void PowExceptional_MatchesOrt()
    {
        // ORT 1.29: 0^0=1, (-1)^0.5=nan, inf^2=inf, 10^100=inf.
        Assert.Equal(1f, Pow1(0f, 0f));
        Assert.True(float.IsNaN(Pow1(-1f, 0.5f)));
        Assert.Equal(float.PositiveInfinity, Pow1(float.PositiveInfinity, 2f));
        Assert.Equal(float.PositiveInfinity, Pow1(10f, 100f));
    }

    [Fact]
    public void PowNegativeBaseIntegerExponents_MatchesOrt()
    {
        // ORT 1.29: [-512, nan, 1024, -0.125] for [-8,-4,2,-2] ^ [3,0.5,10,-3].
        var result = CPU.Pow(DenseTensor<float>.OfValues(new float[] { -8f, -4f, 2f, -2f }), DenseTensor<float>.OfValues(new float[] { 3f, 0.5f, 10f, -3f }), null);
        Assert.Equal(OpStatus.Success, result.Status);
        var values = ((Tensor<float>)result.Outputs![0]).ToArray();
        Assert.Equal(-512f, values[0]);
        Assert.True(float.IsNaN(values[1]));
        Assert.Equal(1024f, values[2]);
        Assert.Equal(-0.125f, values[3]);
    }

    [Fact]
    public void PowNegativeBaseDouble_MatchesOrt()
    {
        // ORT 1.29 double: [-512, nan, 1024, -0.125] for [-8,-4,2,-2] ^ [3,0.5,10,-3].
        var result = CPU.Pow(DenseTensor<double>.OfValues(new double[] { -8, -4, 2, -2 }), DenseTensor<double>.OfValues(new double[] { 3, 0.5, 10, -3 }), null);
        Assert.Equal(OpStatus.Success, result.Status);
        var values = ((Tensor<double>)result.Outputs![0]).ToArray();
        Assert.Equal(-512.0, values[0]);
        Assert.True(double.IsNaN(values[1]));
        Assert.Equal(1024.0, values[2]);
        Assert.Equal(-0.125, values[3]);
    }

    static float Pow1(float a, float b)
    {
        var result = CPU.Pow(DenseTensor<float>.OfValues(new float[] { a }), DenseTensor<float>.OfValues(new float[] { b }), null);
        Assert.Equal(OpStatus.Success, result.Status);
        return ((Tensor<float>)result.Outputs![0])[0];
    }
    [Fact]
    public void ErfInfinite_YieldsSignedOne()
    {
        var result = CPU.Erf(DenseTensor<float>.OfValues(new float[] { float.PositiveInfinity, float.NegativeInfinity, 0f }), null, null);
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.Equal(new float[] { 1f, -1f, 0f }, ((Tensor<float>)result.Outputs![0]).ToArray());
    }
}
