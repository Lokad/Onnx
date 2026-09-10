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
    public void SoftmaxSingleElementRow_YieldsOne()
    {
        // ORT 1.29: softmax over one element is exactly 1 regardless of
        // the input value (exp(x - x) / exp(x - x)).
        var y = RunSoftmax(new float[,] { { 5f }, { -3f } });
        Assert.Equal(new float[] { 1f, 1f }, y.ToArray());
    }

    [Fact]
    public void SoftmaxDoubleDegenerateRows_YieldExpected()
    {
        // ORT 1.29 double: single-element rows yield 1, empty rows yield
        // empty, mirroring the float degenerate set.
        var one = CPU.Softmax(DenseTensor<double>.OfValues(new double[,] { { 5.0 }, { -3.0 } }), -1, null, null, 13);
        Assert.Equal(OpStatus.Success, one.Status);
        Assert.Equal(new double[] { 1.0, 1.0 }, ((Tensor<double>)one.Outputs![0]).ToArray());
        var empty = CPU.Softmax(DenseTensor<double>.OfShape(2, 0), -1, null, null, 13);
        Assert.Equal(OpStatus.Success, empty.Status);
        var y = (Tensor<double>)empty.Outputs![0];
        Assert.Equal(new int[] { 2, 0 }, y.Dimensions.ToArray());
        Assert.Empty(y.ToArray());
    }

    [Fact]
    public void SoftmaxEmptyRows_YieldEmpty()
    {
        // ORT 1.29: softmax over zero-element rows yields [2,0], empty.
        var y = RunSoftmax(new float[2, 0]);
        Assert.Equal(new int[] { 2, 0 }, y.Dimensions.ToArray());
        Assert.Empty(y.ToArray());
    }

    [Fact]
    public void SoftmaxLargeFinite_StaysStable()
    {
        // ORT 1.29: max-subtraction makes [1000, 1001] identical to [0, 1];
        // without it exp overflows to inf/inf = NaN. The only pin covering
        // the stabilization path.
        var f = RunSoftmax(new float[,] { { 1000f, 1001f } }).ToArray();
        Assert.Equal(0.26894143f, f[0], 5);
        Assert.Equal(0.73105860f, f[1], 5);
        var d = RunSoftmaxDouble(new double[,] { { 1000.0, 1001.0 } }).ToArray();
        Assert.Equal(0.26894142137, d[0], 12);
        Assert.Equal(0.73105857863, d[1], 12);
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
    public void DivByZero_SignedInfinities()
    {
        // ORT 1.29: [1/0, -1/0, 1/-0, 0/0] -> [inf, -inf, -inf, nan].
        var result = CPU.Div(DenseTensor<float>.OfValues(new float[] { 1f, -1f, 1f, 0f }), DenseTensor<float>.OfValues(new float[] { 0f, 0f, -0f, 0f }), null, null);
        Assert.Equal(OpStatus.Success, result.Status);
        var values = ((Tensor<float>)result.Outputs![0]).ToArray();
        Assert.Equal(float.PositiveInfinity, values[0]);
        Assert.Equal(float.NegativeInfinity, values[1]);
        Assert.Equal(float.NegativeInfinity, values[2]);
        Assert.True(float.IsNaN(values[3]));
    }

    [Fact]
    public void DivByZero_DoubleSignedInfinities()
    {
        // ORT 1.29 double: [1/0, -1/0, 1/-0, 0/0] -> [inf, -inf, -inf, nan].
        var result = CPU.Div(DenseTensor<double>.OfValues(new double[] { 1.0, -1.0, 1.0, 0.0 }), DenseTensor<double>.OfValues(new double[] { 0.0, 0.0, -0.0, 0.0 }), null, null);
        Assert.Equal(OpStatus.Success, result.Status);
        var values = ((Tensor<double>)result.Outputs![0]).ToArray();
        Assert.Equal(double.PositiveInfinity, values[0]);
        Assert.Equal(double.NegativeInfinity, values[1]);
        Assert.Equal(double.NegativeInfinity, values[2]);
        Assert.True(double.IsNaN(values[3]));
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
    public void GeluLargeFinite_Saturates()
    {
        // ORT 1.29 float: [10,-10] -> [10,-0] exactly (erf saturates).
        // (Double Gelu has no ORT CPU kernel, so no twin exists.)
        var r = CPU.Gelu(DenseTensor<float>.OfValues(new float[] { 10f, -10f }), null, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = ((Tensor<float>)r.Outputs![0]).ToArray();
        Assert.Equal(10f, y[0]);
        Assert.Equal(unchecked((int)0x80000000), System.BitConverter.SingleToInt32Bits(y[1]));
    }

    [Fact]
    public void TanhLargeFinite_Saturates()
    {
        // ORT 1.29 float and double: [1000,-1000] -> [1,-1] exactly.
        var f = CPU.Tanh(DenseTensor<float>.OfValues(new float[] { 1000f, -1000f }), null);
        Assert.Equal(OpStatus.Success, f.Status);
        Assert.Equal(new float[] { 1f, -1f }, ((Tensor<float>)f.Outputs![0]).ToArray());
        var d = CPU.Tanh(DenseTensor<double>.OfValues(new double[] { 1000.0, -1000.0 }), null);
        Assert.Equal(OpStatus.Success, d.Status);
        Assert.Equal(new double[] { 1.0, -1.0 }, ((Tensor<double>)d.Outputs![0]).ToArray());
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
    public void CosSin_BasicsMatchOrt()
    {
        // ORT 1.29 over [0, pi/3, pi/2, pi]: cos rows and sin rows below.
        // (ORT signed zeros assert-safe via IEEE equality.)
        var cf = CPU.Cos(DenseTensor<float>.OfValues(new float[] { 0f, (float)(System.Math.PI / 3.0), (float)(System.Math.PI / 2.0), (float)System.Math.PI }), null);
        Assert.Equal(OpStatus.Success, cf.Status);
        Assert.Equal(1f, ((Tensor<float>)cf.Outputs![0])[0], 5);
        Assert.Equal(0.5f, ((Tensor<float>)cf.Outputs![0])[1], 5);
        Assert.Equal(0f, ((Tensor<float>)cf.Outputs![0])[2], 5);
        Assert.Equal(-1f, ((Tensor<float>)cf.Outputs![0])[3], 5);
        var sf = CPU.Sin(DenseTensor<float>.OfValues(new float[] { 0f, (float)(System.Math.PI / 3.0), (float)(System.Math.PI / 2.0), (float)System.Math.PI }), null);
        Assert.Equal(OpStatus.Success, sf.Status);
        Assert.Equal(0f, ((Tensor<float>)sf.Outputs![0])[0], 5);
        Assert.Equal(0.8660254f, ((Tensor<float>)sf.Outputs![0])[1], 6);
        Assert.Equal(1f, ((Tensor<float>)sf.Outputs![0])[2], 5);
        Assert.Equal(0f, ((Tensor<float>)sf.Outputs![0])[3], 5);
        var cd = CPU.Cos(DenseTensor<double>.OfValues(new double[] { 0.0, System.Math.PI / 3.0, System.Math.PI / 2.0, System.Math.PI }), null);
        Assert.Equal(OpStatus.Success, cd.Status);
        Assert.Equal(1.0, ((Tensor<double>)cd.Outputs![0])[0], 6);
        Assert.Equal(0.5, ((Tensor<double>)cd.Outputs![0])[1], 6);
        Assert.Equal(0.0, ((Tensor<double>)cd.Outputs![0])[2], 6);
        Assert.Equal(-1.0, ((Tensor<double>)cd.Outputs![0])[3], 6);
        var sd = CPU.Sin(DenseTensor<double>.OfValues(new double[] { 0.0, System.Math.PI / 3.0, System.Math.PI / 2.0, System.Math.PI }), null);
        Assert.Equal(OpStatus.Success, sd.Status);
        Assert.Equal(0.0, ((Tensor<double>)sd.Outputs![0])[0], 6);
        Assert.Equal(0.8660254038, ((Tensor<double>)sd.Outputs![0])[1], 6);
        Assert.Equal(1.0, ((Tensor<double>)sd.Outputs![0])[2], 6);
        Assert.Equal(0.0, ((Tensor<double>)sd.Outputs![0])[3], 6);
    }

    [Fact]
    public void NegSqrtAbsNaN_Passthrough()
    {
        // ORT 1.29 float: [nan] each (Neg-double already pins its own).
        var n = CPU.Neg(DenseTensor<float>.OfValues(new float[] { float.NaN }), null);
        Assert.Equal(OpStatus.Success, n.Status);
        Assert.True(float.IsNaN(((Tensor<float>)n.Outputs![0])[0]));
        var s = CPU.Sqrt(DenseTensor<float>.OfValues(new float[] { float.NaN }), null);
        Assert.Equal(OpStatus.Success, s.Status);
        Assert.True(float.IsNaN(((Tensor<float>)s.Outputs![0])[0]));
        var a = CPU.Abs(DenseTensor<float>.OfValues(new float[] { float.NaN }), null);
        Assert.Equal(OpStatus.Success, a.Status);
        Assert.True(float.IsNaN(((Tensor<float>)a.Outputs![0])[0]));
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
    public void SqrtAbsDoubleNaN_Passthrough()
    {
        // ORT 1.29 double: NaN stays NaN through Sqrt and Abs.
        var s = CPU.Sqrt(DenseTensor<double>.OfValues(new double[] { double.NaN }), null);
        Assert.Equal(OpStatus.Success, s.Status);
        Assert.True(double.IsNaN(((Tensor<double>)s.Outputs![0])[0]));
        var a = CPU.Abs(DenseTensor<double>.OfValues(new double[] { double.NaN }), null);
        Assert.Equal(OpStatus.Success, a.Status);
        Assert.True(double.IsNaN(((Tensor<double>)a.Outputs![0])[0]));
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
    public void AddSubMulDoubleNaN_Passthrough()
    {
        // ORT 1.29 double: NaN on either side of Add/Sub/Mul yields NaN,
        // mirroring the float set and the Div double twin.
        var a = CPU.Add(DenseTensor<double>.OfValues(new double[] { double.NaN }), DenseTensor<double>.OfValues(new double[] { 1.0 }), null, null);
        Assert.Equal(OpStatus.Success, a.Status);
        Assert.True(double.IsNaN(((Tensor<double>)a.Outputs![0])[0]));
        var s = CPU.Sub(DenseTensor<double>.OfValues(new double[] { 1.0 }), DenseTensor<double>.OfValues(new double[] { double.NaN }), null);
        Assert.Equal(OpStatus.Success, s.Status);
        Assert.True(double.IsNaN(((Tensor<double>)s.Outputs![0])[0]));
        var m = CPU.Mul(DenseTensor<double>.OfValues(new double[] { double.NaN }), DenseTensor<double>.OfValues(new double[] { 2.0 }), null, null);
        Assert.Equal(OpStatus.Success, m.Status);
        Assert.True(double.IsNaN(((Tensor<double>)m.Outputs![0])[0]));
    }

    [Fact]
    public void DivDoubleNaN_Passthrough()
    {
        // ORT 1.29 double: NaN on either side yields NaN, mirroring the
        // float Div/Pow set above.
        var a = CPU.Div(DenseTensor<double>.OfValues(new double[] { double.NaN }), DenseTensor<double>.OfValues(new double[] { 1.0 }), null, null);
        Assert.Equal(OpStatus.Success, a.Status);
        Assert.True(double.IsNaN(((Tensor<double>)a.Outputs![0])[0]));
        var b = CPU.Div(DenseTensor<double>.OfValues(new double[] { 1.0 }), DenseTensor<double>.OfValues(new double[] { double.NaN }), null, null);
        Assert.Equal(OpStatus.Success, b.Status);
        Assert.True(double.IsNaN(((Tensor<double>)b.Outputs![0])[0]));
    }

    [Fact]
    public void Div_Sqrt_DoubleBasics_MatchOrt()
    {
        // ORT 1.29 double: [7/2, -9/3, 1/4] and sqrt([2, 0.25]).
        var div = CPU.Div(DenseTensor<double>.OfValues(new double[] { 7.0, -9.0, 1.0 }), DenseTensor<double>.OfValues(new double[] { 2.0, 3.0, 4.0 }), null, null);
        Assert.Equal(OpStatus.Success, div.Status);
        Assert.Equal(new double[] { 3.5, -3.0, 0.25 }, ((Tensor<double>)div.Outputs![0]).ToArray());
        var sqrt = CPU.Sqrt(DenseTensor<double>.OfValues(new double[] { 2.0, 0.25 }), null);
        Assert.Equal(OpStatus.Success, sqrt.Status);
        Assert.Equal(1.4142135624, ((Tensor<double>)sqrt.Outputs![0])[0], 6);
        Assert.Equal(0.5, ((Tensor<double>)sqrt.Outputs![0])[1], 6);
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
    public void PowZeroNegativeExponent_SignedInfinity()
    {
        // ORT 1.29: 0^-1=inf, 0^-2=inf, (-0)^-1=-inf (signed zero flows
        // through the reciprocal).
        Assert.Equal(float.PositiveInfinity, Pow1(0f, -1f));
        Assert.Equal(float.PositiveInfinity, Pow1(0f, -2f));
        Assert.Equal(float.NegativeInfinity, Pow1(-0f, -1f));
    }

    [Fact]
    public void PowInfiniteBases_MatchOrt()
    {
        // ORT 1.29: inf^2=inf, (-inf)^2=inf, (-inf)^3=-inf,
        // (-inf)^0.5=inf (non-odd positive exponent), inf^-1=0.
        Assert.Equal(float.PositiveInfinity, Pow1(float.PositiveInfinity, 2f));
        Assert.Equal(float.PositiveInfinity, Pow1(float.NegativeInfinity, 2f));
        Assert.Equal(float.NegativeInfinity, Pow1(float.NegativeInfinity, 3f));
        Assert.Equal(float.PositiveInfinity, Pow1(float.NegativeInfinity, 0.5f));
        Assert.Equal(0f, Pow1(float.PositiveInfinity, -1f));
    }

    [Fact]
    public void PowDoubleNaN_Passthrough()
    {
        // ORT 1.29 double: NaN base or exponent (away from the pinned
        // x^0/1^y identities) yields NaN, mirroring the float set.
        Assert.True(double.IsNaN(Pow1Double(double.NaN, 1.0)));
        Assert.True(double.IsNaN(Pow1Double(2.0, double.NaN)));
    }

    [Fact]
    public void PowZeroNegativeExponentDouble_MatchOrt()
    {
        // ORT 1.29 double: 0^-1=inf, (-0)^-1=-inf, mirroring the float
        // set above through Math.Pow.
        Assert.Equal(double.PositiveInfinity, Pow1Double(0.0, -1.0));
        Assert.Equal(double.NegativeInfinity, Pow1Double(-0.0, -1.0));
    }

    [Fact]
    public void PowInfiniteBasesDouble_MatchOrt()
    {
        // ORT 1.29 double: inf^2=inf, (-inf)^3=-inf, (-inf)^0.5=inf,
        // mirroring the float set above through Math.Pow.
        Assert.Equal(double.PositiveInfinity, Pow1Double(double.PositiveInfinity, 2.0));
        Assert.Equal(double.NegativeInfinity, Pow1Double(double.NegativeInfinity, 3.0));
        Assert.Equal(double.PositiveInfinity, Pow1Double(double.NegativeInfinity, 0.5));
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

    [Fact]
    public void IntPow_MatchesOrt()
    {
        // ORT 1.29 computes integer Pow float-mediated: fractions truncate,
        // overflow and 0^-1 yield min via conversion (not integer wrap).
        var i32 = CPU.Pow(DenseTensor<int>.OfValues(new int[] { 2, 10, 2, 0, 3 }), DenseTensor<int>.OfValues(new int[] { 10, 9, -1, 0, 3 }), null);
        Assert.Equal(OpStatus.Success, i32.Status);
        Assert.Equal(new int[] { 1024, 1000000000, 0, 1, 27 }, ((Tensor<int>)i32.Outputs![0]).ToArray());
        var ovf32 = CPU.Pow(DenseTensor<int>.OfValues(new int[] { 2, -2, 0, -3 }), DenseTensor<int>.OfValues(new int[] { 31, 31, -1, 3 }), null);
        Assert.Equal(OpStatus.Success, ovf32.Status);
        Assert.Equal(new int[] { -2147483648, -2147483648, -2147483648, -27 }, ((Tensor<int>)ovf32.Outputs![0]).ToArray());
        var i64 = CPU.Pow(DenseTensor<long>.OfValues(new long[] { 2L, 10L, 0L }), DenseTensor<long>.OfValues(new long[] { 62L, 18L, -1L }), null);
        Assert.Equal(OpStatus.Success, i64.Status);
        Assert.Equal(new long[] { 4611686018427387904L, 1000000000000000000L, -9223372036854775808L }, ((Tensor<long>)i64.Outputs![0]).ToArray());
        var ovf64 = CPU.Pow(DenseTensor<long>.OfValues(new long[] { 10L }), DenseTensor<long>.OfValues(new long[] { 19L }), null);
        Assert.Equal(OpStatus.Success, ovf64.Status);
        Assert.Equal(new long[] { -9223372036854775808L }, ((Tensor<long>)ovf64.Outputs![0]).ToArray());
    }

    static float Pow1(float a, float b)
    {
        var result = CPU.Pow(DenseTensor<float>.OfValues(new float[] { a }), DenseTensor<float>.OfValues(new float[] { b }), null);
        Assert.Equal(OpStatus.Success, result.Status);
        return ((Tensor<float>)result.Outputs![0])[0];
    }

    static float Div1(float a, float b)
    {
        var result = CPU.Div(DenseTensor<float>.OfValues(new float[] { a }), DenseTensor<float>.OfValues(new float[] { b }), null, null);
        Assert.Equal(OpStatus.Success, result.Status);
        return ((Tensor<float>)result.Outputs![0])[0];
    }

    [Fact]
    public void AddSubMulNaN_Passthrough()
    {
        // ORT 1.29: NaN on either side of Add/Sub/Mul yields NaN,
        // completing the Div/Pow passthrough set below.
        var a = CPU.Add(DenseTensor<float>.OfValues(new float[] { float.NaN }), DenseTensor<float>.OfValues(new float[] { 1f }), null, null);
        Assert.Equal(OpStatus.Success, a.Status);
        Assert.True(float.IsNaN(((Tensor<float>)a.Outputs![0])[0]));
        var s = CPU.Sub(DenseTensor<float>.OfValues(new float[] { 1f }), DenseTensor<float>.OfValues(new float[] { float.NaN }), null);
        Assert.Equal(OpStatus.Success, s.Status);
        Assert.True(float.IsNaN(((Tensor<float>)s.Outputs![0])[0]));
        var m = CPU.Mul(DenseTensor<float>.OfValues(new float[] { float.NaN }), DenseTensor<float>.OfValues(new float[] { 2f }), null, null);
        Assert.Equal(OpStatus.Success, m.Status);
        Assert.True(float.IsNaN(((Tensor<float>)m.Outputs![0])[0]));
    }

    [Fact]
    public void IndeterminateForms_YieldNaN()
    {
        // ORT 1.29: 0*inf, inf/inf, and inf-inf are all NaN (IEEE).
        var m = CPU.Mul(DenseTensor<float>.OfValues(new float[] { 0f }), DenseTensor<float>.OfValues(new float[] { float.PositiveInfinity }), null, null);
        Assert.Equal(OpStatus.Success, m.Status);
        Assert.True(float.IsNaN(((Tensor<float>)m.Outputs![0])[0]));
        Assert.True(float.IsNaN(Div1(float.PositiveInfinity, float.PositiveInfinity)));
        var s = CPU.Sub(DenseTensor<float>.OfValues(new float[] { float.PositiveInfinity }), DenseTensor<float>.OfValues(new float[] { float.PositiveInfinity }), null);
        Assert.Equal(OpStatus.Success, s.Status);
        Assert.True(float.IsNaN(((Tensor<float>)s.Outputs![0])[0]));
    }

    [Fact]
    public void DivPowNaN_Passthrough()
    {
        // ORT 1.29: NaN on either side of Div, and NaN base or exponent
        // of Pow (away from the pinned x^0/1^y identities), yields NaN.
        Assert.True(float.IsNaN(Div1(float.NaN, 1f)));
        Assert.True(float.IsNaN(Div1(1f, float.NaN)));
        Assert.True(float.IsNaN(Pow1(float.NaN, 1f)));
        Assert.True(float.IsNaN(Pow1(2f, float.NaN)));
    }

    static double Pow1Double(double a, double b)
    {
        var result = CPU.Pow(DenseTensor<double>.OfValues(new double[] { a }), DenseTensor<double>.OfValues(new double[] { b }), null);
        Assert.Equal(OpStatus.Success, result.Status);
        return ((Tensor<double>)result.Outputs![0])[0];
    }

    [Fact]
    public void PowIdentities_MatchOrt()
    {
        // ORT 1.29 (probed float and double): IEEE identities hold for
        // infinities and NaN - pow(1, y) = 1, pow(x, 0) = 1, pow(-1, inf) = 1.
        Assert.Equal(1f, Pow1(1f, float.PositiveInfinity));
        Assert.Equal(1f, Pow1(1f, float.NegativeInfinity));
        Assert.Equal(1f, Pow1(float.NaN, 0f));
        Assert.Equal(1f, Pow1(float.PositiveInfinity, 0f));
        Assert.Equal(1f, Pow1(float.NegativeInfinity, 0f));
        Assert.Equal(1f, Pow1(-1f, float.PositiveInfinity));
        Assert.Equal(1.0, Pow1Double(1.0, double.PositiveInfinity));
        Assert.Equal(1.0, Pow1Double(double.NaN, 0.0));
        Assert.Equal(1.0, Pow1Double(double.NegativeInfinity, 0.0));
        Assert.Equal(1.0, Pow1Double(-1.0, double.PositiveInfinity));
    }
    [Fact]
    public void ErfLargeFiniteDouble_MatchesExactFormula()
    {
        // No ORT CPU reference exists (NOT_IMPLEMENTED); math.erf(+/-6)
        // rounds to exactly +/-1.0 in double, like the float saturation.
        var r = CPU.Erf(DenseTensor<double>.OfValues(new double[] { 6.0, -6.0 }), null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new double[] { 1.0, -1.0 }, ((Tensor<double>)r.Outputs![0]).ToArray());
    }

    [Fact]
    public void ErfLargeFinite_Saturates()
    {
        // ORT 1.29 float: erf(+/-6) is exactly +/-1.
        var r = CPU.Erf(DenseTensor<float>.OfValues(new float[] { 6f, -6f }), null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new float[] { 1f, -1f }, ((Tensor<float>)r.Outputs![0]).ToArray());
    }

    [Fact]
    public void ErfNaN_YieldsNaN()
    {
        // ORT 1.29 float: [nan]. Double Erf has no ORT CPU kernel
        // (NOT_IMPLEMENTED), so the double half is pinned exact instead:
        // math.erf(nan) is nan.
        var rf = CPU.Erf(DenseTensor<float>.OfValues(new float[] { float.NaN }), null, null);
        Assert.Equal(OpStatus.Success, rf.Status);
        Assert.True(float.IsNaN(((Tensor<float>)rf.Outputs![0])[0]));
        var rd = CPU.Erf(DenseTensor<double>.OfValues(new double[] { double.NaN }), null, null);
        Assert.Equal(OpStatus.Success, rd.Status);
        Assert.True(double.IsNaN(((Tensor<double>)rd.Outputs![0])[0]));
    }

    [Fact]
    public void ErfInfinite_YieldsSignedOne()
    {
        var result = CPU.Erf(DenseTensor<float>.OfValues(new float[] { float.PositiveInfinity, float.NegativeInfinity, 0f }), null, null);
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.Equal(new float[] { 1f, -1f, 0f }, ((Tensor<float>)result.Outputs![0]).ToArray());
    }

    [Fact]
    public void EqualExceptional_MatchesOrt()
    {
        // ORT 1.29 float and double: NaN equals nothing, not even itself;
        // same-signed infinities are equal; +0 == -0 is true. Guarded
        // because a future bitwise/SIMD Equal could treat NaN as equal.
        var af = DenseTensor<float>.OfValues(new float[] { float.NaN, float.NaN, 1f, float.PositiveInfinity, float.NegativeInfinity, float.PositiveInfinity, 0f, 1f });
        var bf = DenseTensor<float>.OfValues(new float[] { float.NaN, 1f, float.NaN, float.PositiveInfinity, float.NegativeInfinity, float.NegativeInfinity, -0f, 2f });
        var rf = CPU.Equal(af, bf, null);
        Assert.Equal(OpStatus.Success, rf.Status);
        Assert.Equal(new bool[] { false, false, false, true, true, false, true, false }, ((Tensor<bool>)rf.Outputs![0]).ToArray());
        var ad = DenseTensor<double>.OfValues(new double[] { double.NaN, double.NaN, 1.0, double.PositiveInfinity, double.NegativeInfinity, double.PositiveInfinity, 0.0, 1.0 });
        var bd = DenseTensor<double>.OfValues(new double[] { double.NaN, 1.0, double.NaN, double.PositiveInfinity, double.NegativeInfinity, double.NegativeInfinity, -0.0, 2.0 });
        var rd = CPU.Equal(ad, bd, null);
        Assert.Equal(OpStatus.Success, rd.Status);
        Assert.Equal(new bool[] { false, false, false, true, true, false, true, false }, ((Tensor<bool>)rd.Outputs![0]).ToArray());
    }

    [Fact]
    public void LessExceptional_MatchesOrt()
    {
        // ORT 1.29 float and double: any NaN side is false; infinities keep
        // their order; +0 and -0 compare equal (both directions false).
        // Guarded because Comparer<float> sorts NaN below every value.
        var af = DenseTensor<float>.OfValues(new float[] { float.NaN, 1f, float.NaN, float.PositiveInfinity, float.NegativeInfinity, float.NegativeInfinity, float.PositiveInfinity, float.PositiveInfinity, float.NegativeInfinity, 1f, 1f, 1f, 2f, 1f, 0f, -0f });
        var bf = DenseTensor<float>.OfValues(new float[] { 1f, float.NaN, float.NaN, float.PositiveInfinity, float.NegativeInfinity, float.PositiveInfinity, float.NegativeInfinity, 1f, 1f, float.PositiveInfinity, float.NegativeInfinity, 2f, 1f, 1f, -0f, 0f });
        var rf = CPU.Less(af, bf, null);
        Assert.Equal(OpStatus.Success, rf.Status);
        Assert.Equal(new bool[] { false, false, false, false, false, true, false, false, true, true, false, true, false, false, false, false }, ((Tensor<bool>)rf.Outputs![0]).ToArray());
        var ad = DenseTensor<double>.OfValues(new double[] { double.NaN, 1.0, double.NaN, double.PositiveInfinity, double.NegativeInfinity, double.NegativeInfinity, double.PositiveInfinity, double.PositiveInfinity, double.NegativeInfinity, 1.0, 1.0, 1.0, 2.0, 1.0, 0.0, -0.0 });
        var bd = DenseTensor<double>.OfValues(new double[] { 1.0, double.NaN, double.NaN, double.PositiveInfinity, double.NegativeInfinity, double.PositiveInfinity, double.NegativeInfinity, 1.0, 1.0, double.PositiveInfinity, double.NegativeInfinity, 2.0, 1.0, 1.0, -0.0, 0.0 });
        var rd = CPU.Less(ad, bd, null);
        Assert.Equal(OpStatus.Success, rd.Status);
        Assert.Equal(new bool[] { false, false, false, false, false, true, false, false, true, true, false, true, false, false, false, false }, ((Tensor<bool>)rd.Outputs![0]).ToArray());
    }

    [Fact]
    public void ErfSignedZero_MatchesOrt()
    {
        // ORT 1.29 float bits: erf(+0)=+0, erf(-0)=-0. The 40-element
        // alternating input spans the SIMD body and scalar tail on any
        // Vector<float>.Count (4, 8 or 16); both paths must keep the sign.
        var input = new float[40];
        for (int i = 0; i < input.Length; i++) input[i] = (i & 1) == 0 ? 0f : -0f;
        var rf = CPU.Erf(DenseTensor<float>.OfValues(input), null, null);
        Assert.Equal(OpStatus.Success, rf.Status);
        var yf = ((Tensor<float>)rf.Outputs![0]).ToArray();
        for (int i = 0; i < yf.Length; i++)
            Assert.Equal((i & 1) == 0 ? 0 : int.MinValue, System.BitConverter.SingleToInt32Bits(yf[i]));
        // No ORT double kernel (NOT_IMPLEMENTED); erf is odd, so erf(-0)
        // is -0 by the exact-formula idiom. Double Erf is scalar-only.
        var rd = CPU.Erf(DenseTensor<double>.OfValues(new double[] { 0.0, -0.0 }), null, null);
        Assert.Equal(OpStatus.Success, rd.Status);
        var yd = ((Tensor<double>)rd.Outputs![0]).ToArray();
        Assert.Equal(0L, System.BitConverter.DoubleToInt64Bits(yd[0]));
        Assert.Equal(long.MinValue, System.BitConverter.DoubleToInt64Bits(yd[1]));
    }

    [Fact]
    public void UnarySignedZeroBits_MatchesOrt()
    {
        // ORT 1.29 float and double bits for [+0, -0]: Neg flips, Sqrt/Relu/
        // Tanh/Sin preserve, Abs clears. Guards scalar and SIMD spellings
        // (e.g. a future Max(x, 0) Relu would turn -0 into +0).
        static int[] FloatBits(string op, float[] v)
        {
            var r = op switch
            {
                nameof(CPU.Neg) => CPU.Neg(DenseTensor<float>.OfValues(v), null),
                nameof(CPU.Sqrt) => CPU.Sqrt(DenseTensor<float>.OfValues(v), null),
                nameof(CPU.Abs) => CPU.Abs(DenseTensor<float>.OfValues(v), null),
                nameof(CPU.Relu) => CPU.Relu(DenseTensor<float>.OfValues(v), null),
                nameof(CPU.Tanh) => CPU.Tanh(DenseTensor<float>.OfValues(v), null),
                _ => CPU.Sin(DenseTensor<float>.OfValues(v), null),
            };
            Assert.Equal(OpStatus.Success, r.Status);
            return Array.ConvertAll(((Tensor<float>)r.Outputs![0]).ToArray(), System.BitConverter.SingleToInt32Bits);
        }
        static long[] DoubleBits(string op, double[] v)
        {
            var r = op switch
            {
                nameof(CPU.Neg) => CPU.Neg(DenseTensor<double>.OfValues(v), null),
                nameof(CPU.Sqrt) => CPU.Sqrt(DenseTensor<double>.OfValues(v), null),
                nameof(CPU.Abs) => CPU.Abs(DenseTensor<double>.OfValues(v), null),
                nameof(CPU.Relu) => CPU.Relu(DenseTensor<double>.OfValues(v), null),
                nameof(CPU.Tanh) => CPU.Tanh(DenseTensor<double>.OfValues(v), null),
                _ => CPU.Sin(DenseTensor<double>.OfValues(v), null),
            };
            Assert.Equal(OpStatus.Success, r.Status);
            return Array.ConvertAll(((Tensor<double>)r.Outputs![0]).ToArray(), System.BitConverter.DoubleToInt64Bits);
        }
        var fz = new float[] { 0f, -0f };
        var dz = new double[] { 0.0, -0.0 };
        const int NZero = int.MinValue;
        Assert.Equal(new int[] { NZero, 0 }, FloatBits(nameof(CPU.Neg), fz));
        Assert.Equal(new int[] { 0, NZero }, FloatBits(nameof(CPU.Sqrt), fz));
        Assert.Equal(new int[] { 0, 0 }, FloatBits(nameof(CPU.Abs), fz));
        Assert.Equal(new int[] { 0, NZero }, FloatBits(nameof(CPU.Relu), fz));
        Assert.Equal(new int[] { 0, NZero }, FloatBits(nameof(CPU.Tanh), fz));
        Assert.Equal(new int[] { 0, NZero }, FloatBits(nameof(CPU.Sin), fz));
        Assert.Equal(new long[] { long.MinValue, 0L }, DoubleBits(nameof(CPU.Neg), dz));
        Assert.Equal(new long[] { 0L, long.MinValue }, DoubleBits(nameof(CPU.Sqrt), dz));
        Assert.Equal(new long[] { 0L, 0L }, DoubleBits(nameof(CPU.Abs), dz));
        Assert.Equal(new long[] { 0L, long.MinValue }, DoubleBits(nameof(CPU.Relu), dz));
        Assert.Equal(new long[] { 0L, long.MinValue }, DoubleBits(nameof(CPU.Tanh), dz));
        Assert.Equal(new long[] { 0L, long.MinValue }, DoubleBits(nameof(CPU.Sin), dz));
    }

    static readonly int[] Axis0 = new int[] { 0 };

    [Fact]
    public void ReduceSumExceptional_MatchesOrt()
    {
        // ORT 1.29 float and double: NaN poisons the sum; same-signed
        // infinities survive; opposing infinities cancel to NaN.
        var rows = new float[][]
        {
            new float[] { 1f, float.NaN, 3f }, new float[] { float.PositiveInfinity, 1f, 2f },
            new float[] { float.NegativeInfinity, 1f, 2f }, new float[] { float.PositiveInfinity, float.NegativeInfinity, 1f },
            new float[] { float.PositiveInfinity, float.PositiveInfinity, 1f }, new float[] { float.NegativeInfinity, float.NegativeInfinity, 1f },
            new float[] { 1f, 2f, 3f },
        };
        var axes = DenseTensor<int>.OfValues(Axis0);
        var fsumExpected = new float[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity, float.NaN, float.PositiveInfinity, float.NegativeInfinity, 6f };
        for (int i = 0; i < rows.Length; i++)
        {
            var r = CPU.ReduceSum(DenseTensor<float>.OfValues(rows[i]), axes, 0, 0, null);
            Assert.Equal(OpStatus.Success, r.Status);
            float y = ((Tensor<float>)r.Outputs![0]).ToArray()[0];
            if (float.IsNaN(fsumExpected[i])) Assert.True(float.IsNaN(y));
            else Assert.Equal(fsumExpected[i], y);
        }
        var drows = new double[][]
        {
            new double[] { 1.0, double.NaN, 3.0 }, new double[] { double.PositiveInfinity, 1.0, 2.0 },
            new double[] { double.NegativeInfinity, 1.0, 2.0 }, new double[] { double.PositiveInfinity, double.NegativeInfinity, 1.0 },
            new double[] { double.PositiveInfinity, double.PositiveInfinity, 1.0 }, new double[] { double.NegativeInfinity, double.NegativeInfinity, 1.0 },
            new double[] { 1.0, 2.0, 3.0 },
        };
        var dsumExpected = new double[] { double.NaN, double.PositiveInfinity, double.NegativeInfinity, double.NaN, double.PositiveInfinity, double.NegativeInfinity, 6.0 };
        for (int i = 0; i < drows.Length; i++)
        {
            var r = CPU.ReduceSum(DenseTensor<double>.OfValues(drows[i]), axes, 0, 0, null);
            Assert.Equal(OpStatus.Success, r.Status);
            double y = ((Tensor<double>)r.Outputs![0]).ToArray()[0];
            if (double.IsNaN(dsumExpected[i])) Assert.True(double.IsNaN(y));
            else Assert.Equal(dsumExpected[i], y);
        }
    }

    [Fact]
    public void ReduceMeanExceptional_MatchesOrt()
    {
        // ORT 1.29 float and double: same exceptional table as the sum
        // (mean is sum/count); the finite row checks the quotient.
        var axes = DenseTensor<int>.OfValues(Axis0);
        var rows = new float[][]
        {
            new float[] { 1f, float.NaN, 3f }, new float[] { float.PositiveInfinity, 1f, 2f },
            new float[] { float.NegativeInfinity, 1f, 2f }, new float[] { float.PositiveInfinity, float.NegativeInfinity, 1f },
            new float[] { float.PositiveInfinity, float.PositiveInfinity, 1f }, new float[] { float.NegativeInfinity, float.NegativeInfinity, 1f },
            new float[] { 1f, 2f, 3f },
        };
        var fmeanExpected = new float[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity, float.NaN, float.PositiveInfinity, float.NegativeInfinity, 2f };
        for (int i = 0; i < rows.Length; i++)
        {
            var r = CPU.ReduceMean(DenseTensor<float>.OfValues(rows[i]), axes, 0, 0, null);
            Assert.Equal(OpStatus.Success, r.Status);
            float y = ((Tensor<float>)r.Outputs![0]).ToArray()[0];
            if (float.IsNaN(fmeanExpected[i])) Assert.True(float.IsNaN(y));
            else Assert.Equal(fmeanExpected[i], y);
        }
        var drows = new double[][]
        {
            new double[] { 1.0, double.NaN, 3.0 }, new double[] { double.PositiveInfinity, 1.0, 2.0 },
            new double[] { double.NegativeInfinity, 1.0, 2.0 }, new double[] { double.PositiveInfinity, double.NegativeInfinity, 1.0 },
            new double[] { double.PositiveInfinity, double.PositiveInfinity, 1.0 }, new double[] { double.NegativeInfinity, double.NegativeInfinity, 1.0 },
            new double[] { 1.0, 2.0, 3.0 },
        };
        var dmeanExpected = new double[] { double.NaN, double.PositiveInfinity, double.NegativeInfinity, double.NaN, double.PositiveInfinity, double.NegativeInfinity, 2.0 };
        for (int i = 0; i < drows.Length; i++)
        {
            var r = CPU.ReduceMean(DenseTensor<double>.OfValues(drows[i]), axes, 0, 0, null);
            Assert.Equal(OpStatus.Success, r.Status);
            double y = ((Tensor<double>)r.Outputs![0]).ToArray()[0];
            if (double.IsNaN(dmeanExpected[i])) Assert.True(double.IsNaN(y));
            else Assert.Equal(dmeanExpected[i], y);
        }
    }

    [Fact]
    public void ReduceMaxExceptional_MatchesOrt()
    {
        // ORT 1.29 float and double: a leading NaN poisons (seed), but a
        // later NaN never wins a comparison, so [1,nan,3] is 3. Guarded
        // because a Math.Max fold would propagate every NaN instead.
        var axes = DenseTensor<int>.OfValues(Axis0);
        var rows = new float[][]
        {
            new float[] { 1f, float.NaN, 3f }, new float[] { float.PositiveInfinity, 1f, 2f },
            new float[] { float.NegativeInfinity, 1f, 2f }, new float[] { float.PositiveInfinity, float.NegativeInfinity, 1f },
            new float[] { float.PositiveInfinity, float.PositiveInfinity, 1f }, new float[] { float.NegativeInfinity, float.NegativeInfinity, 1f },
            new float[] { 1f, 2f, 3f }, new float[] { float.NaN, 1f, 3f },
            new float[] { 1f, 3f, float.NaN }, new float[] { float.NaN, float.NaN, float.NaN },
            new float[] { float.NegativeInfinity, float.NaN, 1f }, new float[] { 1f, float.NaN, float.NegativeInfinity },
        };
        var expected = new float[] { 3f, float.PositiveInfinity, 2f, float.PositiveInfinity, float.PositiveInfinity, 1f, 3f, float.NaN, 3f, float.NaN, 1f, 1f };
        for (int i = 0; i < rows.Length; i++)
        {
            var r = CPU.ReduceMax(DenseTensor<float>.OfValues(rows[i]), axes, 0, null, null);
            Assert.Equal(OpStatus.Success, r.Status);
            float y = ((Tensor<float>)r.Outputs![0]).ToArray()[0];
            if (float.IsNaN(expected[i])) Assert.True(float.IsNaN(y));
            else Assert.Equal(expected[i], y);
        }
        var drows = new double[][]
        {
            new double[] { 1.0, double.NaN, 3.0 }, new double[] { double.PositiveInfinity, 1.0, 2.0 },
            new double[] { double.NegativeInfinity, 1.0, 2.0 }, new double[] { double.PositiveInfinity, double.NegativeInfinity, 1.0 },
            new double[] { double.PositiveInfinity, double.PositiveInfinity, 1.0 }, new double[] { double.NegativeInfinity, double.NegativeInfinity, 1.0 },
            new double[] { 1.0, 2.0, 3.0 }, new double[] { double.NaN, 1.0, 3.0 },
            new double[] { 1.0, 3.0, double.NaN }, new double[] { double.NaN, double.NaN, double.NaN },
            new double[] { double.NegativeInfinity, double.NaN, 1.0 }, new double[] { 1.0, double.NaN, double.NegativeInfinity },
        };
        var dexpected = new double[] { 3.0, double.PositiveInfinity, 2.0, double.PositiveInfinity, double.PositiveInfinity, 1.0, 3.0, double.NaN, 3.0, double.NaN, 1.0, 1.0 };
        for (int i = 0; i < drows.Length; i++)
        {
            var r = CPU.ReduceMax(DenseTensor<double>.OfValues(drows[i]), axes, 0, null, null);
            Assert.Equal(OpStatus.Success, r.Status);
            double y = ((Tensor<double>)r.Outputs![0]).ToArray()[0];
            if (double.IsNaN(dexpected[i])) Assert.True(double.IsNaN(y));
            else Assert.Equal(dexpected[i], y);
        }
    }
}