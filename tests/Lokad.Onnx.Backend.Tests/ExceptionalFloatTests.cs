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
    public void ErfInfinite_YieldsSignedOne()
    {
        var result = CPU.Erf(DenseTensor<float>.OfValues(new float[] { float.PositiveInfinity, float.NegativeInfinity, 0f }), null, null);
        Assert.Equal(OpStatus.Success, result.Status);
        Assert.Equal(new float[] { 1f, -1f, 0f }, ((Tensor<float>)result.Outputs![0]).ToArray());
    }
}
