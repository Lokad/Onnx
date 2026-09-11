namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// The vectorized Relu arms agree bit-for-bit with the scalar contract on
/// signs, signed zero, infinities and NaN, in float and double, across modes.
/// </summary>
public class ReluVectorAgreementTests
{
    static readonly float[] EdgesF = new float[]
    {
        -123.5f, -1f, -float.Epsilon, -0f, 0f, float.Epsilon, 1f, 123.5f,
        float.PositiveInfinity, float.NegativeInfinity, float.NaN, -float.NaN, 7f, -7f, 0.5f, -0.5f,
    };

    static readonly double[] EdgesD = new double[]
    {
        -123.5, -1.0, -double.Epsilon, -0.0, 0.0, double.Epsilon, 1.0, 123.5,
        double.PositiveInfinity, double.NegativeInfinity, double.NaN, -double.NaN, 7.0, -7.0, 0.5, -0.5,
    };

    [Fact]
    public void FloatVector_MatchesScalarBitwise()
    {
        var rnd = new Random(41);
        var data = new float[130];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rnd.NextDouble() * 4 - 2);
        Array.Copy(EdgesF, 0, data, 0, EdgesF.Length);
        var x = DenseTensor<float>.OfValues((System.ReadOnlySpan<float>)data, new int[] { 2, 65 });
        var scalar = Tensor<float>.Relu(x);
        var vec = Tensor<float>.Relu(x, TensorExecutionOptions.Auto);
        Assert.Equal(scalar.ToArray(), vec.ToArray());
        var bits = vec.ToArray().Select(System.BitConverter.SingleToInt32Bits).ToArray();
        var refBits = scalar.ToArray().Select(System.BitConverter.SingleToInt32Bits).ToArray();
        Assert.Equal(refBits, bits);
    }

    [Fact]
    public void DoubleVector_MatchesScalarBitwise()
    {
        var rnd = new Random(43);
        var data = new double[66];
        for (int i = 0; i < data.Length; i++) data[i] = rnd.NextDouble() * 4 - 2;
        Array.Copy(EdgesD, 0, data, 0, EdgesD.Length);
        var x = DenseTensor<double>.OfValues((System.ReadOnlySpan<double>)data, new int[] { 6, 11 });
        var scalar = Tensor<double>.Relu(x);
        var vec = Tensor<double>.Relu(x, TensorExecutionOptions.Auto);
        Assert.Equal(
            scalar.ToArray().Select(System.BitConverter.DoubleToInt64Bits).ToArray(),
            vec.ToArray().Select(System.BitConverter.DoubleToInt64Bits).ToArray());
    }

    [Fact]
    public void FloatEdges_PinContract()
    {
        var x = DenseTensor<float>.OfValues((System.ReadOnlySpan<float>)EdgesF, new int[] { 4, 4 });
        var y = Tensor<float>.Relu(x, TensorExecutionOptions.Auto).ToArray();
        Assert.Equal(0f, y[0]);
        Assert.Equal(0f, y[1]);
        Assert.Equal(0f, y[2]);
        Assert.Equal(-0f, y[3]);
        Assert.Equal(0f, y[4]);
        Assert.Equal(float.Epsilon, y[5]);
        Assert.Equal(1f, y[6]);
        Assert.Equal(123.5f, y[7]);
        Assert.Equal(float.PositiveInfinity, y[8]);
        Assert.Equal(0f, y[9]);
        Assert.True(float.IsNaN(y[10]));
        Assert.True(float.IsNaN(y[11]));
        Assert.Equal(System.BitConverter.SingleToInt32Bits(-0f), System.BitConverter.SingleToInt32Bits(y[3]));
    }

    [Fact]
    public void ProviderRelu_UsesVectorPathWithSameValues()
    {
        var r = CPUExecutionProvider.Relu(DenseTensor<float>.OfValues(new float[] { -2f, -0f, 0f, 3f, float.NaN }), null);
        Assert.Equal(OpStatus.Success, r.Status);
        var got = ((Tensor<float>)r.Outputs![0]).ToArray();
        Assert.Equal(0f, got[0]);
        Assert.Equal(-0f, got[1]);
        Assert.Equal(0f, got[2]);
        Assert.Equal(3f, got[3]);
        Assert.True(float.IsNaN(got[4]));
    }
}
