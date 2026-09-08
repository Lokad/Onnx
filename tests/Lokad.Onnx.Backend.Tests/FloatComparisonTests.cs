namespace Lokad.Onnx.Backend.Tests;

public class FloatComparisonTests
{
    [Fact]
    public void Equal_NanAlwaysFalse()
    {
        var x = DenseTensor<float>.OfValues(new float[] { float.NaN, float.NaN, 1f, float.NaN });
        var y = DenseTensor<float>.OfValues(new float[] { float.NaN, 1f, float.NaN, float.NaN });
        var r = Tensor<float>.Equal(x, y);
        Assert.Equal(new bool[] { false, false, false, false }, r.ToArray());
    }

    [Fact]
    public void Equal_InfinitiesAndSignedZero()
    {
        var x = DenseTensor<float>.OfValues(new float[] { float.PositiveInfinity, float.NegativeInfinity, 0f, 1f, 1f });
        var y = DenseTensor<float>.OfValues(new float[] { float.PositiveInfinity, float.NegativeInfinity, -0f, 1f, 2f });
        var r = Tensor<float>.Equal(x, y);
        Assert.Equal(new bool[] { true, true, true, true, false }, r.ToArray());
    }

    [Fact]
    public void Less_NanAlwaysFalse()
    {
        var x = DenseTensor<float>.OfValues(new float[] { float.NaN, 0f, float.NaN });
        var y = DenseTensor<float>.OfValues(new float[] { 0f, float.NaN, float.NaN });
        var r = Tensor<float>.Less(x, y);
        Assert.Equal(new bool[] { false, false, false }, r.ToArray());
    }

    [Fact]
    public void Less_InfinitiesAndSignedZero()
    {
        var x = DenseTensor<float>.OfValues(new float[] { float.NegativeInfinity, 1f, float.PositiveInfinity, 0f, -0f, 1f, 2f });
        var y = DenseTensor<float>.OfValues(new float[] { float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity, -0f, 0f, 2f, 1f });
        var r = Tensor<float>.Less(x, y);
        Assert.Equal(new bool[] { true, true, false, false, false, true, false }, r.ToArray());
    }

    [Fact]
    public void Double_NanSemantics()
    {
        var x = DenseTensor<double>.OfValues(new double[] { double.NaN, double.NaN, 0.0 });
        var y = DenseTensor<double>.OfValues(new double[] { double.NaN, 0.0, double.NaN });
        Assert.Equal(new bool[] { false, false, false }, Tensor<double>.Equal(x, y).ToArray());
        Assert.Equal(new bool[] { false, false, false }, Tensor<double>.Less(x, y).ToArray());
    }

    [Fact]
    public void Broadcast_ComparisonsFollowBroadcastShape()
    {
        var x = DenseTensor<float>.OfValues(new float[,] { { 1f }, { 2f } });
        var y = DenseTensor<float>.OfValues(new float[,] { { 10f, 20f } });
        var eq = CPUExecutionProvider.Equal(x, y, null);
        Assert.Equal(OpStatus.Success, eq.Status);
        Assert.Equal(new int[] { 2, 2 }, ((Tensor<bool>)eq.Outputs[0]).Dimensions.ToArray());
        Assert.Equal(new bool[] { false, false, false, false }, ((Tensor<bool>)eq.Outputs[0]).ToArray());
        var lt = CPUExecutionProvider.Less(x, y, null);
        Assert.Equal(OpStatus.Success, lt.Status);
        Assert.Equal(new bool[] { true, true, true, true }, ((Tensor<bool>)lt.Outputs[0]).ToArray());
    }

    [Fact]
    public void Relu_NanPropagatesAndPreservesSignedZero()
    {
        var x = DenseTensor<float>.OfValues(new float[] { float.NaN, -1f, -0f, 0f, 2.5f, float.NegativeInfinity, float.PositiveInfinity });
        var r = Tensor<float>.Relu(x).ToArray();
        Assert.True(float.IsNaN(r[0]));
        Assert.Equal(0f, r[1]);
        Assert.Equal(System.BitConverter.SingleToInt32Bits(-0f), System.BitConverter.SingleToInt32Bits(r[2]));
        Assert.Equal(0f, r[3]);
        Assert.Equal(2.5f, r[4]);
        Assert.Equal(0f, r[5]);
        Assert.Equal(float.PositiveInfinity, r[6]);
    }

    [Fact]
    public void Abs_NanStaysPositive()
    {
        var posNan = System.BitConverter.Int32BitsToSingle(0x7FC00000);
        var r = Tensor<float>.Abs(DenseTensor<float>.OfValues(new float[] { float.NaN, posNan, -1.5f, -0f })).ToArray();
        Assert.True(float.IsNaN(r[0]));
        Assert.False(System.BitConverter.SingleToInt32Bits(r[0]) < 0);
        Assert.True(float.IsNaN(r[1]));
        Assert.False(System.BitConverter.SingleToInt32Bits(r[1]) < 0);
        Assert.Equal(1.5f, r[2]);
        Assert.Equal(0f, r[3]);
    }

    [Fact]
    public void Double_AdjacentOpsMatchOracle()
    {
        var rel = Tensor<double>.Relu(DenseTensor<double>.OfValues(new double[] { double.NaN, -2.0, 3.0 })).ToArray();
        Assert.True(double.IsNaN(rel[0]));
        Assert.Equal(0.0, rel[1]);
        Assert.Equal(3.0, rel[2]);
        Assert.True(double.IsNaN(Tensor<double>.Abs(DenseTensor<double>.OfValues(new double[] { double.NaN })).ToArray()[0]));
        Assert.Equal(-2.0, Tensor<double>.ReduceMax(DenseTensor<double>.OfValues(new double[] { -5.0, -2.0 }), null, null, null).ToArray()[0]);
    }


    [Fact]
    public void ReduceMax_NegativesNanAndEmpty()
    {
        Assert.Equal(-2f, Tensor<float>.ReduceMax(DenseTensor<float>.OfValues(new float[] { -5f, -2f }), null, null, null).ToArray()[0]);
        Assert.True(float.IsNaN(Tensor<float>.ReduceMax(DenseTensor<float>.OfValues(new float[] { float.NaN, 1f }), null, null, null).ToArray()[0]));
        Assert.Equal(1f, Tensor<float>.ReduceMax(DenseTensor<float>.OfValues(new float[] { 1f, float.NaN }), null, null, null).ToArray()[0]);
        Assert.Equal(float.NegativeInfinity, Tensor<float>.ReduceMax(DenseTensor<float>.OfValues(new float[0]), null, null, null).ToArray()[0]);
    }
}
