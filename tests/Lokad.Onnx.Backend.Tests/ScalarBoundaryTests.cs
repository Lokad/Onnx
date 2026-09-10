using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Pins rank-0 scalar boundaries against ORT 1.29 probe values: scalar
/// elementwise pairs compute, scalar reductions are identity, and scalar
/// Softmax plus MatMul reject on both sides of the reference.
/// </summary>
public class ScalarBoundaryTests
{
    static DenseTensor<float> Scalar(float value)
    {
        var s = DenseTensor<float>.OfShape();
        s.SetValue(0, value);
        return s;
    }

    static float[] Apply(OpType op, float x, float y)
    {
        var result = op switch
        {
            OpType.Add => CPU.Add(Scalar(x), Scalar(y), null, null),
            OpType.Sub => CPU.Sub(Scalar(x), Scalar(y), null),
            OpType.Mul => CPU.Mul(Scalar(x), Scalar(y), null, null),
            OpType.Div => CPU.Div(Scalar(x), Scalar(y), null, null),
            _ => throw new ArgumentOutOfRangeException(nameof(op)),
        };
        Assert.Equal(OpStatus.Success, result.Status);
        return ((Tensor<float>)result.Outputs![0]).ToArray();
    }

    [Fact]
    public void ScalarElementwise_MatchesReference()
    {
        Assert.Equal(new float[] { 5f }, Apply(OpType.Add, 3f, 2f));
        Assert.Equal(new float[] { 1f }, Apply(OpType.Sub, 3f, 2f));
        Assert.Equal(new float[] { 6f }, Apply(OpType.Mul, 3f, 2f));
        Assert.Equal(new float[] { 1.5f }, Apply(OpType.Div, 3f, 2f));
        Assert.Equal(new float[] { float.PositiveInfinity }, Apply(OpType.Div, 1f, 0f));
    }

    static DenseTensor<double> ScalarDouble(double value)
    {
        var s = DenseTensor<double>.OfShape();
        s.SetValue(0, value);
        return s;
    }

    static double[] ApplyDouble(OpType op, double x, double y)
    {
        var result = op switch
        {
            OpType.Add => CPU.Add(ScalarDouble(x), ScalarDouble(y), null, null),
            OpType.Sub => CPU.Sub(ScalarDouble(x), ScalarDouble(y), null),
            OpType.Mul => CPU.Mul(ScalarDouble(x), ScalarDouble(y), null, null),
            OpType.Div => CPU.Div(ScalarDouble(x), ScalarDouble(y), null, null),
            OpType.Pow => CPU.Pow(ScalarDouble(x), ScalarDouble(y), null),
            _ => throw new ArgumentOutOfRangeException(nameof(op)),
        };
        Assert.Equal(OpStatus.Success, result.Status);
        return ((Tensor<double>)result.Outputs![0]).ToArray();
    }

    [Fact]
    public void ScalarDoubleElementwise_MatchesReference()
    {
        // ORT 1.29 rank-0 double: 5, 1, 6, 1.5, 9.
        Assert.Equal(new double[] { 5.0 }, ApplyDouble(OpType.Add, 3.0, 2.0));
        Assert.Equal(new double[] { 1.0 }, ApplyDouble(OpType.Sub, 3.0, 2.0));
        Assert.Equal(new double[] { 6.0 }, ApplyDouble(OpType.Mul, 3.0, 2.0));
        Assert.Equal(new double[] { 1.5 }, ApplyDouble(OpType.Div, 3.0, 2.0));
        Assert.Equal(new double[] { 9.0 }, ApplyDouble(OpType.Pow, 3.0, 2.0));
    }

    [Fact]
    public void ScalarPow_MatchesReference()
    {
        // ORT 1.29 rank-0: Pow(-8,3)=-512, Pow(3,2)=9.
        var neg = CPU.Pow(Scalar(-8f), Scalar(3f), null);
        Assert.Equal(OpStatus.Success, neg.Status);
        Assert.Equal(new float[] { -512f }, ((Tensor<float>)neg.Outputs![0]).ToArray());
        var sq = CPU.Pow(Scalar(3f), Scalar(2f), null);
        Assert.Equal(OpStatus.Success, sq.Status);
        Assert.Equal(new float[] { 9f }, ((Tensor<float>)sq.Outputs![0]).ToArray());
    }

    [Fact]
    public void ScalarReduction_IsIdentity()
    {
        var mean = Tensor<float>.ReduceMean(Scalar(5f), null, false, false);
        Assert.Equal(new int[0], mean.Dimensions.ToArray());
        Assert.Equal(new float[] { 5f }, mean.ToArray());
        var max = Tensor<float>.ReduceMax(Scalar(5f), null, false, false);
        Assert.Equal(new float[] { 5f }, max.ToArray());
        var sum = Tensor<float>.ReduceSum(Scalar(5f), null, false, false);
        Assert.Equal(new float[] { 5f }, sum.ToArray());
    }

    [Fact]
    public void ScalarSoftmax_Rejected()
    {
        Assert.Throws<ArgumentException>(() => Tensor<float>.Softmax(Scalar(5f), -1, null, 13));
    }

    [Fact]
    public void ScalarMatMul_Rejected()
    {
        Assert.Throws<ArgumentException>(() => Tensor<float>.MatMul(Scalar(3f), Scalar(2f), TensorExecutionOptions.Scalar));
    }
}
