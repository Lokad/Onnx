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
