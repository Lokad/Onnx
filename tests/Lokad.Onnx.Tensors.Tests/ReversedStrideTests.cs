using System;

namespace Lokad.Onnx.Tensors.Tests;

// Column-major (reversed-stride) tensors must expose logical contents
// through every ordered surface: ToArray, ToDenseTensor normalization,
// and provider kernels agree elementwise with the logical oracle.
public class ReversedStrideTests
{
    // Column-major physical [-2,-1,-0.5,0,0.5,2] with dims [2,3] decodes to
    // logical [[-2,-0.5,0.5],[-1,0,2]].
    static DenseTensor<double> ReversedInput() =>
        new DenseTensor<double>(new double[] { -2.0, -1.0, -0.5, 0.0, 0.5, 2.0 }, new[] { 2, 3 }, true);

    static readonly double[] Logical = new double[] { -2.0, -0.5, 0.5, -1.0, 0.0, 2.0 };

    [Fact]
    public void ToArray_ReturnsLogicalOrder()
    {
        Assert.Equal(Logical, ReversedInput().ToArray());
    }

    [Fact]
    public void ToDenseTensor_MatchesToArray()
    {
        var dense = ReversedInput().ToDenseTensor();
        Assert.False(dense.IsReversedStride);
        Assert.Equal(Logical, dense.ToArray());
    }

    [Fact]
    public void ProviderTanh_MatchesLogicalOracle()
    {
        var r = CPUExecutionProvider.Tanh(ReversedInput(), null);
        Assert.Equal(OpStatus.Success, r.Status);
        var actual = ((Tensor<double>)r.Outputs[0]).ToArray();
        Assert.Equal(Logical.Length, actual.Length);
        for (int i = 0; i < Logical.Length; i++)
            Assert.True(Math.Abs(actual[i] - Math.Tanh(Logical[i])) < 1e-12, "tanh[" + i + "]");
    }

    [Fact]
    public void ProviderSigmoid_MatchesLogicalOracle()
    {
        var r = CPUExecutionProvider.Sigmoid(ReversedInput(), null);
        Assert.Equal(OpStatus.Success, r.Status);
        var actual = ((Tensor<double>)r.Outputs[0]).ToArray();
        for (int i = 0; i < Logical.Length; i++)
            Assert.True(Math.Abs(actual[i] - (1.0 / (1.0 + Math.Exp(-Logical[i])))) < 1e-12, "sigmoid[" + i + "]");
    }
}
