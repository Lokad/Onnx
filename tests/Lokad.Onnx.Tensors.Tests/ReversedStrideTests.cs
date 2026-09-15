using System;

namespace Lokad.Onnx.Tensors.Tests;

// Column-major (reversed-stride) tensors must expose logical contents
// through every ordered surface: ToArray decodes through the strides and
// ToDenseTensor normalization agrees. Ported from the proven voice-branch
// fix (S04); provider-kernel logical coverage there stays on that line
// (no Sigmoid operator exists here yet).
public class ReversedStrideTests
{
    // Column-major physical [-2,-1,-0.5,0,0.5,2] with dims [2,3] decodes to
    // logical [[-2,-0.5,0.5],[-1,0,2]].
    static DenseTensor<double> ReversedInput() =>
        new DenseTensor<double>(new Memory<double>(new double[] { -2.0, -1.0, -0.5, 0.0, 0.5, 2.0 }), new[] { 2, 3 }, true);

    static readonly double[] Logical = new double[] { -2.0, -0.5, 0.5, -1.0, 0.0, 2.0 };

    [Fact]
    public void ToArray_ReturnsLogicalOrder()
    {
        Assert.Equal(Logical, (double[])((ITensor)ReversedInput()).ToArray());
    }

    [Fact]
    public void ToDenseTensor_MatchesToArray()
    {
        var dense = ReversedInput().ToDenseTensor();
        Assert.False(dense.IsReversedStride);
        Assert.Equal(Logical, (double[])((ITensor)dense).ToArray());
    }
}
