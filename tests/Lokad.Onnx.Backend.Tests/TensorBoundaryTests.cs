using System.Collections.Generic;

namespace Lokad.Onnx.Backend.Tests;

public class TensorBoundaryTests
{
    static void AssertThrowsArg(System.Action a)
    {
        Assert.ThrowsAny<System.ArgumentException>(a);
    }

    [Fact]
    public void EmptyPrintData_ReturnsBrackets()
    {
        // PrintData divided element count by the last extent: 0/0 on any
        // tensor with a zero last dim. Empties render as [] instead.
        Assert.Equal("[]", DenseTensor<float>.OfShape(2, 0).PrintData(false));
        Assert.Equal("[]", DenseTensor<float>.OfShape(0).PrintData(true));
        Assert.Equal("[1.00000,2.00000]", DenseTensor<float>.OfValues(new float[] { 1f, 2f }).PrintData(false));
    }

    [Fact]
    public void NegativeDimension_ThrowsBeforeUnsafe()
    {
        AssertThrowsArg(() => new DenseTensor<float>(new int[] { -1, 4 }));
        AssertThrowsArg(() => ArrayUtilities.GetStrides(new int[] { -1, 4 }));
    }

    [Fact]
    public void OverflowingStrides_ThrowInsteadOfWrapping()
    {
        AssertThrowsArg(() => ArrayUtilities.GetStrides(new int[] { 100000, 100000 }));
        AssertThrowsArg(() => ArrayUtilities.ComputeOffsetForReduction(new int[] { 100000, 100000 }, 0));
    }

    [Fact]
    public void ConvGroupMismatch_ThrowsBeforePinning()
    {
        var x = DenseTensor<float>.OfShape(1, 4, 4, 4); x.Fill(1f);
        var w = DenseTensor<float>.OfShape(8, 2, 3, 3); w.Fill(1f);
        AssertThrowsArg(() => Tensor<float>.Conv2D(x, w, 3, MathOps.PadType.Valid, null, null, null, null, null));
    }

    [Fact]
    public void ConvKernelAndBiasMismatch_ThrowBeforePinning()
    {
        var x = DenseTensor<float>.OfShape(1, 2, 5, 5); x.Fill(1f);
        var wOk = DenseTensor<float>.OfShape(2, 2, 3, 3); wOk.Fill(1f);
        var biasBad = DenseTensor<float>.OfValues(new float[] { 1f });
        AssertThrowsArg(() => Tensor<float>.Conv2D(x, wOk, 1, bias: biasBad, padtype: MathOps.PadType.Valid, padvalue: null, kernelshape: null, strides: null, dilations: null));
        var wBad = DenseTensor<float>.OfShape(2, 1, 3, 3); wBad.Fill(1f);
        AssertThrowsArg(() => Tensor<float>.Conv2D(x, wBad, 1, MathOps.PadType.Valid, null, null, null, null, null));
        AssertThrowsArg(() => Tensor<float>.Conv2D(x, wOk, 1, strides: new int[] { 0, 1 }, padtype: MathOps.PadType.Valid, padvalue: null, bias: null, kernelshape: null, dilations: null));
    }

    static Tensor<float> ReferenceBatchedMatMul(Tensor<float> x, Tensor<float> y)
    {
        var dx = x.ToDenseTensor();
        var dy = y.ToDenseTensor();
        return Tensor<float>.MatMul(dx, dy, TensorExecutionOptions.Scalar);
    }

    [Fact]
    public void SlicedBatchedMatMul_MatchesDenseReference()
    {
        var baseX = DenseTensor<float>.OfShape(2, 2, 4);
        for (int i = 0; i < baseX.Length; i++) baseX.SetValue(i, (float)(i + 1));
        var x = baseX.Slice(new SliceIndex(null, null), new SliceIndex(null, null), new SliceIndex(0, 2));
        var y = DenseTensor<float>.OfShape(2, 2, 2);
        for (int i = 0; i < y.Length; i++) y.SetValue(i, (float)(i + 1));
        var actual = Tensor<float>.MatMul(x, y, TensorExecutionOptions.Scalar);
        var expected = ReferenceBatchedMatMul(x, y);
        var a = actual.ToArray(); var e = expected.ToArray();
        Assert.Equal(e.Length, a.Length);
        for (int i = 0; i < e.Length; i++) Assert.Equal(e[i], a[i], 4);
    }

    [Fact]
    public void BroadcastBatchedMatMul_MatchesDenseReference()
    {
        var single = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f } });
        var x = Tensor<float>.BroadcastTo(single, new int[] { 2, 2, 2 });
        var y = DenseTensor<float>.OfShape(2, 2, 2);
        for (int i = 0; i < y.Length; i++) y.SetValue(i, (float)(i + 1));
        var actual = Tensor<float>.MatMul(x, y, TensorExecutionOptions.Scalar);
        var expected = ReferenceBatchedMatMul(x, y);
        var a = actual.ToArray(); var e = expected.ToArray();
        Assert.Equal(e.Length, a.Length);
        for (int i = 0; i < e.Length; i++) Assert.Equal(e[i], a[i], 4);
    }
}

