using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// First ConstantOfShape coverage anywhere: default and explicit fills,
/// zero extents, scalar output, and descriptive rejections, against ORT
/// 1.29 probe values.
/// </summary>
public class ConstantOfShapeBoundaryTests
{
    static Tensor<float> Run(long[] dims, float[]? value)
    {
        DenseTensor<float>? v = value is null ? null : DenseTensor<float>.OfValues(value);
        var result = CPU.ConstantOfShape(DenseTensor<long>.OfValues(dims), v, null);
        Assert.Equal(OpStatus.Success, result.Status);
        return (Tensor<float>)result.Outputs![0];
    }

    [Fact]
    public void DefaultValue_FillsZeros()
    {
        // ORT 1.29: zeros(2, 2).
        var y = Run(new long[] { 2L, 2L }, null);
        Assert.Equal(new int[] { 2, 2 }, y.Dimensions.ToArray());
        Assert.Equal(new float[] { 0f, 0f, 0f, 0f }, y.ToArray());
    }

    [Fact]
    public void ExplicitValue_FillsConstant()
    {
        // ORT 1.29: [7, 7, 7].
        Assert.Equal(new float[] { 7f, 7f, 7f }, Run(new long[] { 3L }, new float[] { 7f }).ToArray());
    }

    [Fact]
    public void DoubleValue_FillsConstant()
    {
        // ORT 1.29: 2x2 of 3.25.
        var result = CPU.ConstantOfShape(DenseTensor<long>.OfValues(new long[] { 2L, 2L }), DenseTensor<double>.OfValues(new double[] { 3.25 }), null);
        Assert.Equal(OpStatus.Success, result.Status);
        var y = (Tensor<double>)result.Outputs![0];
        Assert.Equal(new int[] { 2, 2 }, y.Dimensions.ToArray());
        Assert.Equal(new double[] { 3.25, 3.25, 3.25, 3.25 }, y.ToArray());
    }

    [Fact]
    public void ZeroDim_ReturnsEmpty()
    {
        // ORT 1.29: shape (2, 0), no elements.
        var y = Run(new long[] { 2L, 0L }, null);
        Assert.Equal(new int[] { 2, 0 }, y.Dimensions.ToArray());
        Assert.Empty(y.ToArray());
    }

    [Fact]
    public void EmptyDims_ReturnsScalar()
    {
        // ORT 1.29: no dims at all -> rank-0 zero.
        var result = CPU.ConstantOfShape(DenseTensor<long>.OfShape(0), null, null);
        Assert.Equal(OpStatus.Success, result.Status);
        var y = (Tensor<float>)result.Outputs![0];
        Assert.Equal(new int[0], y.Dimensions.ToArray());
        Assert.Equal(new float[] { 0f }, y.ToArray());
    }

    [Fact]
    public void NegativeDims_FailsCleanly()
    {
        // ORT 1.29 fails the run; Lokad returns a descriptive Failure.
        var r = CPU.ConstantOfShape(DenseTensor<long>.OfValues(new long[] { 2L, -1L }), null, null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("non-negative", r.Message ?? "");
    }

    [Fact]
    public void MultiElementValue_FailsCleanly()
    {
        var r = CPU.ConstantOfShape(
            DenseTensor<long>.OfValues(new long[] { 2L }),
            DenseTensor<float>.OfValues(new float[] { 1f, 2f }), null);
        Assert.Equal(OpStatus.Failure, r.Status);
        Assert.Contains("single element", r.Message ?? "");
    }
}
