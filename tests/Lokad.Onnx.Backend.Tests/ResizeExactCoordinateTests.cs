namespace Lokad.Onnx.Backend.Tests;

public class ResizeExactCoordinateTests
{
    [Theory]
    [InlineData(false, false, false)]
    [InlineData(false, true, false)]
    [InlineData(true, false, false)]
    [InlineData(true, true, false)]
    [InlineData(false, false, true)]
    [InlineData(false, true, true)]
    [InlineData(true, false, true)]
    [InlineData(true, true, true)]
    public void IdentitySamplingPreservesSourceBitsWithNonfiniteNeighbors(bool wide, bool nan, bool useScales)
    {
        double exceptional = nan ? double.NaN : double.PositiveInfinity;
        double[] input = [-0d, exceptional, 3d, 4d];
        var result = Resize(wide, input, 2, 2, 2, 2, "asymmetric", useScales);
        Assert.Equal(input, result);
        Assert.Equal(BitConverter.DoubleToInt64Bits(-0d), BitConverter.DoubleToInt64Bits(result[0]));
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void ResamplingRetainsNativeZeroWeightArithmetic(bool wide, bool horizontal)
    {
        double[] input = horizontal ? [2d, 6d, double.NaN, double.PositiveInfinity]
            : [2d, double.NaN, 6d, double.PositiveInfinity];
        var output = Resize(wide, input, 2, 2, horizontal ? 2 : 3, horizontal ? 3 : 2, "align_corners");
        Assert.True(double.IsNaN(output[horizontal ? 1 : 2]));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NonzeroWeightsStillPropagateExceptionalValuesAndFiniteBlendsStayUnchanged(bool wide)
    {
        var infinity = Resize(wide, [2d, double.PositiveInfinity, 6d, 4d], 2, 2, 3, 3, "align_corners");
        Assert.Equal(double.PositiveInfinity, infinity[4]);
        var nan = Resize(wide, [2d, double.NaN, 6d, 4d], 2, 2, 3, 3, "align_corners");
        Assert.True(double.IsNaN(nan[4]));
        Assert.Equal(new[] { 1d, 1.5d, 2d, 2d, 2.5d, 3d, 3d, 3.5d, 4d },
            Resize(wide, [1d, 2d, 3d, 4d], 2, 2, 3, 3, "align_corners"));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ClampedHalfPixelEdgesStillReadOnlyTheirActiveColumn(bool wide)
    {
        var result = Resize(wide, [2d, double.NaN, 6d, double.NaN], 2, 2, 4, 4, "half_pixel");
        Assert.Equal(new[] { 2d, 3d, 5d, 6d }, new[] { result[0], result[4], result[8], result[12] });
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NonunitScalesWithUnchangedSizesStillCopy(bool wide)
    {
        ITensor input = wide ? new DenseTensor<double>(new[] { 1d, double.PositiveInfinity, 3d, 4d }, [1, 1, 2, 2])
            : new DenseTensor<float>(new[] { 1f, float.PositiveInfinity, 3f, 4f }, [1, 1, 2, 2]);
        ITensor scales = wide ? DenseTensor<double>.OfValues(new[] { 1d, 1d, 1d, 1.25d })
            : DenseTensor<float>.OfValues(new[] { 1f, 1f, 1f, 1.25f });
        var result = CPUExecutionProvider.Resize(input, null, scales, null, "linear", "asymmetric", null, null, null, null);
        Assert.Equal(OpStatus.Success, result.Status);
        var values = wide ? ((Tensor<double>)result.Outputs![0]).ToArray()
            : ((Tensor<float>)result.Outputs![0]).ToArray().Select(v => (double)v).ToArray();
        Assert.Equal(new[] { 1d, double.PositiveInfinity, 3d, 4d }, values);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void IdentityResizeCopiesLogicalContentsWithoutAliasing(bool reversed)
    {
        var input = new DenseTensor<float>(reversed ? new[] { 1f, 3f, 2f, 4f } : new[] { 1f, 2f, 3f, 4f },
            new[] { 1, 1, 2, 2 }, reverseStride: reversed);
        var result = Tensor<float>.Resize(input, [1, 1, 2, 2], MathOps.ResizeMode.Linear,
            MathOps.ResizeCoordinateTransformation.Asymmetric, MathOps.ResizeNearestMode.Floor, -.75f, null);
        input.SetValue(0, 99f);
        Assert.Equal(new[] { 1f, 2f, 3f, 4f }, result.ToArray());
    }

    static double[] Resize(bool wide, double[] values, int height, int width, int outputHeight, int outputWidth, string coordinates, bool useScales = false)
    {
        ITensor input = wide ? new DenseTensor<double>(values, [1, 1, height, width])
            : new DenseTensor<float>(values.Select(v => (float)v).ToArray(), [1, 1, height, width]);
        var result = CPUExecutionProvider.Resize(input, null, useScales ? DenseTensor<float>.OfValues(new[] { 1f, 1f, 1f, 1f }) : null,
            useScales ? null : DenseTensor<long>.OfValues(new long[] { 1, 1, outputHeight, outputWidth }), "linear", coordinates, null, null, null, null);
        Assert.Equal(OpStatus.Success, result.Status);
        return wide ? ((Tensor<double>)result.Outputs![0]).ToArray()
            : ((Tensor<float>)result.Outputs![0]).ToArray().Select(v => (double)v).ToArray();
    }
}
