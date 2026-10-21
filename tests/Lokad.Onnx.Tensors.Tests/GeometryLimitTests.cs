namespace Lokad.Onnx.Tensors.Tests;

public class GeometryLimitTests
{
    const long Bound = 1L << 48;

    [Theory]
    [InlineData(281474976710657L)]
    [InlineData(-281474976710657L)]
    [InlineData(long.MaxValue)]
    [InlineData(long.MinValue)]
    public void RangeRejectsOversizedArgumentsEvenWhenEmpty(long value)
    {
        Assert.ThrowsAny<ArgumentException>(() => Tensor<long>.Range(value, value, 1));
        Assert.ThrowsAny<ArgumentException>(() => Tensor<long>.Range(0, value, value));
        Assert.ThrowsAny<ArgumentException>(() => Tensor<long>.Range(0, 0, value));
        Assert.ThrowsAny<ArgumentException>(() => Tensor<double>.Range((double)value, value, 1d));
        float outside = value > 0 ? MathF.BitIncrement((float)Bound) : MathF.BitDecrement(-(float)Bound);
        Assert.ThrowsAny<ArgumentException>(() => Tensor<float>.Range(outside, outside, 1f));
    }

    [Fact]
    public void InclusiveRangeBoundariesRemainUsable()
    {
        Assert.Equal(new[] { Bound - 2, Bound - 1 }, Tensor<long>.Range(Bound - 2, Bound, 1).ToArray());
        Assert.Equal(new[] { -Bound + 2, -Bound + 1 }, Tensor<long>.Range(-Bound + 2, -Bound, -1).ToArray());
        Assert.Equal(new[] { -Bound, 0L }, Tensor<long>.Range(-Bound, Bound, Bound).ToArray());
        Assert.Equal(new[] { Bound, 0L }, Tensor<long>.Range(Bound, -Bound, -Bound).ToArray());
        Assert.Empty(Tensor<long>.Range(Bound, Bound, 1).ToArray());
        Assert.Equal(new[] { (double)Bound - 1 }, Tensor<double>.Range(Bound - 1d, Bound, 1).ToArray());
        Assert.Equal(new[] { -(float)Bound, 0f }, Tensor<float>.Range(-(float)Bound, (float)Bound, (float)Bound).ToArray());
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void RangeRejectsNonFiniteArguments(double value)
    {
        Assert.ThrowsAny<ArgumentException>(() => Tensor<double>.Range(value, 0, 1));
        Assert.ThrowsAny<ArgumentException>(() => Tensor<double>.Range(0, value, 1));
        Assert.ThrowsAny<ArgumentException>(() => Tensor<double>.Range(0, 0, value));
        Assert.ThrowsAny<ArgumentException>(() => Tensor<float>.Range((float)value, 0, 1));
        Assert.ThrowsAny<ArgumentException>(() => Tensor<float>.Range(0, (float)value, 1));
        Assert.ThrowsAny<ArgumentException>(() => Tensor<float>.Range(0, 0, (float)value));
    }

    [Fact]
    public void LongShapeProductsRespectTheGeometryBound()
    {
        Assert.Equal(Bound, ArrayUtilities.ComputeOffsetForReductionLong(new[] { 1 << 24, 1 << 24 }, 0));
        Assert.ThrowsAny<ArgumentException>(() => ArrayUtilities.ComputeOffsetForReductionLong(new[] { 1 << 24, (1 << 24) + 1 }, 0));
        Assert.ThrowsAny<ArgumentException>(() => ArrayUtilities.ComputeOffsetForReductionLong(new[] { int.MaxValue, int.MaxValue, int.MaxValue }, 0));
        Assert.ThrowsAny<ArgumentException>(() => DenseTensor<float>.OfShape(int.MaxValue, int.MaxValue, int.MaxValue));
    }

    [Fact]
    public void NumericPayloadsKeepTheirFullDomain()
    {
        var values = new[] { long.MinValue, long.MaxValue };
        Assert.Equal(values, DenseTensor<long>.OfValues(values).ToArray());
        Assert.Equal(new[] { double.NegativeInfinity, double.PositiveInfinity }, DenseTensor<double>.OfValues(new[] { double.NegativeInfinity, double.PositiveInfinity }).ToArray());
    }

    [Fact]
    public void ShapeArithmeticCannotWrapBeforeTighterStorageChecks()
    {
        var empty = DenseTensor<float>.OfShape(0, int.MaxValue);
        Assert.ThrowsAny<ArgumentException>(() => Tensor<float>.Concat(new Tensor<float>[] { empty, empty, empty }, 1));
        var source = DenseTensor<float>.OfShape(65536, 65536, 0);
        var shape = DenseTensor<long>.OfValues(new[] { 65536L, -1L, 0L });
        // The existing Int32 reshape-volume limit is tighter: reject rather
        // than wrapping the nonzero volume and inferring a spurious zero axis.
        Assert.ThrowsAny<ArgumentException>(() => Tensor<float>.Reshape(source, shape, true));
    }
}
