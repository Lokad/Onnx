namespace Lokad.Onnx.Tensors.Tests;

using System.Numerics;

public class RowScalarBroadcastTests
{
    public static IEnumerable<object[]> Shapes()
    {
        int width = Vector<float>.Count * 2 + 3;
        yield return new object[] { new[] { 3, width }, new[] { 3, 1 } };
        yield return new object[] { new[] { 2, 3, width }, new[] { 1, 3, 1 } };
        yield return new object[] { new[] { 2, 1, width }, new[] { 1, 3, 1 } };
        yield return new object[] { new[] { 2, 3, width }, new[] { 3, 1 } };
        yield return new object[] { new[] { 1, 3, 2, width }, new[] { 4, 1, 2, 1 } };
        yield return new object[] { new[] { 0, 3, width }, new[] { 3, 1 } };
        yield return new object[] { new[] { 3, 2 }, new[] { 3, 1 } };
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void RowsMatchIndependentCoordinateReferenceInBothOperandOrders(int[] leftShape, int[] rightShape)
    {
        var left = Values(leftShape);
        var right = Values(rightShape);
        Verify(left, right);
        Verify(right, left);
    }

    [Fact]
    public void NoncontiguousViewsAndReversedStorageKeepLogicalValues()
    {
        int width = Vector<float>.Count + 3;
        var source = Values(new[] { 4, width * 2 });
        var sliced = source.Slice(new SliceIndex(0, 4, 2), new SliceIndex(0, width * 2, 2));
        var reversed = new DenseTensor<float>(new[] { 2, width }, true);
        for (int row = 0; row < 2; row++)
            for (int col = 0; col < width; col++) reversed[row, col] = (row * width + col + 1) * .25f;
        var scalars = Values(new[] { 2, 1 });
        Verify(sliced, scalars);
        Verify(scalars, sliced);
        Verify(reversed, scalars);
        Verify(scalars, reversed);
    }

    [Fact]
    public void SignedZeroInfinityNaNAndSubnormalsFollowScalarArithmetic()
    {
        float[] special = { 0f, -0f, float.PositiveInfinity, float.NegativeInfinity,
            float.NaN, float.Epsilon, -float.Epsilon, 1f, -2f };
        int width = Vector<float>.Count * 2 + 3;
        var values = new DenseTensor<float>(Enumerable.Range(0, 4 * width).Select(i => special[i % special.Length]).ToArray(), new[] { 4, width });
        var scalars = new DenseTensor<float>(new[] { 0f, -0f, float.PositiveInfinity, -2f }, new[] { 4, 1 });
        Verify(values, scalars);
        Verify(scalars, values);
    }

    [Fact]
    public void DestinationWindowAndInPlaceFallbackPreserveInputsAndCanaries()
    {
        int width = Vector<float>.Count + 3;
        var left = Values(new[] { 3, width });
        var right = Values(new[] { 3, 1 });
        float[] original = left.ToArray();
        var buffer = Enumerable.Repeat(-999f, original.Length + 4).ToArray();
        var destination = new DenseTensor<float>(buffer.AsMemory(2, original.Length), left.Dimensions);
        left.BroadcastApply<MultiplyBroadcast<float>>(right, destination, TensorExecutionOptions.Simd);
        for (int i = 0; i < original.Length; i++) Assert.Equal(original[i] * right.GetValue(i / width), buffer[i + 2]);
        Assert.Equal(new[] { -999f, -999f, -999f, -999f }, new[] { buffer[0], buffer[1], buffer[^2], buffer[^1] });
        Assert.Equal(original, left.ToArray());
        left.BroadcastApply<MultiplyBroadcast<float>>(right, left, TensorExecutionOptions.Simd);
        Assert.Equal(destination.ToArray(), left.ToArray());
    }

    [Fact]
    public void DoubleAndIntegerRowsUseTheSameBroadcastContract()
    {
        int width = Math.Max(Vector<double>.Count, Vector<int>.Count) * 2 + 1;
        var integers = new DenseTensor<int>(Enumerable.Range(1, 3 * width).ToArray(), new[] { 3, width });
        var factors = new DenseTensor<int>(new[] { 2, -3, 4 }, new[] { 3, 1 });
        var product = integers.BroadcastApply<MultiplyBroadcast<int>>(factors, TensorExecutionOptions.Simd);
        var doubles = new DenseTensor<double>(Enumerable.Range(1, 3 * width).Select(i => i * .25).ToArray(), new[] { 3, width });
        var divisors = new DenseTensor<double>(new[] { 2.0, -3.0, 4.0 }, new[] { 3, 1 });
        var quotient = divisors.BroadcastApply<DivideBroadcast<double>>(doubles, TensorExecutionOptions.Simd);
        for (int i = 0; i < 3 * width; i++)
        {
            Assert.Equal(integers.GetValue(i) * factors.GetValue(i / width), product.GetValue(i));
            Assert.Equal(divisors.GetValue(i / width) / doubles.GetValue(i), quotient.GetValue(i));
        }
    }

    static DenseTensor<float> Values(int[] shape) => new(
        Enumerable.Range(0, shape.Aggregate(1, (a, b) => a * b)).Select(i => (i % 29 + 1) * .125f).ToArray(), shape);

    static void Verify(Tensor<float> left, Tensor<float> right)
    {
        VerifyOperation<MultiplyBroadcast<float>>(left, right, (a, b) => a * b);
        VerifyOperation<AddBroadcast<float>>(left, right, (a, b) => a + b);
        VerifyOperation<SubtractBroadcast<float>>(left, right, (a, b) => a - b);
        VerifyOperation<DivideBroadcast<float>>(left, right, (a, b) => a / b);
    }

    static void VerifyOperation<TOp>(Tensor<float> left, Tensor<float> right, Func<float, float, float> op)
        where TOp : struct, IBroadcastOperator<float>
    {
        int rank = Math.Max(left.Rank, right.Rank);
        var shape = Enumerable.Range(0, rank).Select(d =>
        {
            int x = d < rank - left.Rank ? 1 : left.Dimensions[d - (rank - left.Rank)];
            int y = d < rank - right.Rank ? 1 : right.Dimensions[d - (rank - right.Rank)];
            return x == 1 ? y : x;
        }).ToArray();
        var expected = new float[shape.Aggregate(1, (a, b) => a * b)];
        var coordinates = new int[rank];
        float Read(Tensor<float> tensor)
        {
            var at = Enumerable.Range(0, tensor.Rank).Select(d => tensor.Dimensions[d] == 1 ? 0 : coordinates[d + rank - tensor.Rank]).ToArray();
            return tensor[at];
        }
        for (int i = 0; i < expected.Length; i++)
        {
            int remainder = i;
            for (int d = rank - 1; d >= 0; d--) { coordinates[d] = remainder % shape[d]; remainder /= shape[d]; }
            expected[i] = op(Read(left), Read(right));
        }
        float[] beforeLeft = left.ToArray(), beforeRight = right.ToArray();
        foreach (var options in new[] { TensorExecutionOptions.Scalar, TensorExecutionOptions.Simd, TensorExecutionOptions.Auto })
        {
            var actual = left.BroadcastApply<TOp>(right, options);
            Assert.Equal(shape, actual.Dimensions.ToArray());
            for (int i = 0; i < expected.Length; i++)
            {
                if (float.IsNaN(expected[i])) Assert.True(float.IsNaN(actual.GetValue(i)));
                else Assert.Equal(BitConverter.SingleToInt32Bits(expected[i]), BitConverter.SingleToInt32Bits(actual.GetValue(i)));
            }
        }
        Assert.Equal(beforeLeft, left.ToArray());
        Assert.Equal(beforeRight, right.ToArray());
    }
}
