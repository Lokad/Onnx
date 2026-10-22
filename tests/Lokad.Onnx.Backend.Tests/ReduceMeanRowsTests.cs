namespace Lokad.Onnx.Backend.Tests;

public class ReduceMeanRowsTests
{
    static int[] Bits(Tensor<float> tensor) => tensor.ToArray().Select(BitConverter.SingleToInt32Bits).ToArray();

    static Tensor<float> Input(int kind, int rows, int width)
    {
        var storage = kind == 1
            ? new DenseTensor<float>(new[] { rows * 2, width * 2 })
            : new DenseTensor<float>(new[] { rows, width }, kind == 2);
        Tensor<float> result = kind == 1
            ? storage.Slice(new SliceIndex(0, rows * 2, 2), new SliceIndex(0, width * 2, 2)) : storage;
        // Cancellation makes reassociation observable, even for a tiny row.
        float[] pattern = { 1e10f, 1f, -1e10f, .125f, -.25f, float.Epsilon, -0f };
        for (int row = 0; row < rows; row++)
        for (int column = 0; column < width; column++)
            result[row, column] = pattern[(row + column) % pattern.Length];
        return result;
    }

    [Theory]
    [InlineData(0, 0)]
    [InlineData(0, 1)]
    [InlineData(1, 0)]
    [InlineData(1, 1)]
    [InlineData(2, 0)]
    [InlineData(2, 1)]
    public void EveryRowBoundaryAndLayoutMatchesCanonicalScalarBits(int kind, int axis)
    {
        foreach (int rows in new[] { 0, 1, 3, 4, 5, 7, 8, 9, 32 })
        foreach (int width in new[] { 0, 1, 31, 32, 33, 576 })
        {
            var data = Input(kind, rows, width);
            var before = Bits(data);
            var axes = DenseTensor<int>.OfValues(new[] { axis == 1 ? -1 : 0 });
            var canonical = new DenseTensor<float>(data.ToArray(), data.Dimensions);
            foreach (bool keep in new[] { false, true })
            {
                // Native arithmetic is checked independently by FloatMeanNativeTests.
                // Here pin mode/layout equivalence and output ownership across tails.
                var expected = Bits(Tensor<float>.ReduceMean(canonical, axes, keep, false, TensorExecutionOptions.Scalar));
                var scalar = Tensor<float>.ReduceMean(data, axes, keep, false, TensorExecutionOptions.Scalar);
                var actual = Tensor<float>.ReduceMean(data, axes, keep, false, TensorExecutionOptions.Simd);
                Assert.Equal(expected, Bits(scalar));
                Assert.Equal(expected, Bits(actual));
                Assert.Equal(scalar.Dimensions.ToArray(), actual.Dimensions.ToArray());
                Assert.Equal(before, Bits(data));
                for (int i = 0; i < actual.Length; i++) actual.SetValue(i, 0f);
                Assert.Equal(expected, Bits(scalar));
                Assert.Equal(before, Bits(data));
            }
        }
    }

    [Fact]
    public void IndependentRowsPreserveNonfiniteAndSignedZeroSemantics()
    {
        var data = new DenseTensor<float>(new[] { 9, 33 });
        for (int row = 0; row < 9; row++)
        for (int column = 0; column < 33; column++) data[row, column] = -0f;
        data[0, 0] = float.PositiveInfinity;
        data[1, 0] = float.NegativeInfinity;
        data[2, 0] = float.PositiveInfinity; data[2, 32] = float.NegativeInfinity;
        data[3, 1] = BitConverter.Int32BitsToSingle(0x7fc01234);
        data[4, 32] = float.Epsilon;
        var axes = DenseTensor<int>.OfValues(new[] { 1 });
        var expected = Tensor<float>.ReduceMean(data, axes, false, false, TensorExecutionOptions.Scalar).ToArray();
        var actual = Tensor<float>.ReduceMean(data, axes, false, false, TensorExecutionOptions.Simd).ToArray();
        for (int i = 0; i < expected.Length; i++)
        {
            if (float.IsNaN(expected[i])) Assert.True(float.IsNaN(actual[i]));
            else Assert.Equal(BitConverter.SingleToInt32Bits(expected[i]), BitConverter.SingleToInt32Bits(actual[i]));
        }
    }
}
