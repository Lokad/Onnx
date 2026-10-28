namespace Lokad.Onnx.Tensors.Tests;

public class TransposeMatrixDispatchTests
{
    [Theory]
    [InlineData(1, 1)]
    [InlineData(7, 9)]
    [InlineData(8, 8)]
    [InlineData(9, 17)]
    [InlineData(128, 256)]
    [InlineData(256, 256)]
    [InlineData(0, 8)]
    [InlineData(8, 0)]
    public void MatrixSwapPreservesEveryBitAndWindowGuards(int rows, int columns)
    {
        const int guard = 11;
        int count = rows * columns;
        int[] special = { 0, int.MinValue, 0x7f800000, unchecked((int)0xff800000),
            0x7fc00001, 0x7fa00002, unchecked((int)0xffc54321), 1 };
        foreach (bool reversed in new[] { false, true })
        {
            var storage = new float[count + 2 * guard];
            var source = new DenseTensor<float>(storage.AsMemory(guard, count), new[] { rows, columns }, reversed);
            var expected = new int[count];
            for (int row = 0; row < rows; row++)
            for (int col = 0; col < columns; col++)
            {
                int index = row * columns + col;
                int bits = index < special.Length ? special[index] : unchecked(index * 1664525 + 1013904223);
                source[row, col] = BitConverter.Int32BitsToSingle(bits);
                expected[col * rows + row] = bits;
            }
            var before = storage.Select(BitConverter.SingleToInt32Bits).ToArray();
            var backing = Enumerable.Repeat(-123f, count + 2 * guard).ToArray();
            var destination = new DenseTensor<float>(backing.AsMemory(guard, count), new[] { columns, rows });
            Assert.Same(destination, Tensor<float>.Transpose(source, destination, new[] { 1, 0 }));
            Assert.Equal(expected, destination.ToArray().Select(BitConverter.SingleToInt32Bits));
            var allocated = Tensor<float>.Transpose(source, null);
            Assert.Equal(expected, allocated.ToArray().Select(BitConverter.SingleToInt32Bits));
            Assert.Equal(expected, destination.ToArray().Select(BitConverter.SingleToInt32Bits));
            Assert.Equal(before, storage.Select(BitConverter.SingleToInt32Bits));
            Assert.All(backing.Take(guard).Concat(backing.Skip(guard + count)), value => Assert.Equal(-123f, value));
        }
    }
}
