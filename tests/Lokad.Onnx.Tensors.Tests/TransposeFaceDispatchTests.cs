namespace Lokad.Onnx.Tensors.Tests;

/// <summary>Independent coordinate/bit oracle for both attention permutations.
/// Run with VECTOR_TRANSPOSE_FACES off/on and with hardware intrinsics disabled.</summary>
public class TransposeFaceDispatchTests
{
    public static IEnumerable<object[]> Cases()
    {
        int[][] shapes = {
            new[] { 1, 12, 8, 32 }, new[] { 1, 12, 30, 32 }, new[] { 1, 12, 128, 32 },
            new[] { 1, 12, 512, 32 }, new[] { 2, 3, 17, 9 }, new[] { 1, 7, 9, 7 },
            new[] { 1, 1, 1, 1 }, new[] { 1, 20, 7, 64 }, new[] { 0, 12, 8, 32 },
            new[] { 1, 0, 8, 32 }, new[] { 1, 12, 0, 32 }, new[] { 1, 12, 8, 0 }
        };
        foreach (var shape in shapes)
            foreach (bool merge in new[] { false, true })
                foreach (string layout in new[] { "window", "reversed", "slice" })
                    yield return new object[] { shape, merge, layout };
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void PublicTransposePreservesBitsAndGuards(int[] shape, bool merge, string layout)
    {
        const int guard = 11;
        int count = shape.Aggregate(1, (a, b) => a * b);
        int[] perm = merge ? new[] { 0, 2, 3, 1 } : new[] { 0, 1, 3, 2 };
        int[] destShape = perm.Select(i => shape[i]).ToArray();
        Tensor<float> source;
        float[] storage;
        float sentinel = BitConverter.Int32BitsToSingle(unchecked((int)0xffa12345));
        if (layout == "slice")
        {
            int[] fullShape = (int[])shape.Clone(); fullShape[3] += 2;
            storage = Enumerable.Repeat(sentinel, fullShape.Aggregate(1, (a, b) => a * b)).ToArray();
            var full = new DenseTensor<float>(storage, fullShape);
            source = new TensorSlice<float>(full, new[] { SliceIndex.All, SliceIndex.All, SliceIndex.All, new SliceIndex(1, shape[3] + 1) });
        }
        else
        {
            storage = Enumerable.Repeat(sentinel, count + 2 * guard).ToArray();
            source = new DenseTensor<float>(storage.AsMemory(guard, count), shape, layout == "reversed");
        }
        int[] bits = { 0, int.MinValue, 0x7f800000, unchecked((int)0xff800000),
            0x7fc00001, 0x7fa00002, unchecked((int)0xffc54321), 1, unchecked((int)0x80000001) };
        var expected = new int[count];
        int flat = 0;
        for (int b = 0; b < shape[0]; b++)
            for (int h = 0; h < shape[1]; h++)
                for (int s = 0; s < shape[2]; s++)
                    for (int d = 0; d < shape[3]; d++)
                    {
                        int value = flat < bits.Length ? bits[flat] : unchecked(flat * 1664525 + 1013904223);
                        source[b, h, s, d] = BitConverter.Int32BitsToSingle(value);
                        int destinationIndex = merge ? ((b * shape[2] + s) * shape[3] + d) * shape[1] + h
                            : ((b * shape[1] + h) * shape[3] + d) * shape[2] + s;
                        expected[destinationIndex] = value;
                        flat++;
                    }
        var before = storage.Select(BitConverter.SingleToInt32Bits).ToArray();
        var backing = Enumerable.Repeat(sentinel, count + 2 * guard).ToArray();
        var destination = new DenseTensor<float>(backing.AsMemory(guard, count), destShape);
        var permBefore = (int[])perm.Clone();
        Assert.Same(destination, Tensor<float>.Transpose(source, destination, perm));
        Assert.Equal(expected, destination.ToArray().Select(BitConverter.SingleToInt32Bits));
        Assert.Equal(before, storage.Select(BitConverter.SingleToInt32Bits));
        Assert.Equal(permBefore, perm);
        Assert.All(backing.Take(guard).Concat(backing.Skip(guard + count)),
            value => Assert.Equal(BitConverter.SingleToInt32Bits(sentinel), BitConverter.SingleToInt32Bits(value)));
        var allocated = Tensor<float>.Transpose(source, perm);
        Assert.Equal(destShape, allocated.Dimensions.ToArray());
        Assert.Equal(expected, allocated.ToArray().Select(BitConverter.SingleToInt32Bits));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DisjointWindowsWorkButOverlapIsRejected(bool merge)
    {
        int[] shape = { 1, 8, 8, 8 }, perm = merge ? new[] { 0, 2, 3, 1 } : new[] { 0, 1, 3, 2 };
        var storage = Enumerable.Range(0, 1024).Select(i => (float)i).ToArray();
        var source = new DenseTensor<float>(storage.AsMemory(0, 512), shape);
        var destination = new DenseTensor<float>(storage.AsMemory(512, 512), shape);
        var expected = Tensor<float>.Transpose(source, perm).ToArray();
        Tensor<float>.Transpose(source, destination, perm);
        Assert.Equal(expected, destination.ToArray());
        var before = (float[])storage.Clone();
        var overlap = new DenseTensor<float>(storage.AsMemory(128, 512), shape);
        Assert.Throws<ArgumentException>(() => Tensor<float>.Transpose(source, overlap, perm));
        Assert.Equal(before, storage);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void OtherDtypesKeepTheirElements(bool merge)
    {
        int[] shape = { 2, 9, 7, 3 }, perm = merge ? new[] { 0, 2, 3, 1 } : new[] { 0, 1, 3, 2 };
        var source = new DenseTensor<long>(Enumerable.Range(0, 378).Select(i => long.MaxValue - i).ToArray(), shape);
        var result = Tensor<long>.Transpose(source, perm);
        for (int b = 0; b < 2; b++)
            for (int h = 0; h < 9; h++)
                for (int s = 0; s < 7; s++)
                    for (int d = 0; d < 3; d++)
                        Assert.Equal(source[b, h, s, d], merge ? result[b, s, d, h] : result[b, h, d, s]);
    }
}
