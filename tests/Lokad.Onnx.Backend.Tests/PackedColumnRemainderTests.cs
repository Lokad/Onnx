using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx.Backend.Tests;

public class PackedColumnRemainderTests
{
    const int Offset = 7;
    const float Guard = -12345.5f;

    [SkippableFact]
    public void NarrowWidthsAccumulateWithoutAllocationOrOverwrite()
    {
        Skip.If(!Avx2.IsSupported || !Fma.IsSupported, "The raw packed kernel requires AVX2 and FMA.");
        // Covers the complete width domain omitted by the original isolated fixtures.
        for (int columns = 0; columns < 32; columns++) Check(64, 64, columns);
    }

    [SkippableFact]
    public void RowAndPanelBoundariesPreserveAccumulationAndOwnership()
    {
        Skip.If(!Avx2.IsSupported || !Fma.IsSupported, "The raw packed kernel requires AVX2 and FMA.");
        foreach (int rows in new[] { 0, 2, 62, 64, 66, 68, 70, 72 })
            foreach (int columns in new[] { 7, 8, 9, 31, 32, 33, 63, 64, 65 })
                Check(rows, 65, columns);
        foreach (int reduction in new[] { 0, 1, 63, 64 }) Check(70, reduction, 31);
        foreach (int columns in new[] { 222, 224, 225 }) Check(70, 65, columns);
    }

    static float[] Guarded(int length) => Enumerable.Repeat(Guard, length + 2 * Offset).ToArray();

    static void EqualBits(float[] expected, float[] actual) => Assert.True(
        MemoryMarshal.AsBytes(expected.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(actual.AsSpan())));

    static void CheckGuards(float[] values)
    {
        for (int i = 0; i < Offset; i++)
        {
            Assert.Equal(Guard, values[i]);
            Assert.Equal(Guard, values[values.Length - 1 - i]);
        }
    }

    static unsafe void Check(int rows, int reduction, int columns)
    {
        var a = Guarded(rows * reduction);
        var b = Guarded(reduction * columns);
        var packed = Guarded(reduction * columns);
        var output = Guarded(rows * columns);
        for (int i = 0; i < rows * reduction; i++) a[Offset + i] = (i * 13 + 7) % 17 - 8;
        for (int i = 0; i < reduction * columns; i++) b[Offset + i] = (i * 7 + 3) % 19 - 9;
        for (int i = 0; i < rows * columns; i++) output[Offset + i] = i % 7 + 1;
        var beforeA = a.ToArray();
        var beforeB = b.ToArray();
        var initial = output.ToArray();
        var sums = new long[rows * columns];
        // Every input and all intermediate sums are small exact integers. Integer
        // multiplication/addition is independent of packing, FMA and vector width.
        for (int row = 0; row < rows; row++)
            for (int col = 0; col < columns; col++)
                for (int index = 0; index < reduction; index++)
                    sums[row * columns + col] += (long)a[Offset + row * reduction + index]
                        * (long)b[Offset + index * columns + col];

        fixed (float* ap = a, bp = b, pp = packed, cp = output)
        {
            MathOps.PackPanelsB(reduction, columns, bp + Offset, pp + Offset);
            CheckGuards(packed);
            var beforePacked = packed.ToArray();
            for (int repeat = 0; repeat < 3; repeat++)
            {
                long before = GC.GetAllocatedBytesForCurrentThread();
                MathOps.mm_unsafe_vectorized_intrinsics_2x4packed_bump(
                    rows, reduction, columns, ap + Offset, pp + Offset, cp + Offset);
                long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
                // The first call also checks correctness; later calls check steady allocation.
                if (repeat > 0) Assert.Equal(0L, allocated);
                for (int index = 0; index < sums.Length; index++)
                {
                    float expected = (float)((long)initial[Offset + index] + (repeat + 1) * sums[index]);
                    Assert.Equal(BitConverter.SingleToInt32Bits(expected),
                        BitConverter.SingleToInt32Bits(output[Offset + index]));
                }
                CheckGuards(output);
                EqualBits(beforeA, a);
                EqualBits(beforeB, b);
                EqualBits(beforePacked, packed);
            }
        }
    }
}
