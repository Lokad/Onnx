namespace Lokad.Onnx.Backend.Tests;

using System.Runtime.Intrinsics.X86;

public class PackedAvx512RowTests
{
    // Keep this test active on AVX2 machines: the candidate must refuse safely,
    // without dereferencing pointers, before any AVX-512 instruction executes.
    [SkippableFact]
    public unsafe void UnsupportedHardwareDeclinesWithoutTouchingMemory()
    {
        Skip.If(Avx512F.IsSupported && Fma.IsSupported, "This host supports the candidate.");
        Assert.False(MathOps.TryPackedAvx512Rows(12, 384, 384, null, null, null));
    }

    [Fact]
    public unsafe void SmallRowsAndColumnTailsStayOnExistingRoutes()
    {
        foreach (int m in new[] { 0, 1, 2, 3, 7 })
            Assert.False(MathOps.TryPackedAvx512Rows(m, 384, 384, null, null, null));
        foreach (int k in new[] { 0, 1, 7, 8, 15, 16, 31, 33, 63, 95 })
            Assert.False(MathOps.TryPackedAvx512Rows(12, 384, k, null, null, null));
        Assert.False(MathOps.TryPackedAvx512Rows(12, 0, 384, null, null, null));
    }

    [SkippableFact]
    public unsafe void RowGroupsMatchIndependentFmaOracleAndPreserveGuards()
    {
        Skip.If(!Avx512F.IsSupported || !Fma.IsSupported, "AVX-512F and FMA required.");
        var random = new Random(20260918);
        foreach (int m in new[] { 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 64, 128, 512 })
        foreach (var (n, k) in new[] { (1, 32), (3, 64), (31, 96), (65, 384) })
        {
            const int offset = 3;
            const float guard = -12345.5f;
            var a = Enumerable.Repeat(guard, offset + m * n + offset).ToArray();
            var b = new float[n * k];
            var packed = Enumerable.Repeat(guard, offset + n * k + offset).ToArray();
            var output = Enumerable.Repeat(guard, offset + m * k + offset).ToArray();
            for (int i = 0; i < m * n; i++) a[offset + i] = random.NextSingle() * 2 - 1;
            for (int i = 0; i < b.Length; i++) b[i] = random.NextSingle() * 2 - 1;
            // Nonzero destinations exercise accumulation, not just fresh zeros.
            for (int i = 0; i < m * k; i++) output[offset + i] = random.NextSingle() * 2 - 1;
            var expected = (float[])output.Clone();
            var originalA = (float[])a.Clone();
            var originalB = (float[])b.Clone();
            for (int row = 0; row < m; row++)
            for (int col = 0; col < k; col++)
            {
                float acc = expected[offset + row * k + col];
                for (int reduction = 0; reduction < n; reduction++)
                    acc = MathF.FusedMultiplyAdd(b[reduction * k + col], a[offset + row * n + reduction], acc);
                expected[offset + row * k + col] = acc;
            }
            fixed (float* ap = a, bp = b, pp = packed, cp = output)
            {
                MathOps.PackPanelsB(n, k, bp, pp + offset);
                var packedBefore = (float[])packed.Clone();
                Assert.True(MathOps.TryPackedAvx512Rows(m, n, k, ap + offset, pp + offset, cp + offset));
                Assert.Equal(packedBefore, packed);
            }
            Assert.Equal(originalA, a);
            Assert.Equal(originalB, b);
            for (int i = 0; i < output.Length; i++)
                Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(output[i]),
                    $"{m}x{n}x{k} index {i}: expected {expected[i]:R}, actual {output[i]:R}");
            Assert.All(packed.Take(offset).Concat(packed.TakeLast(offset)), x => Assert.Equal(guard, x));
        }
    }
}
