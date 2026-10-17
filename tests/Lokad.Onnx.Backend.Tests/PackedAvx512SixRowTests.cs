using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx.Backend.Tests;

public class PackedAvx512SixRowTests
{
    static void EqualBits(float[] expected, float[] actual)
    {
        if (MemoryMarshal.AsBytes(expected.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(actual.AsSpan()))) return;
        int index = 0;
        while (index < expected.Length && BitConverter.SingleToInt32Bits(expected[index]) == BitConverter.SingleToInt32Bits(actual[index])) index++;
        Assert.Fail($"Different bits at {index}: expected {expected[index]:R}, actual {actual[index]:R}");
    }

    [SkippableFact]
    public unsafe void UnsupportedHardwareDeclinesBeforeReadingPointers()
    {
        Skip.If(Avx512F.IsSupported && Fma.IsSupported, "Candidate supported on this host");
        Assert.False(MathOps.TryPackedAvx512SixRows(256, 512, 512, null, null, null));
    }

    [Theory]
    [InlineData(-1, 512, 512)]
    [InlineData(0, 512, 512)]
    [InlineData(8, 512, 512)]
    [InlineData(128, 512, 512)]
    [InlineData(255, 512, 512)]
    [InlineData(257, 512, 512)]
    [InlineData(256, 0, 512)]
    [InlineData(256, 511, 512)]
    [InlineData(256, 512, 0)]
    [InlineData(256, 512, 511)]
    [InlineData(256, 512, 544)]
    [InlineData(256, 512, 513)]
    public unsafe void UntimedShapesDeclineWithoutTouchingPointers(int m, int n, int k) =>
        Assert.False(MathOps.TryPackedAvx512SixRows(m, n, k, null, null, null));

    // Row remainders cover sixteen, zero, and eight rows. Odd reduction widths
    // and non-power-of-two panel counts exercise address arithmetic too.
    [SkippableTheory]
    [InlineData(256, 512, 512, false)]
    [InlineData(264, 512, 512, false)]
    [InlineData(272, 512, 576, false)]
    [InlineData(384, 513, 512, false)]
    [InlineData(512, 512, 640, false)]
    [InlineData(256, 512, 512, true)]
    [InlineData(264, 512, 512, true)]
    [InlineData(272, 513, 576, true)]
    public unsafe void AccumulationPreservesReferenceBitsGuardsAndInputs(int m, int n, int k, bool exceptional)
    {
        Skip.If(!Avx512F.IsSupported || !Fma.IsSupported, "AVX-512F and FMA required");
        const int offset = 5;
        const float guard = -12345.5f;
        var random = new Random(2801 + m);
        var a = Enumerable.Repeat(guard, m * n + 2 * offset).ToArray();
        var b = new float[n * k];
        var packed = Enumerable.Repeat(guard, n * k + 2 * offset).ToArray();
        var actual = Enumerable.Repeat(guard, m * k + 2 * offset).ToArray();
        for (int i = 0; i < m * n; i++) a[offset + i] = random.NextSingle() * 2 - 1;
        for (int i = 0; i < b.Length; i++) b[i] = random.NextSingle() * 2 - 1;
        for (int i = 0; i < m * k; i++) actual[offset + i] = random.NextSingle() * 2 - 1;
        if (exceptional)
        {
            a[offset] = BitConverter.Int32BitsToSingle(0x7FA12345);
            a[offset + 6 * n + 33] = BitConverter.Int32BitsToSingle(unchecked((int)0xFFA76543));
            a[offset + (m - 1) * n + 31] = BitConverter.Int32BitsToSingle(0x7FC01357);
            a[offset + (m - 2) * n] = -0f;
            a[offset + (m - 2) * n + 1] = float.Epsilon;
            b[7 * k + 17] = float.PositiveInfinity;
            b[^1] = float.NegativeInfinity;
            actual[offset + (m - 1) * k] = -0f;
            actual[offset + (m - 3) * k + 63] = BitConverter.Int32BitsToSingle(0x7FC2468A);
        }
        var initial = actual.ToArray();
        var reference = actual.ToArray();
        var originalA = a.ToArray();
        var originalB = b.ToArray();
        fixed (float* ap = a, bp = b, pp = packed, cp = actual, rp = reference)
        {
            MathOps.PackPanelsB(n, k, bp, pp + offset);
            var originalPacked = packed.ToArray();
            for (int repeat = 0; repeat < 2; repeat++)
            {
                Assert.True(MathOps.TryPackedAvx512Rows(m, n, k, ap + offset, pp + offset, rp + offset));
                Assert.True(MathOps.TryPackedAvx512SixRows(m, n, k, ap + offset, pp + offset, cp + offset));
                EqualBits(reference, actual);
            }
            EqualBits(originalPacked, packed);
        }
        EqualBits(originalA, a);
        EqualBits(originalB, b);
        Assert.All(actual.Take(offset).Concat(actual.TakeLast(offset)), value => Assert.Equal(guard, value));
        Assert.All(packed.Take(offset).Concat(packed.TakeLast(offset)), value => Assert.Equal(guard, value));
        if (!exceptional)
        {
            // Every finite coordinate is independently calculated, including
            // the original nonzero C and both complete accumulation passes.
            for (int row = 0; row < m; row++)
            for (int col = 0; col < k; col++)
            {
                float expected = initial[offset + row * k + col];
                for (int repeat = 0; repeat < 2; repeat++)
                    for (int reduction = 0; reduction < n; reduction++)
                        expected = MathF.FusedMultiplyAdd(a[offset + row * n + reduction], b[reduction * k + col], expected);
                int actualBits = BitConverter.SingleToInt32Bits(actual[offset + row * k + col]);
                Assert.Equal(BitConverter.SingleToInt32Bits(expected), actualBits);
            }
        }
    }
}
