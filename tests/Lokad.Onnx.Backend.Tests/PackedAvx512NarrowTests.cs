using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx.Backend.Tests;

public class PackedAvx512NarrowTests
{
    static void EqualBits(float[] expected, float[] actual) =>
        Assert.True(MemoryMarshal.AsBytes(expected.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(actual.AsSpan())));

    [SkippableFact]
    public unsafe void UnsupportedHardwareDeclinesBeforeReadingPointers()
    {
        Skip.If(Avx512F.IsSupported && Fma.IsSupported, "Candidate supported on this host");
        Assert.False(MathOps.TryPackedAvx512NarrowRows(30, 384, 384, null, null, null));
    }

    [Fact]
    public unsafe void InvalidGeometryAndRowsWithoutNarrowRemaindersDecline()
    {
        foreach (int m in new[] { -1, 0, 1, 2, 3, 7, 8, 12, 16, 20, 24, 28, 32, 64, 128, 512 })
            Assert.False(MathOps.TryPackedAvx512NarrowRows(m, 384, 384, null, null, null));
        foreach (int k in new[] { -1, 0, 1, 16, 31, 33, 63, 95, 383, 385 })
            Assert.False(MathOps.TryPackedAvx512NarrowRows(30, 384, k, null, null, null));
        Assert.False(MathOps.TryPackedAvx512NarrowRows(30, 0, 384, null, null, null));
    }

    public static IEnumerable<object[]> Rows()
    {
        foreach (int rows in new[] { 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 30, 31, 33, 34, 35, 37 })
            foreach (bool exceptional in new[] { false, true }) yield return new object[] { rows, exceptional };
    }

    [SkippableTheory]
    [MemberData(nameof(Rows))]
    public unsafe void RemaindersPreserveCompleteBitsAccumulationAndOwnership(int m, bool exceptional)
    {
        Skip.If(!Avx512F.IsSupported || !Fma.IsSupported, "AVX-512F and FMA required");
        const int n = 65, k = 96, offset = 5;
        const float guard = -12345.5f;
        var random = new Random(901 + m);
        var a = Enumerable.Repeat(guard, m * n + 2 * offset).ToArray();
        var b = new float[n * k];
        var packed = Enumerable.Repeat(guard, n * k + 2 * offset).ToArray();
        var actual = Enumerable.Repeat(guard, m * k + 2 * offset).ToArray();
        for (int i = 0; i < m * n; i++) a[offset + i] = random.NextSingle() * 2 - 1;
        for (int i = 0; i < b.Length; i++) b[i] = random.NextSingle() * 2 - 1;
        for (int i = 0; i < m * k; i++) actual[offset + i] = random.NextSingle() * 2 - 1;
        if (exceptional)
        {
            // Put exceptions in the final rows too, so the changed narrow leaves execute them.
            a[offset + (m - 1) * n + 31] = BitConverter.Int32BitsToSingle(0x7FA12345);
            a[offset + (m - 2) * n] = -0f;
            a[offset + (m - 2) * n + 1] = float.Epsilon;
            b[7 * k + 17] = float.PositiveInfinity;
            b[^1] = float.NegativeInfinity;
            actual[offset + (m - 1) * k] = -0f;
        }
        var initial = actual.ToArray(); var originalA = a.ToArray(); var originalB = b.ToArray();
        var reference = actual.ToArray();
        fixed (float* ap = a, bp = b, pp = packed, cp = actual, rp = reference)
        {
            MathOps.PackPanelsB(n, k, bp, pp + offset); var before = packed.ToArray();
            for (int repeat = 0; repeat < 2; repeat++)
            {
                Assert.True(MathOps.TryPackedAvx512Rows(m, n, k, ap + offset, pp + offset, rp + offset));
                Assert.True(MathOps.TryPackedAvx512NarrowRows(m, n, k, ap + offset, pp + offset, cp + offset));
                EqualBits(reference, actual); // Includes every guard, signed zero and NaN payload.
            }
            EqualBits(before, packed);
        }
        EqualBits(originalA, a); EqualBits(originalB, b);
        Assert.All(packed.Take(offset).Concat(packed.TakeLast(offset)), v => Assert.Equal(guard, v));
        // Scalar FMA is independent of either vector implementation and checks every finite output.
        for (int row = 0; row < m; row++)
        for (int col = 0; col < k; col++)
        {
            float expected = initial[offset + row * k + col];
            for (int repeat = 0; repeat < 2; repeat++)
                for (int reduction = 0; reduction < n; reduction++)
                    expected = MathF.FusedMultiplyAdd(b[reduction * k + col], a[offset + row * n + reduction], expected);
            float value = actual[offset + row * k + col];
            if (float.IsNaN(expected)) Assert.True(float.IsNaN(value));
            else Assert.Equal(BitConverter.SingleToInt32Bits(expected), BitConverter.SingleToInt32Bits(value));
        }
    }
}
