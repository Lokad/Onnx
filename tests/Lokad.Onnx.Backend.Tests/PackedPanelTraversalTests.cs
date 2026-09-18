using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx.Backend.Tests;

public class PackedPanelTraversalTests
{
    static void EqualBits(float[] expected, float[] actual) =>
        Assert.True(MemoryMarshal.AsBytes(expected.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(actual.AsSpan())));

    [SkippableFact]
    public unsafe void UnsupportedHardwareDeclinesBeforeReadingPointers()
    {
        Skip.If(Avx512F.IsSupported && Fma.IsSupported, "Candidate supported on this host");
        Assert.False(MathOps.TryPackedAvx512Panels(30, 1536, 384, null, null, null));
    }

    [Fact]
    public unsafe void SingleSweepsSmallWeightsAndColumnTailsDecline()
    {
        foreach (int m in new[] { 0, 1, 2, 3, 7, 8, 9, 12, 16, 24, 36, 48 })
            Assert.False(MathOps.TryPackedAvx512Panels(m, 1536, 384, null, null, null));
        foreach (int k in new[] { 0, 1, 7, 8, 15, 16, 31, 33, 63, 95, 383, 385 })
            Assert.False(MathOps.TryPackedAvx512Panels(30, 1536, k, null, null, null));
        foreach (var (n, k) in new[] { (0, 384), (384, 384), (8191, 32), (8192, 32) })
            Assert.False(MathOps.TryPackedAvx512Panels(30, n, k, null, null, null));
    }

    [SkippableFact]
    public unsafe void NarrowPanelLeavesMatchIndependentFmaAndPreserveGuards()
    {
        Skip.If(!Fma.IsSupported, "FMA required");
        var random = new Random(20260919);
        foreach (var (group, rows) in new[] { (2, 2), (2, 4), (2, 6), (3, 3), (3, 6), (3, 9) })
        foreach (int n in new[] { 1, 3, 31, 65 })
        {
            const int offset = 3, k = 96;
            var a = Enumerable.Repeat(-12345.5f, rows * n + 2 * offset).ToArray();
            var b = new float[n * k];
            var packed = Enumerable.Repeat(-12345.5f, n * k + 2 * offset).ToArray();
            var output = Enumerable.Repeat(-12345.5f, rows * k + 2 * offset).ToArray();
            for (int i = 0; i < rows * n; i++) a[offset + i] = random.NextSingle() * 2 - 1;
            for (int i = 0; i < b.Length; i++) b[i] = random.NextSingle() * 2 - 1;
            for (int i = 0; i < rows * k; i++) output[offset + i] = random.NextSingle() * 2 - 1;
            var expected = (float[])output.Clone();
            var originalA = (float[])a.Clone();
            for (int row = 0; row < rows; row++)
            for (int col = 0; col < k; col++)
                for (int reduction = 0; reduction < n; reduction++)
                    expected[offset + row * k + col] = MathF.FusedMultiplyAdd(b[reduction * k + col], a[offset + row * n + reduction], expected[offset + row * k + col]);
            fixed (float* ap = a, bp = b, pp = packed, cp = output)
            {
                MathOps.PackPanelsB(n, k, bp, pp + offset);
                var before = (float[])packed.Clone();
                for (int col = 0; col < k; col += 32)
                    if (group == 2) MathOps.PackedPanel2(rows, n, ap + offset, pp + offset + col * n, cp + offset, k, col);
                    else MathOps.PackedPanel3(rows, n, ap + offset, pp + offset + col * n, cp + offset, k, col);
                EqualBits(before, packed);
            }
            EqualBits(expected, output);
            EqualBits(originalA, a);
        }
    }

    public static IEnumerable<object[]> MixedRows()
    {
        foreach (int rows in new[] { 10, 11, 13, 14, 15, 17, 23, 25, 26, 27, 28, 29, 30, 31, 32, 64, 128, 512 })
            foreach (bool exceptional in new[] { false, true }) yield return new object[] { rows, exceptional };
    }

    [SkippableTheory]
    [MemberData(nameof(MixedRows))]
    public unsafe void ComposerMatchesOriginalBitsAndIndependentCoordinates(int m, bool exceptional)
    {
        Skip.If(!Avx512F.IsSupported || !Fma.IsSupported, "AVX-512F and FMA required");
        const int n = 385, k = 704, offset = 3;
        var random = new Random(m);
        var a = Enumerable.Repeat(-12345.5f, m * n + 2 * offset).ToArray();
        var b = new float[n * k];
        var packed = Enumerable.Repeat(-12345.5f, n * k + 2 * offset).ToArray();
        var output = Enumerable.Repeat(-12345.5f, m * k + 2 * offset).ToArray();
        for (int i = 0; i < m * n; i++) a[offset + i] = random.NextSingle() * 2 - 1;
        for (int i = 0; i < b.Length; i++) b[i] = random.NextSingle() * 2 - 1;
        for (int i = 0; i < m * k; i++) output[offset + i] = random.NextSingle() * 2 - 1;
        if (exceptional)
        {
            a[offset] = -0f; a[offset + n + 31] = BitConverter.Int32BitsToSingle(0x7FA12345);
            a[offset + 2 * n + 1] = float.Epsilon;
            b[7 * k + 17] = float.PositiveInfinity; b[^1] = float.NegativeInfinity;
            output[offset + k] = -0f;
        }
        var originalA = (float[])a.Clone();
        var originalB = (float[])b.Clone();
        var initial = (float[])output.Clone();
        var reference = (float[])output.Clone();
        fixed (float* ap = a, bp = b, pp = packed, cp = output, rp = reference)
        {
            MathOps.PackPanelsB(n, k, bp, pp + offset);
            var before = (float[])packed.Clone();
            Assert.True(MathOps.TryPackedAvx512Rows(m, n, k, ap + offset, pp + offset, rp + offset));
            Assert.True(MathOps.TryPackedAvx512Panels(m, n, k, ap + offset, pp + offset, cp + offset));
            EqualBits(before, packed);
        }
        EqualBits(reference, output); // Includes every destination guard and NaN payload.
        EqualBits(originalA, a); EqualBits(originalB, b);
        foreach (int col in new[] { 0, 17, 31, 32, 63, k - 33, k - 1 })
        for (int row = 0; row < m; row++)
        {
            float expected = initial[offset + row * k + col];
            for (int reduction = 0; reduction < n; reduction++)
                expected = MathF.FusedMultiplyAdd(b[reduction * k + col], a[offset + row * n + reduction], expected);
            float actual = output[offset + row * k + col];
            if (float.IsNaN(expected)) Assert.True(float.IsNaN(actual));
            else Assert.Equal(BitConverter.SingleToInt32Bits(expected), BitConverter.SingleToInt32Bits(actual));
        }
    }
}
