using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx.Backend.Tests;

public class ConvDirectOutputTests
{
    [SkippableFact]
    public unsafe void WideStridedKernelKeepsZeroAndAccumulatingSeeds()
    {
        Skip.If(!Avx512F.IsSupported || !Fma.IsSupported, "AVX-512F and FMA required.");
        const int offset = 3;
        const float guard = -12345.25f;
        foreach (var (rows, reduction, columns) in new[] { (8, 1, 16), (16, 7, 27),
            (24, 31, 32), (40, 65, 59), (64, 64, 128), (96, 65, 95),
            (128, 966, 59), (256, 1542, 32), (256, 256, 127), (1024, 64, 33) })
        foreach (bool zeroSeed in new[] { true, false })
        {
            if (!zeroSeed && columns < 32) continue;
            int stride = columns + 11;
            var a = Enumerable.Range(0, rows * reduction).Select(i => (i % 31 - 15) / 997f).ToArray();
            var b = Enumerable.Range(0, reduction * columns).Select(i => (i % 37 - 18) / 991f).ToArray();
            var packed = new float[b.Length];
            var beforeA = a.ToArray(); var beforeB = b.ToArray();
            var output = Enumerable.Repeat(guard, offset + rows * stride + offset).ToArray();
            for (int row = 0; row < rows; row++)
                for (int col = 0; col < columns; col++)
                    output[offset + row * stride + col] = zeroSeed ? float.NaN : (row % 7 - 3) * .125f;
            var expected = output.ToArray();
            for (int row = 0; row < rows; row++)
                for (int col = 0; col < columns; col++)
                    expected[offset + row * stride + col] = ReferenceProduct(a, b, row, reduction, columns, col,
                        zeroSeed ? 0f : expected[offset + row * stride + col]);
            fixed (float* ap = a, bp = b, pp = packed, cp = output)
            {
                MathOps.PackPanelsB(reduction, columns, bp, pp);
                var oldPacked = packed.ToArray();
                for (int repeat = 0; repeat < 2; repeat++)
                {
                    if (repeat != 0)
                        for (int row = 0; row < rows; row++)
                            for (int col = 0; col < columns; col++)
                                output[offset + row * stride + col] = zeroSeed ? float.NaN : (row % 7 - 3) * .125f;
                    long before = GC.GetAllocatedBytesForCurrentThread();
                    Assert.True(MathOps.TryPackedAvx512Rows(rows, reduction, columns, ap, pp, cp + offset, stride, zeroSeed));
                    long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
                    if (repeat != 0) Assert.Equal(0, allocated);
                    EqualBits(expected, output); EqualBits(oldPacked, packed);
                }
            }
            EqualBits(beforeA, a); EqualBits(beforeB, b);
        }
    }

    [SkippableFact]
    public void WideDirectOutputPreservesLiveInputsSpecialValuesAndUnusedTemporary()
    {
        Skip.If(!Avx512F.IsSupported || !Fma.IsSupported, "AVX-512F and FMA required.");
        foreach (var (rows, reduction, columns) in new[] { (64, 64, 16), (64, 65, 27),
            (96, 64, 59), (128, 966, 59), (160, 896, 128), (256, 1542, 32) })
        foreach (bool hasBias in new[] { false, true })
        {
            float[] specials = reduction < 100 ? new[] { 0f, -0f, float.Epsilon,
                float.MaxValue, float.PositiveInfinity, float.NegativeInfinity, float.NaN } : new[] { .125f };
            foreach (float special in specials)
            {
                int stride = columns + 11;
                var a = Enumerable.Range(0, rows * reduction).Select(i => (i % 7 - 3) * .125f).ToArray();
                var b = Enumerable.Range(0, reduction * columns).Select(i => (i % 11 - 5) * .125f).ToArray();
                var bias = hasBias ? Enumerable.Range(0, rows).Select(i => (i % 7 - 3) * .25f).ToArray() : Array.Empty<float>();
                a[0] = special; a[(rows - 1) * reduction + 1] = special; b[columns - 1] = special;
                for (int pass = 0; pass < 2; pass++)
                {
                    a[reduction] += .25f; b[columns] -= .5f; if (hasBias) bias[1] -= .125f;
                    var oldA = a.ToArray(); var oldB = b.ToArray(); var oldBias = bias.ToArray();
                    var output = Enumerable.Repeat(-12345.25f, rows * stride + 6).ToArray();
                    var expected = output.ToArray();
                    for (int row = 0; row < rows; row++)
                        for (int col = 0; col < columns; col++)
                        {
                            float value = ReferenceProduct(a, b, row, reduction, columns, col, 0f);
                            expected[3 + row * stride + col] = hasBias ? value + bias[row] : value;
                        }
                    var temporary = Enumerable.Repeat(float.NaN, rows * columns).ToArray();
                    var oldTemporary = temporary.ToArray(); var accountant = new ScratchAccountant();
                    Assert.True(Tensor<float>.TryConvDirectOutput(a, b, temporary,
                        output.AsSpan(3, rows * stride), bias, hasBias, rows, reduction, columns, stride,
                        TensorExecutionOptions.Auto with { ScratchReporter = accountant }));
                    for (int i = 0; i < expected.Length; i++)
                        if (float.IsNaN(expected[i])) Assert.True(float.IsNaN(output[i]));
                        else Assert.Equal(BitConverter.SingleToInt32Bits(expected[i]), BitConverter.SingleToInt32Bits(output[i]));
                    EqualBits(oldA, a); EqualBits(oldB, b); EqualBits(oldBias, bias); EqualBits(oldTemporary, temporary);
                    Assert.Equal(columns <= 32 ? 0L : (long)reduction * columns * sizeof(float), accountant.TotalScratchBytes);
                }
            }
        }
    }

    [SkippableFact]
    public unsafe void WideDirectStorageAndModeRefusalPreserveMemory()
    {
        Skip.If(!Avx512F.IsSupported || !Fma.IsSupported, "AVX-512F and FMA required.");
        Assert.False(MathOps.TryPackedAvx512Rows(9, 64, 32, null, null, null, 43, true));
        Assert.False(MathOps.TryPackedAvx512Rows(64, 64, 32, null, null, null, 31, true));
        Assert.False(MathOps.TryPackedAvx512Rows(64, 64, 15, null, null, null, 26, true));
        foreach (string broken in new[] { "weights", "patch", "temporary", "output", "bias", "stride", "overlap" })
        {
            const int rows = 96, reduction = 64, columns = 27;
            var a = new float[rows * reduction]; var b = new float[reduction * columns];
            var temp = new float[rows * columns]; var output = Enumerable.Repeat(19f, rows * 38).ToArray();
            var bias = new float[rows]; int stride = 38;
            switch (broken)
            {
                case "weights": a = new float[a.Length - 1]; break;
                case "patch": b = new float[b.Length - 1]; break;
                case "temporary": temp = new float[temp.Length - 1]; break;
                case "output": output = new float[(rows - 1) * stride + columns - 1]; break;
                case "bias": bias = new float[rows - 1]; break;
                case "stride": stride = 26; break;
                case "overlap": output = a; break;
            }
            var before = new[] { a, b, temp, output, bias }.Select(v => v.ToArray()).ToArray();
            var accountant = new ScratchAccountant();
            Assert.Throws<ArgumentException>(() => Tensor<float>.TryConvDirectOutput(a, b, temp, output, bias,
                true, rows, reduction, columns, stride, TensorExecutionOptions.Auto with { ScratchReporter = accountant }));
            foreach (var pair in before.Zip(new[] { a, b, temp, output, bias })) EqualBits(pair.First, pair.Second);
            Assert.Equal(0, accountant.TotalScratchBytes);
        }
    }

    static float ReferenceProduct(float[] a, float[] b, int row, int reduction, int columns, int col, float seed)
    {
        int fusedEnd = columns / 32 * 32 + columns % 32 / 8 * 8;
        for (int j = 0; j < reduction; j++)
            seed = col < fusedEnd ? MathF.FusedMultiplyAdd(a[row * reduction + j], b[j * columns + col], seed)
                : seed + a[row * reduction + j] * b[j * columns + col];
        return seed;
    }

    static void EqualBits(float[] expected, float[] actual) =>
        Assert.True(MemoryMarshal.AsBytes(expected.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(actual.AsSpan())));

    [Theory]
    [InlineData(32, 64, 2, false)]
    [InlineData(32, 65, 7, true)]
    [InlineData(64, 64, 32, true)]
    [InlineData(64, 64, 33, true)]
    [InlineData(64, 64, 64, false)]
    [InlineData(128, 65, 8, false)]
    [InlineData(256, 1152, 2, true)]
    [InlineData(256, 2304, 2, false)]
    [InlineData(256, 256, 32, true)]
    [InlineData(602, 600, 32, false)]
    [InlineData(602, 600, 45, true)]
    [InlineData(602, 600, 64, false)]
    public void StridedProductPreservesInputsAndGuards(int rows, int reduction, int columns, bool hasBias)
    {
        int stride = columns + 11;
        var w = Enumerable.Range(0, rows * reduction).Select(i => (i % 31 - 15) / 997f).ToArray();
        var p = Enumerable.Range(0, reduction * columns).Select(i => (i % 37 - 18) / 991f).ToArray();
        var bias = hasBias ? Enumerable.Range(0, rows).Select(i => i / 992f).ToArray() : Array.Empty<float>();
        for (int pass = 0; pass < 2; pass++)
        {
            w[0] += .125f; p[0] -= .25f; if (hasBias) bias[0] += .5f;
            var oldW = (float[])w.Clone(); var oldP = (float[])p.Clone(); var oldBias = (float[])bias.Clone();
            var expected = Enumerable.Repeat(-12345.25f, rows * stride + 6).ToArray();
            var actual = (float[])expected.Clone();
            var temp = Enumerable.Repeat(-12345.25f, rows * columns + 6).ToArray();
            var contiguous = new float[rows * columns];
            var accountant = new ScratchAccountant(); var options = TensorExecutionOptions.Auto with { ScratchReporter = accountant };
            bool handled = Tensor<float>.TryConvDirectOutput(w, p, temp.AsSpan(3, rows * columns),
                actual.AsSpan(3, rows * stride), bias, hasBias, rows, reduction, columns, stride, options);
            Assert.Equal(Fma.IsSupported, handled);
            EqualBits(oldW, w); EqualBits(oldP, p); EqualBits(oldBias, bias);
            if (handled)
            {
                Assert.True(Tensor<float>.TryConvPortableRows(w, p, contiguous, rows, reduction, columns, TensorExecutionOptions.Auto));
                for (int row = 0; row < rows; row++) for (int col = 0; col < columns; col++)
                    expected[3 + row * stride + col] = hasBias ? contiguous[row * columns + col] + bias[row] : contiguous[row * columns + col];
                Assert.Equal(columns <= 32 ? 0L : (long)reduction * columns * sizeof(float), accountant.TotalScratchBytes);
            }
            else Assert.Equal(0, accountant.TotalScratchBytes);
            EqualBits(expected, actual);
            Assert.All(temp.Take(3).Concat(temp.TakeLast(3)), value => Assert.Equal(-12345.25f, value));
        }
    }

    [Fact]
    public void WideRowsPreserveExceptionalClassificationAndSignedZeros()
    {
        if (!Fma.IsSupported) return;
        const int rows = 602, reduction = 600, columns = 45, stride = columns + 7;
        foreach (float special in new[] { -0f, float.Epsilon, float.MaxValue,
            float.PositiveInfinity, float.NegativeInfinity, float.NaN })
        {
            var w = Enumerable.Range(0, rows * reduction).Select(i => (i % 7 - 3) * .125f).ToArray();
            var p = Enumerable.Range(0, reduction * columns).Select(i => (i % 11 - 5) * .125f).ToArray();
            w[0] = special; w[reduction * 599 + 17] = special; p[0] = special;
            var expected = new float[rows * columns];
            Assert.True(Tensor<float>.TryConvPortableRows(w, p, expected, rows, reduction, columns, TensorExecutionOptions.Auto));
            var actual = Enumerable.Repeat(-12345.25f, rows * stride).ToArray();
            var temporary = Enumerable.Repeat(float.NaN, rows * columns).ToArray();
            Assert.True(Tensor<float>.TryConvDirectOutput(w, p, temporary, actual, default, false,
                rows, reduction, columns, stride, TensorExecutionOptions.Auto));
            for (int row = 0; row < rows; row++)
            for (int col = 0; col < stride; col++)
            {
                float value = col < columns ? expected[row * columns + col] : -12345.25f;
                if (float.IsNaN(value)) Assert.True(float.IsNaN(actual[row * stride + col]));
                else Assert.Equal(BitConverter.SingleToInt32Bits(value), BitConverter.SingleToInt32Bits(actual[row * stride + col]));
            }
        }
    }

    [Fact]
    public void RefusalDoesNotRentOrWrite()
    {
        var output = Enumerable.Repeat(123f, 16).ToArray();
        var accountant = new ScratchAccountant();
        foreach (var option in new[] { TensorExecutionOptions.Scalar, TensorExecutionOptions.Simd })
            Assert.False(Tensor<float>.TryConvDirectOutput(default, default, default, output, default, false,
                32, 64, 2, 2, option with { ScratchReporter = accountant }));
        foreach (var shape in new[] { (0, 64, 32), (30, 64, 32), (33, 64, 32), (95, 64, 32),
            (32, 63, 32), (32, 64, 0), (32, 1024, 65), (64, int.MaxValue, 2) })
            Assert.False(Tensor<float>.TryConvDirectOutput(default, default, default, output, default, false,
                shape.Item1, shape.Item2, shape.Item3, shape.Item3, TensorExecutionOptions.Auto with { ScratchReporter = accountant }));
        Assert.All(output, x => Assert.Equal(123f, x)); Assert.Equal(0, accountant.TotalScratchBytes);
    }

    [Theory]
    [InlineData("weights")]
    [InlineData("patch")]
    [InlineData("temporary")]
    [InlineData("output")]
    [InlineData("bias")]
    [InlineData("stride")]
    [InlineData("overflow")]
    [InlineData("weights-output")]
    [InlineData("patch-output")]
    [InlineData("temporary-output")]
    [InlineData("bias-output")]
    [InlineData("weights-temporary")]
    [InlineData("patch-temporary")]
    [InlineData("bias-temporary")]
    public void InvalidStorageFailsBeforeMutationOrRent(string broken)
    {
        if (!Fma.IsSupported) return;
        const int rows = 32, reduction = 64, columns = 2;
        var w = new float[rows * reduction]; var p = new float[reduction * columns];
        var t = Enumerable.Repeat(17f, rows * columns).ToArray(); var o = Enumerable.Repeat(19f, rows * 5).ToArray();
        var b = new float[rows]; int stride = 5;
        switch (broken)
        {
            case "weights": w = new float[w.Length - 1]; break;
            case "patch": p = new float[p.Length - 1]; break;
            case "temporary": t = new float[t.Length - 1]; break;
            case "output": o = new float[(rows - 1) * stride + columns - 1]; break;
            case "bias": b = new float[b.Length - 1]; break;
            case "stride": stride = 1; break;
            case "overflow": stride = int.MaxValue; break;
            case "weights-output": o = w; break;
            case "patch-output": o = p = new float[rows * 5]; break;
            case "temporary-output": t = o; break;
            case "bias-output": b = o; break;
            case "weights-temporary": t = w; break;
            case "patch-temporary": t = p; break;
            case "bias-temporary": b = t; break;
        }
        var originals = new[] { w, p, t, o, b }.Select(x => (float[])x.Clone()).ToArray();
        var accountant = new ScratchAccountant(); var options = TensorExecutionOptions.Auto with { ScratchReporter = accountant };
        Assert.Throws<ArgumentException>(() => Tensor<float>.TryConvDirectOutput(w, p, t, o, b, true,
            rows, reduction, columns, stride, options));
        foreach (var pair in originals.Zip(new[] { w, p, t, o, b })) EqualBits(pair.First, pair.Second);
        Assert.Equal(0, accountant.TotalScratchBytes);
    }
}
