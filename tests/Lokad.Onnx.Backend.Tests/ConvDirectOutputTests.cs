using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx.Backend.Tests;

public class ConvDirectOutputTests
{
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
    public void RefusalDoesNotRentOrWrite()
    {
        var output = Enumerable.Repeat(123f, 16).ToArray();
        var accountant = new ScratchAccountant();
        foreach (var option in new[] { TensorExecutionOptions.Scalar, TensorExecutionOptions.Simd })
            Assert.False(Tensor<float>.TryConvDirectOutput(default, default, default, output, default, false,
                32, 64, 2, 2, option with { ScratchReporter = accountant }));
        foreach (var shape in new[] { (0, 64, 32), (30, 64, 32), (33, 64, 32), (96, 64, 32),
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
