using System;
using System.Linq;

namespace Lokad.Onnx.Backend.Tests;

public class SpatialCopyTests
{
    [Theory]
    [InlineData(1, 1, 1, 1, 2)]
    [InlineData(1, 1, 65, 65, 3)]
    [InlineData(2, 3, 4, 7, 5)]
    [InlineData(3, 2, 35, 40, 1)]
    public unsafe void RangeCopiesMatchIndependentCoordinatesAndPreservePayloadBits(
        int stride, int dilation, int padY, int padX, int width)
    {
        const int channels = 2, height = 4, kh = 3, kw = 2;
        int outH = (height + 2 * padY - ((kh - 1) * dilation + 1)) / stride + 1;
        int outW = (width + 2 * padX - ((kw - 1) * dilation + 1)) / stride + 1;
        int total = outH * outW;
        int[] payload = { 0, int.MinValue, 0x7fc12345, 0x7f800000, unchecked((int)0xff800000), 1, 0x3f800000 };
        var input = Enumerable.Range(0, channels * height * width).Select(i => BitConverter.Int32BitsToSingle(payload[i % payload.Length])).ToArray();
        var before = input.Select(BitConverter.SingleToInt32Bits).ToArray();
        foreach (int chunk in new[] { 1, 7, 32, 53, total })
        for (int start = 0; start < total; start += chunk)
        {
            int count = Math.Min(chunk, total - start), size = channels * kh * kw * count;
            var output = Enumerable.Repeat(12345f, size + 6).ToArray();
            fixed (float* x = input)
            fixed (float* y = output)
                MathOps.Im2colRange(x, channels, height, width, kh, kw, dilation, dilation,
                    stride, stride, padY, padX, padY, padX, outW, start, count, y + 3);
            for (int c = 0; c < channels; c++)
            for (int ky = 0; ky < kh; ky++)
            for (int kx = 0; kx < kw; kx++)
            for (int i = 0; i < count; i++)
            {
                int position = start + i;
                int sy = position / outW * stride + ky * dilation - padY;
                int sx = position % outW * stride + kx * dilation - padX;
                int expected = (uint)sy < height && (uint)sx < width ? before[(c * height + sy) * width + sx] : 0;
                Assert.Equal(expected, BitConverter.SingleToInt32Bits(output[3 + ((c * kh + ky) * kw + kx) * count + i]));
            }
            Assert.All(output.Take(3).Concat(output.TakeLast(3)), value => Assert.Equal(12345f, value));
        }
        Assert.Equal(before, input.Select(BitConverter.SingleToInt32Bits));
    }

    [Theory]
    [InlineData(1)] [InlineData(7)] [InlineData(31)] [InlineData(32)]
    [InlineData(33)] [InlineData(63)] [InlineData(64)] [InlineData(65)] [InlineData(192)]
    public unsafe void PackedPanelsKeepEveryBitAndGuard(int width)
    {
        const int reduction = 13;
        var source = Enumerable.Range(0, reduction * width).Select(i => BitConverter.Int32BitsToSingle(unchecked((int)(0x7f801234u + (uint)i * 0x1234567u)))).ToArray();
        var before = source.Select(BitConverter.SingleToInt32Bits).ToArray();
        var target = Enumerable.Repeat(12345f, source.Length + 6).ToArray();
        fixed (float* x = source)
        fixed (float* y = target) MathOps.PackPanelsB(reduction, width, x, y + 3);
        int full = width / 32 * 32;
        for (int r = 0; r < reduction; r++)
        for (int c = 0; c < width; c++)
        {
            int offset = c < full ? c / 32 * reduction * 32 + r * 32 + c % 32 : full * reduction + r * (width - full) + c - full;
            Assert.Equal(before[r * width + c], BitConverter.SingleToInt32Bits(target[3 + offset]));
        }
        Assert.Equal(before, source.Select(BitConverter.SingleToInt32Bits));
        Assert.All(target.Take(3).Concat(target.TakeLast(3)), value => Assert.Equal(12345f, value));
    }

    [Fact]
    public void ExpandedScratchIsCheckedBeforeRental()
    {
        var plan = Tensor<float>.PlanConvSpatialScratch(32, 3, 3, 32, 80, 998);
        Assert.Equal((79840, 288, 192, 61440), plan);
        Assert.Throws<OverflowException>(() => Tensor<float>.PlanConvSpatialScratch(int.MaxValue, 2, 2, 1, 1, 1));
        Assert.Throws<OverflowException>(() => Tensor<float>.PlanConvSpatialScratch(1, 1, 1, 1, 65536, 65536));
        Assert.Throws<ArgumentException>(() => Tensor<float>.PlanConvSpatialScratch(int.MaxValue / 16, 1, 1, 1, 1, 64));
        // Each individual term fits Int32; their combined rental does not.
        Assert.Throws<ArgumentException>(() => Tensor<float>.PlanConvSpatialScratch(40000000, 1, 1, 40000000, 1, 64));
        var last = Tensor<float>.PlanConvSpatialScratch(1, 1, 1, 1, 1, int.MaxValue);
        Assert.True(last.blockColumns > 0 && last.blockColumns < last.columns);
        Assert.True(last.scratchElements <= 65536);
    }
}
