using System;

namespace Lokad.Onnx.Tensors.Tests;

public class Float16Tests
{
    [Fact]
    public void Float16_Roundtrip_And_Compare()
    {
        var values = new float[] { -2.5f, -1f, 0f, 0.5f, 1f, 2.25f, 10f };
        foreach (var value in values)
        {
            var f16 = (Half)value;
            var back = (float)f16;
            Assert.True(Math.Abs(back - value) < 0.01f);
        }

        var zero = Half.Zero;
        var one = Half.One;
        Assert.True(zero < one);
        Assert.True(one > zero);
        Assert.True(one == Half.One);
    }

    [Fact]
    public void BFloat16_Roundtrip_And_Compare()
    {
        var values = new float[] { -3f, -1f, 0f, 0.25f, 1f, 3.5f, 12f };
        foreach (var value in values)
        {
            var bf = (BFloat16)value;
            var back = (float)bf;
            Assert.True(Math.Abs(back - value) < 0.01f);
        }

        var zero = BFloat16.Zero;
        var one = BFloat16.One;
        Assert.True(zero < one);
        Assert.True(one > zero);
        Assert.True(one == BFloat16.One);
    }

    [Fact]
    public void Float16_ExactBits_EdgeCases()
    {
        Assert.Equal((ushort)0x3C00, BitConverter.HalfToUInt16Bits((Half)1f));
        Assert.Equal((ushort)0xBC00, BitConverter.HalfToUInt16Bits((Half)(-1f)));
        Assert.Equal((ushort)0x3800, BitConverter.HalfToUInt16Bits((Half)0.5f));
        Assert.Equal((ushort)0x0000, BitConverter.HalfToUInt16Bits((Half)0f));
        Assert.Equal((ushort)0x8000, BitConverter.HalfToUInt16Bits((Half)(-0f)));
        Assert.Equal((ushort)0x7C00, BitConverter.HalfToUInt16Bits((Half)float.PositiveInfinity));
        Assert.Equal((ushort)0xFC00, BitConverter.HalfToUInt16Bits((Half)float.NegativeInfinity));
        Assert.Equal((ushort)0x7BFF, BitConverter.HalfToUInt16Bits((Half)65504f));
        Assert.Equal((ushort)0x0400, BitConverter.HalfToUInt16Bits((Half)6.103515625E-05f));
        Assert.Equal((ushort)0x0001, BitConverter.HalfToUInt16Bits((Half)5.96046448E-08f));
        Assert.Equal((ushort)0x2E66, BitConverter.HalfToUInt16Bits((Half)0.1f));
        Assert.Equal((ushort)0x7C00, BitConverter.HalfToUInt16Bits((Half)100000f));
        Assert.Equal((ushort)0x0000, BitConverter.HalfToUInt16Bits((Half)1e-40f));
    }

    [Fact]
    public void Float16_NaN_PreservesSignAndPayload()
    {
        Assert.Equal((ushort)0x7E00, BitConverter.HalfToUInt16Bits((Half)BitConverter.UInt32BitsToSingle(0x7FC00000)));
        Assert.Equal((ushort)0xFE00, BitConverter.HalfToUInt16Bits((Half)BitConverter.UInt32BitsToSingle(0xFFC00000)));
        Assert.Equal((ushort)0x7E00, BitConverter.HalfToUInt16Bits((Half)BitConverter.UInt32BitsToSingle(0x7F800001)));
        Assert.Equal((ushort)0x7F01, BitConverter.HalfToUInt16Bits((Half)BitConverter.UInt32BitsToSingle(0x7FA02000)));
        Assert.Equal(0x7FE02000u, BitConverter.SingleToUInt32Bits((float)BitConverter.UInt16BitsToHalf(0x7F01)));
        Assert.Equal(0xFFC00000u, BitConverter.SingleToUInt32Bits((float)BitConverter.UInt16BitsToHalf(0xFE00)));
    }

    [Fact]
    public void Float16_Subnormal_Roundtrip_PreservesBits()
    {
        foreach (ushort bits in new ushort[] { 0x0001, 0x0002, 0x0200, 0x03FF, 0x83FF })
        {
            Assert.Equal(bits, BitConverter.HalfToUInt16Bits((Half)(float)BitConverter.UInt16BitsToHalf(bits)));
        }
    }

    [Fact]
    public void BFloat16_ExactBits_EdgeCases()
    {
        Assert.Equal(0x3F80, ((BFloat16)1f).value);
        Assert.Equal(0x3F00, ((BFloat16)0.5f).value);
        Assert.Equal(0x8000, ((BFloat16)(-0f)).value);
        Assert.Equal(0x7F80, ((BFloat16)float.PositiveInfinity).value);
        Assert.Equal(0xFFC1, ((BFloat16)float.NaN).value);
        Assert.True(BFloat16.IsNaN((BFloat16)float.NaN));
        Assert.Equal(0x4060, ((BFloat16)3.5f).value);
    }

    [Fact]
    public void BFloat16_Roundtrip_PreservesAllNonNaNBits()
    {
        for (int bits = 0; bits < 65536; bits++)
        {
            ushort b = (ushort)bits;
            var back = (BFloat16)(float)new BFloat16(b);
            if ((b & 0x7F80) == 0x7F80 && (b & 0x007F) != 0)
                Assert.Equal((ushort)0xFFC1, back.value);
            else
                Assert.Equal(b, back.value);
        }
    }

    [Fact]
    public void BFloat16_Comparisons_FollowIeeeOrder()
    {
        var neg = (BFloat16)(-2f);
        var negOne = (BFloat16)(-1f);
        var negZero = (BFloat16)(-0f);
        var posZero = (BFloat16)0f;
        var posOne = (BFloat16)1f;
        var pos = (BFloat16)2f;
        var nan = (BFloat16)float.NaN;
        Assert.True(neg < negOne);
        Assert.True(negOne > neg);
        Assert.True(neg <= negOne);
        Assert.True(negOne >= neg);
        Assert.False(pos < posOne);
        Assert.True(posOne == (BFloat16)1f);
        Assert.False(negZero < posZero);
        Assert.False(negZero > posZero);
        Assert.True(negZero <= posZero);
        Assert.True(negZero >= posZero);
        Assert.False(negZero == posZero);
        Assert.True(negZero != posZero);
        Assert.False(nan < posOne);
        Assert.False(nan > posOne);
        Assert.False(nan <= posOne);
        Assert.False(nan >= posOne);
        Assert.False(nan == nan);
        Assert.True(nan != nan);
        var sub = new BFloat16(0x0001);
        var minNormal = new BFloat16(0x0080);
        Assert.True(sub < minNormal);
        Assert.True(minNormal > sub);
        Assert.True(sub <= minNormal);
        Assert.True(posZero.Equals(negZero));
        Assert.True(nan.Equals(nan));
    }
}
