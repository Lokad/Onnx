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
            var f16 = (Float16)value;
            var back = (float)f16;
            Assert.True(Math.Abs(back - value) < 0.01f);
        }

        var zero = Float16.Zero;
        var one = Float16.One;
        Assert.True(zero < one);
        Assert.True(one > zero);
        Assert.True(one == Float16.One);
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
        Assert.Equal(0x3C00, ((Float16)1f).value);
        Assert.Equal(0xBC00, ((Float16)(-1f)).value);
        Assert.Equal(0x3800, ((Float16)0.5f).value);
        Assert.Equal(0x0000, ((Float16)0f).value);
        Assert.Equal(0x8000, ((Float16)(-0f)).value);
        Assert.Equal(0x7C00, ((Float16)float.PositiveInfinity).value);
        Assert.Equal(0xFC00, ((Float16)float.NegativeInfinity).value);
        Assert.Equal(0x7BFF, ((Float16)65504f).value);
        Assert.Equal(0x0400, ((Float16)6.103515625E-05f).value);
        Assert.Equal(0x0001, ((Float16)5.96046448E-08f).value);
        Assert.Equal(0x2E66, ((Float16)0.1f).value);
        Assert.Equal(0x7C00, ((Float16)100000f).value);
        Assert.Equal(0x0000, ((Float16)1e-40f).value);
    }

    [Fact]
    public void Float16_NaN_PreservesSignAndPayload()
    {
        Assert.Equal(0x7E00, ((Float16)BitConverter.UInt32BitsToSingle(0x7FC00000)).value);
        Assert.Equal(0xFE00, ((Float16)BitConverter.UInt32BitsToSingle(0xFFC00000)).value);
        Assert.Equal(0x7E00, ((Float16)BitConverter.UInt32BitsToSingle(0x7F800001)).value);
        Assert.Equal(0x7F01, ((Float16)BitConverter.UInt32BitsToSingle(0x7FA02000)).value);
        Assert.Equal(0x7FE02000u, BitConverter.SingleToUInt32Bits((float)new Float16(0x7F01)));
        Assert.Equal(0xFFC00000u, BitConverter.SingleToUInt32Bits((float)new Float16(0xFE00)));
    }

    [Fact]
    public void Float16_Subnormal_Roundtrip_PreservesBits()
    {
        foreach (ushort bits in new ushort[] { 0x0001, 0x0002, 0x0200, 0x03FF, 0x83FF })
        {
            Assert.Equal(bits, ((Float16)(float)new Float16(bits)).value);
        }
    }

    [Fact]
    public void Float16_AgreesWithSystemHalf_OnFiniteValues()
    {
        var values = new float[]
        {
            0f, -0f, 1f, -1f, 0.5f, 0.1f, 2.25f, 10f, -2.5f, 3.14159274f,
            65504f, 100000f, -100000f, 1e10f, 6.103515625E-05f, 5.96046448E-08f,
            6.09756E-05f, 1e-40f, 1.23456789e-10f, float.PositiveInfinity, float.NegativeInfinity,
        };
        foreach (var value in values)
        {
            Assert.Equal(BitConverter.HalfToUInt16Bits((Half)value), ((Float16)value).value);
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
}
