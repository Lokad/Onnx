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
}
