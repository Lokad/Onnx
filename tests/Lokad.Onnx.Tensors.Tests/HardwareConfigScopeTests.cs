namespace Lokad.Onnx.Tensors.Tests;

public class HardwareConfigScopeTests
{
    [Fact]
    public void RestoresValuesOnException()
    {
        bool simdBefore = HardwareConfig.UseSimd;
        bool intrBefore = HardwareConfig.UseIntrinsics;
        const int Seed = 20260904;
        try
        {
            using (new HardwareConfigScope(!simdBefore, !intrBefore))
            {
                Assert.Equal(!simdBefore, HardwareConfig.UseSimd);
                throw new InvalidOperationException("seeded failure " + Seed);
            }
        }
        catch (InvalidOperationException ex) when (ex.Message.Contains(Seed.ToString()))
        {
        }
        Assert.Equal(simdBefore, HardwareConfig.UseSimd);
        Assert.Equal(intrBefore, HardwareConfig.UseIntrinsics);
    }
}
