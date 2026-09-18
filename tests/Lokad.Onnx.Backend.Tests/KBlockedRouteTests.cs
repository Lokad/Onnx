using Lokad.Onnx.Bench;

namespace Lokad.Onnx.Backend.Tests;

public class KBlockedRouteTests
{
    static TensorExecutionOptions KBlocked()
    {
        return TensorExecutionOptions.Auto with { UseKBlockedPanels = true };
    }

    static GemmMatrixResult RunOne(GemmShape s, TensorExecutionOptions o)
    {
        var results = GemmMatrix.Run(new GemmShape[] { s }, o);
        Assert.Single(results);
        return results[0];
    }

    [Fact]
    public void SwitchOff_PreservesLegacyRoute()
    {
        var r = RunOne(new GemmShape("kb-off", 16, 1024, 1024, true, "test"), TensorExecutionOptions.Auto);
        Assert.Equal("prep-grouped", r.Route);
        Assert.True(r.Pass, "maxScaled=" + r.MaxScaled);
    }

    [Fact]
    public void SmallN_PassthroughKeepsLegacyRoute()
    {
        var r = RunOne(new GemmShape("kb-small", 16, 64, 64, true, "test"), KBlocked());
        Assert.Equal("prep-grouped", r.Route);
        Assert.True(r.Pass, "maxScaled=" + r.MaxScaled);
    }

    [Fact]
    public void BlockedRoute_AgreesWithDouble()
    {
        var r = RunOne(new GemmShape("kb-proj", 16, 1024, 1024, true, "test"), KBlocked());
        Assert.Equal("prep-kblocked", r.Route);
        Assert.True(r.Pass, "maxScaled=" + r.MaxScaled);
    }

    [Fact]
    public void ReductionTail_AgreesWithDouble()
    {
        var r = RunOne(new GemmShape("kb-tail", 16, 300, 64, true, "test"), KBlocked());
        Assert.Equal("prep-kblocked", r.Route);
        Assert.True(r.Pass, "maxScaled=" + r.MaxScaled);
    }

    [Fact]
    public void DynamicPath_UntouchedBySwitch()
    {
        var r = RunOne(new GemmShape("kb-dyn", 16, 128, 128, false, "test"), KBlocked());
        Assert.Equal("tiled-2x4", r.Route);
        Assert.True(r.Pass, "maxScaled=" + r.MaxScaled);
    }

}
