using Lokad.Onnx.Bench;

namespace Lokad.Onnx.Backend.Tests;

public class ConvLayerTests
{
    [Fact]
    public void TinyLayers_AgreeAndReport()
    {
        var layers = new ConvLayer[]
        {
            new ConvLayer("t-blocked", 1, 16, 6, 6, 16, 3, 3, 1, 1, 0, false, false, true, "test"),
            new ConvLayer("t-legacy", 1, 8, 6, 6, 8, 3, 3, 1, 1, 0, false, false, false, "test"),
        };
        var results = ConvLayerMatrix.Run(layers);
        Assert.Equal(2, results.Count);
        foreach (var r in results) Assert.True(r.Pass, r.Name + " legacy=" + r.LegacyRoute + " blocked=" + r.BlockedRoute + " maxScaled=" + r.MaxScaled);
        Assert.Contains(results[0].BlockedRoute, "conv-blocked");
    }

}
