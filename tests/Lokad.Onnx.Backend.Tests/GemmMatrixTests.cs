using Lokad.Onnx.Bench;

namespace Lokad.Onnx.Backend.Tests;

public class GemmMatrixTests
{
    [Fact]
    public void TinyShapes_AgreeWithDoubleReference()
    {
        var shapes = new GemmShape[]
        {
            new GemmShape("t-2d-prep", 4, 8, 16, true, "test"),
            new GemmShape("t-odd", 7, 5, 9, false, "test"),
            new GemmShape("t-row", 1, 32, 16, false, "test"),
        };
        var results = GemmMatrix.Run(shapes);
        Assert.Equal(3, results.Count);
        foreach (var r in results) Assert.True(r.Pass, r.Name + " route=" + r.Route + " maxScaled=" + r.MaxScaled);
        Assert.Contains(results, r => r.Route.Length > 0 && r.Route != "none");
    }
}
