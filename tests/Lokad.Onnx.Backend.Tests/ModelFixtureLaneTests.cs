namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Pins the local-model lane policy: missing assets skip with a reason by
/// default and fail only when explicitly requested, while present assets
/// always resolve.
/// </summary>
public class ModelFixtureLaneTests
{
    static void WithLaneVariable(string? value, Action exercise)
    {
        var previous = Environment.GetEnvironmentVariable(ModelFixture.LaneVariable);
        Environment.SetEnvironmentVariable(ModelFixture.LaneVariable, value);
        try
        {
            exercise();
        }
        finally
        {
            Environment.SetEnvironmentVariable(ModelFixture.LaneVariable, previous);
        }
    }

    [Fact]
    public void MissingAsset_WithoutLaneVariable_SkipsWithReason()
    {
        WithLaneVariable(null, () =>
        {
            var ex = Record.Exception(() => ModelFixture.RequireModelOrSkip("missing", "no-such-dir", "no-such-file.onnx"));
            Assert.NotNull(ex);
            Assert.Contains("not present", ex.Message);
        });
    }

    [Fact]
    public void MissingAsset_WithLaneVariable_FailsRequest()
    {
        WithLaneVariable("1", () =>
        {
            var ex = Record.Exception(() => ModelFixture.RequireModelOrSkip("missing", "no-such-dir", "no-such-file.onnx"));
            Assert.NotNull(ex);
            Assert.Contains("requested via", ex.Message);
        });
    }

    [Fact]
    public void PresentAsset_ResolvesRegardlessOfLane()
    {
        var parts = new string[] { "tests", "Lokad.Onnx.Backend.Tests", "models", "mnist-8.onnx" };
        WithLaneVariable(null, () => Assert.True(File.Exists(ModelFixture.RequireModelOrSkip("mnist", parts))));
        WithLaneVariable("1", () => Assert.True(File.Exists(ModelFixture.RequireModelOrSkip("mnist", parts))));
    }
}
