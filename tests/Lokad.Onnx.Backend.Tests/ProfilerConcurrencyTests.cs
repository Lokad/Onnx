using System.Threading.Tasks;

namespace Lokad.Onnx.Backend.Tests;

public class ProfilerConcurrencyTests
{
    [Fact]
    public async Task ConcurrentProfiledGraphs_AreDisjoint()
    {
        using var profilerScope = Profiler.BeginExecution(true);
        var g1 = OnnxImport.Load("models\\mnist-8.onnx")!;
        var g2 = OnnxImport.Load("models\\mnist-8.onnx")!;
        var ui1 = Data.GetInputTensorsFromFileArgs(new[] { "images\\mnist4.png::mnist" })!;
        var ui2 = Data.GetInputTensorsFromFileArgs(new[] { "images\\mnist2.png::mnist" })!;
        var r = await Task.WhenAll(
            Task.Run(() => g1.Execute(ui1, true, ExecutionProvider.CPU, ExecutionOptions.Scalar)),
            Task.Run(() => g2.Execute(ui2, true, ExecutionProvider.CPU, ExecutionOptions.Scalar)));
        Assert.True(r[0]);
        Assert.True(r[1]);
        Assert.NotNull(g1.LastProfile);
        Assert.NotNull(g2.LastProfile);
        Assert.Equal(g1.Nodes.Count, g1.LastProfile!.Count);
        Assert.Equal(g2.Nodes.Count, g2.LastProfile!.Count);
    }
}
