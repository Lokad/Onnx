using System.Threading.Tasks;

namespace Lokad.Onnx.Backend.Tests;

public class ExecutionIsolationTests
{
    static float[] RunToArray(ComputationalGraph graph, string image, ExecutionOptions options)
    {
        var ui = Data.GetInputTensorsFromFileArgs(new[] { "images\\" + image + "::mnist" })!;
        Assert.True(graph.Execute(ui, true, ExecutionProvider.CPU, options));
        var output = (Tensor<float>)graph.Outputs.Values.First()!;
        return output.ToArray();
    }

    [Fact]
    public async Task ScalarVsIntrinsics_MatchSerialReferences()
    {
        var gScalar = OnnxImport.Load("models\\mnist-8.onnx")!;
        var gIntrinsics = OnnxImport.Load("models\\mnist-8.onnx")!;
        var serialScalar = RunToArray(gScalar, "mnist4.png", ExecutionOptions.Scalar);
        var serialIntrinsics = RunToArray(gIntrinsics, "mnist2.png", ExecutionOptions.Intrinsics);

        var r = await Task.WhenAll(
            Task.Run(() =>
            {
                var gg = OnnxImport.Load("models\\mnist-8.onnx")!;
                return RunToArray(gg, "mnist4.png", ExecutionOptions.Scalar);
            }),
            Task.Run(() =>
            {
                var gg = OnnxImport.Load("models\\mnist-8.onnx")!;
                return RunToArray(gg, "mnist2.png", ExecutionOptions.Intrinsics);
            }));

        Assert.Equal(serialScalar, r[0]);
        Assert.Equal(serialIntrinsics, r[1]);
    }
}
