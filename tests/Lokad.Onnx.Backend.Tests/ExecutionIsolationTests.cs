using System.Threading.Tasks;
using Lokad.Onnx.Tests.Support;

namespace Lokad.Onnx.Backend.Tests;

public class ExecutionIsolationTests
{
    static float[] RunToArray(ComputationalGraph graph, string image, ExecutionOptions options)
    {
        var ui = Data.GetInputTensorsFromFileArgs(new[] { TestSupport.CommittedImage(image) + "::mnist" })!;
        Assert.True(graph.Execute(ui, true, ExecutionProvider.CPU, options));
        var output = (Tensor<float>)graph.Outputs.Values.First()!;
        return output.ToArray();
    }

    [SkippableFact]
    public async Task ScalarVsIntrinsics_MatchSerialReferences()
    {
        Skip.If(!System.Runtime.Intrinsics.X86.Fma.IsSupported, "x86 FMA not available on this machine.");
        var gScalar = OnnxImport.Load(TestSupport.CommittedModel("mnist-8.onnx"))!;
        var gIntrinsics = OnnxImport.Load(TestSupport.CommittedModel("mnist-8.onnx"))!;
        var serialScalar = RunToArray(gScalar, "mnist4.png", ExecutionOptions.Scalar);
        var serialIntrinsics = RunToArray(gIntrinsics, "mnist2.png", ExecutionOptions.Intrinsics);

        var r = await Task.WhenAll(
            Task.Run(() =>
            {
                var gg = OnnxImport.Load(TestSupport.CommittedModel("mnist-8.onnx"))!;
                return RunToArray(gg, "mnist4.png", ExecutionOptions.Scalar);
            }),
            Task.Run(() =>
            {
                var gg = OnnxImport.Load(TestSupport.CommittedModel("mnist-8.onnx"))!;
                return RunToArray(gg, "mnist2.png", ExecutionOptions.Intrinsics);
            }));

        Assert.Equal(serialScalar, r[0]);
        Assert.Equal(serialIntrinsics, r[1]);
    }
}
