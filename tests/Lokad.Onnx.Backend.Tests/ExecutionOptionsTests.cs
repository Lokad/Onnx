using System.IO;

namespace Lokad.Onnx.Backend.Tests;

[Collection("ProcessState")]
public class ExecutionOptionsTests
{
    [Fact]
    public void TensorPresets_CarryFlags()
    {
        Assert.False(TensorExecutionOptions.Scalar.UseSimd);
        Assert.False(TensorExecutionOptions.Scalar.UseIntrinsics);
        Assert.True(TensorExecutionOptions.Simd.UseSimd);
        Assert.False(TensorExecutionOptions.Simd.UseIntrinsics);
        Assert.True(TensorExecutionOptions.Intrinsics.UseSimd);
        Assert.True(TensorExecutionOptions.Intrinsics.UseIntrinsics);
        Assert.Equal(1, TensorExecutionOptions.Scalar.MaxDegreeOfParallelism);
        Assert.Equal(1, TensorExecutionOptions.Simd.MaxDegreeOfParallelism);
        Assert.Equal(1, TensorExecutionOptions.Intrinsics.MaxDegreeOfParallelism);
        Assert.Equal(1, TensorExecutionOptions.Auto.MaxDegreeOfParallelism);
        var parallel = TensorExecutionOptions.Parallel(4);
        Assert.True(parallel.UseSimd);
        Assert.True(parallel.UseIntrinsics);
        Assert.Equal(4, parallel.MaxDegreeOfParallelism);
    }

    [Fact]
    public void TensorAuto_ReflectsHardwareConfig()
    {
#pragma warning disable CS0618 // This test pins the documented Auto-resolves-legacy-defaults contract.
        bool simd = HardwareConfig.UseSimd;
        bool intr = HardwareConfig.UseIntrinsics;
        try
        {
            HardwareConfig.UseSimd = false;
            HardwareConfig.UseIntrinsics = false;
            Assert.False(TensorExecutionOptions.Auto.UseSimd);
            HardwareConfig.UseSimd = true;
            Assert.True(TensorExecutionOptions.Auto.UseSimd);
        }
        finally
        {
            HardwareConfig.UseSimd = simd;
            HardwareConfig.UseIntrinsics = intr;
        }
#pragma warning restore CS0618
    }

    [Fact]
    public void MatMul2D_ExplicitScalar_MatchesDefault()
    {
        var a = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f } });
        var b = DenseTensor<float>.OfValues(new float[,] { { 5f, 6f }, { 7f, 8f } });
        var expected = Tensor<float>.MatMul2D(a, b);
        var actual = Tensor<float>.MatMul2D(a, b, TensorExecutionOptions.Scalar);
        Assert.Equal(expected.Dimensions.ToArray(), actual.Dimensions.ToArray());
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                Assert.Equal(expected[i, j], actual[i, j], 5);
            }
        }
    }

    [Fact]
    public void SupportedOps_IsReadOnly()
    {
        Assert.IsAssignableFrom<System.Collections.Generic.IReadOnlyList<OpType>>(CPUExecutionProvider.SupportedOps);
        Assert.Contains(OpType.MatMul, CPUExecutionProvider.SupportedOps);
    }

    [Fact]
    public async Task Graphs_WithDifferentOptions_RunConcurrently()
    {
        var modelPath = Path.Combine(Directory.GetCurrentDirectory(), "models", "mnist-8.onnx");
        var imageArg = Path.Combine(Directory.GetCurrentDirectory(), "images", "mnist4.png") + "::mnist";
        var baselineGraph = Model.Load(modelPath)!;
        var ui = Data.GetInputTensorsFromFileArgs(new[] { imageArg })!;
        Assert.True(baselineGraph.Execute(ui, true));
        var baseline = ((Tensor<float>)baselineGraph.Outputs.Values.First()).ToArray();

        var tasks = new Task<float[]>[4];
        for (int i = 0; i < tasks.Length; i++)
        {
            int mode = i;
            tasks[i] = Task.Run(() =>
            {
                var g = Model.Load(modelPath)!;
                g.Options = mode % 2 == 0
                    ? new ExecutionOptions(OptimizationMode.Speed, TensorExecutionOptions.Scalar)
                    : new ExecutionOptions(OptimizationMode.Memory, TensorExecutionOptions.Intrinsics);
                var inputs = Data.GetInputTensorsFromFileArgs(new[] { imageArg })!;
                Assert.True(g.Execute(inputs, true, ExecutionProvider.CPU, g.Options));
                return ((Tensor<float>)g.Outputs.Values.First()).ToArray();
            });
        }
        var results = await Task.WhenAll(tasks);
        foreach (var r in results)
        {
            Assert.Equal(baseline.Length, r.Length);
            for (int i = 0; i < r.Length; i++)
            {
                Assert.Equal(baseline[i], r[i], 4);
            }
        }
    }
}
