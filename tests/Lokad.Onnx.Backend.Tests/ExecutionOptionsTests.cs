using System.IO;

namespace Lokad.Onnx.Backend.Tests;

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
    public void TensorAuto_ProbesHardwareDirectly()
    {
        Assert.True(TensorExecutionOptions.Auto.UseSimd);
        Assert.Equal(System.Runtime.Intrinsics.X86.Fma.IsSupported, TensorExecutionOptions.Auto.UseIntrinsics);
        Assert.Equal(1, TensorExecutionOptions.Auto.MaxDegreeOfParallelism);
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
        var baselineGraph = OnnxImport.Load(modelPath)!;
        var ui = Data.GetInputTensorsFromFileArgs(new[] { imageArg })!;
        Assert.True(baselineGraph.Execute(ui, true));
        var baseline = ((Tensor<float>)baselineGraph.Outputs.Values.First()).ToArray();

        var tasks = new Task<float[]>[4];
        for (int i = 0; i < tasks.Length; i++)
        {
            int mode = i;
            tasks[i] = Task.Run(() =>
            {
                var g = OnnxImport.Load(modelPath)!;
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

    [Fact]
    public void BatchedMatMul_Parallel_MatchesSequential_AndRepeatsBitwise()
    {
        static ComputationalGraph TinyBatchedMatMul()
        {
            var graph = new ComputationalGraph();
            graph.Metadata["Name"] = "test";
            var ad = new float[24];
            var bd = new float[24];
            for (int i = 0; i < 24; i++) { ad[i] = 0.25f * i + 1f; bd[i] = 0.125f * i - 1f; }
            var a = new DenseTensor<float>(ad, new[] { 4, 2, 3 });
            var b = new DenseTensor<float>(bd, new[] { 4, 3, 2 });
            graph.Inputs["a"] = a;
            graph.Inputs["b"] = b;
            graph.Outputs["c"] = DenseTensor<float>.OfShape(4, 2, 2);
            graph.Nodes.Add(new Node { Name = "mm", Op = OpType.MatMul, Inputs = new[] { "a", "b" }, Outputs = new[] { "c" } });
            graph.RefreshLifetimeAnalysis();
            return graph;
        }
        static float[] Run(ComputationalGraph graph, ExecutionOptions options)
        {
            var inputs = new System.Collections.Generic.Dictionary<string, ITensor>
            {
                { "a", graph.Inputs["a"] },
                { "b", graph.Inputs["b"] },
            };
            Assert.True(graph.Execute(inputs, true, ExecutionProvider.CPU, options));
            return ((Tensor<float>)graph.Outputs["c"]).ToArray();
        }
        var sequential = Run(TinyBatchedMatMul(), new ExecutionOptions(OptimizationMode.Speed, TensorExecutionOptions.Intrinsics));
        var parallel = Run(TinyBatchedMatMul(), new ExecutionOptions(OptimizationMode.Speed, TensorExecutionOptions.Parallel(4)));
        var repeat = Run(TinyBatchedMatMul(), new ExecutionOptions(OptimizationMode.Speed, TensorExecutionOptions.Parallel(4)));
        Assert.Equal(sequential.Length, parallel.Length);
        for (int i = 0; i < sequential.Length; i++)
        {
            Assert.Equal(sequential[i], parallel[i], 5);
        }
        Assert.Equal(parallel, repeat);
    }
}
