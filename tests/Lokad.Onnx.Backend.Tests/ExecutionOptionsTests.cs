using System.IO;
using Lokad.Onnx.Tests.Support;

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

    [SkippableFact]
    public async Task Graphs_WithDifferentOptions_RunConcurrently()
    {
        Skip.If(!System.Runtime.Intrinsics.X86.Fma.IsSupported, "x86 FMA not available on this machine.");
        var modelPath = TestSupport.CommittedModel("mnist-8.onnx");
        var imageArg = TestSupport.CommittedImage("mnist4.png") + "::mnist";
        var baselineGraph = OnnxImport.Load(modelPath)!;
        var ui = Data.GetInputTensorsFromFileArgs(new[] { imageArg })!;
        Assert.True(baselineGraph.Execute(ui, true));
        var baseline = ((Tensor<float>)baselineGraph.Outputs.Values.First()!).ToArray();

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
                return ((Tensor<float>)g.Outputs.Values.First()!).ToArray();
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

    [SkippableFact]
    public void BatchedMatMul_Parallel_MatchesSequential_AndRepeatsBitwise()
    {
        Skip.If(!System.Runtime.Intrinsics.X86.Fma.IsSupported, "x86 FMA not available on this machine.");
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

    [Fact]
    public void Validate_AcceptsPortablePresets_RejectsBadConfigurations()
    {
        foreach (var valid in new[]
        {
            TensorExecutionOptions.Scalar,
            TensorExecutionOptions.Simd,
            TensorExecutionOptions.Auto,
        })
        {
            valid.Validate();
        }
        Assert.Throws<ArgumentOutOfRangeException>(() => new TensorExecutionOptions(true, true, 0).Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => new TensorExecutionOptions(true, false, -2).Validate());
        Assert.Throws<ArgumentException>(() => new TensorExecutionOptions(false, true, 1).Validate());
    }

    [SkippableFact]
    public void Validate_AcceptsIntrinsicsPresets()
    {
        Skip.If(!System.Runtime.Intrinsics.X86.Fma.IsSupported, "x86 FMA not available on this machine.");
        TensorExecutionOptions.Intrinsics.Validate();
        TensorExecutionOptions.Parallel(4).Validate();
    }

    [Fact]
    public void Provider_RejectsInvalidParallelism()
    {
        var a = DenseTensor<float>.OfValues(new float[] { 1f, 2f });
        var b = DenseTensor<float>.OfValues(new float[] { 3f, 4f });
        var bad = new ExecutionOptions(OptimizationMode.Speed, new TensorExecutionOptions(true, true, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => CPUExecutionProvider.Add(a, b, bad, null));
    }

    [Fact]
    public void ScalarMode_OnSimdHardware_MatchesAutomatic()
    {
        var a = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f } });
        var b = DenseTensor<float>.OfValues(new float[,] { { 5f, 6f }, { 7f, 8f } });
        var mmScalar = (Tensor<float>)CPUExecutionProvider.MatMul(a, b, ExecutionOptions.Scalar, null).Outputs[0];
        var mmAuto = (Tensor<float>)CPUExecutionProvider.MatMul(a, b, null, null).Outputs[0];
        Assert.Equal(mmAuto.ToArray(), mmScalar.ToArray());
        var gemmScalar = (Tensor<float>)CPUExecutionProvider.Gemm(a, b, null, 1f, 0f, ExecutionOptions.Scalar, 0, 0).Outputs[0];
        var gemmAuto = (Tensor<float>)CPUExecutionProvider.Gemm(a, b, null, 1f, 0f, null, 0, 0).Outputs[0];
        Assert.Equal(gemmAuto.ToArray(), gemmScalar.ToArray());
        var sm = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f } });
        Assert.Equal(
            Tensor<float>.Softmax(sm, 0, TensorExecutionOptions.Auto, 13).ToArray(),
            Tensor<float>.Softmax(sm, 0, TensorExecutionOptions.Scalar, 13).ToArray());
        Assert.Equal(
            Tensor<float>.Softmax(sm, 1, TensorExecutionOptions.Auto, 13).ToArray(),
            Tensor<float>.Softmax(sm, 1, TensorExecutionOptions.Scalar, 13).ToArray());
        var t = DenseTensor<float>.OfValues(new float[] { -1f, 0f, 1f });
        var tanhScalar = (Tensor<float>)CPUExecutionProvider.Tanh(t, ExecutionOptions.Scalar).Outputs[0];
        var tanhAuto = (Tensor<float>)CPUExecutionProvider.Tanh(t, null).Outputs[0];
        Assert.Equal(tanhAuto.ToArray(), tanhScalar.ToArray());
    }

    [SkippableFact]
    public void Gemm_Threading_MatchesSequential()
    {
        Skip.If(!System.Runtime.Intrinsics.X86.Fma.IsSupported, "x86 FMA not available on this machine.");
        var ad = new float[64 * 8];
        var bd = new float[8 * 4];
        for (int i = 0; i < ad.Length; i++) ad[i] = 0.1f * (i % 7) - 0.3f;
        for (int i = 0; i < bd.Length; i++) bd[i] = 0.05f * (i % 5) + 0.25f;
        var a = new DenseTensor<float>(ad, new[] { 64, 8 });
        var b = new DenseTensor<float>(bd, new[] { 8, 4 });
        var sequential = (Tensor<float>)CPUExecutionProvider.Gemm(a, b, null, 1f, 0f,
            new ExecutionOptions(OptimizationMode.Speed, TensorExecutionOptions.Scalar), 0, 0).Outputs[0];
        var threaded = (Tensor<float>)CPUExecutionProvider.Gemm(a, b, null, 1f, 0f,
            new ExecutionOptions(OptimizationMode.Speed, TensorExecutionOptions.Parallel(4)), 0, 0).Outputs[0];
        Assert.Equal(sequential.ToArray(), threaded.ToArray());
    }

    [Fact]
    public void DoubleErfAndGelu_HonorExplicitScalar()
    {
        var x = DenseTensor<double>.OfValues(new double[] { -1.0, 0.0, 1.0 });
        var erfOverload = typeof(Tensor<double>).GetMethod("Erf",
            new[] { typeof(Tensor<double>), typeof(TensorExecutionOptions) });
        Assert.NotNull(erfOverload);
        var geluOverload = typeof(Tensor<double>).GetMethod("Gelu",
            new[] { typeof(Tensor<double>), typeof(TensorExecutionOptions) });
        Assert.NotNull(geluOverload);
        var erfScalar = (Tensor<double>)erfOverload!.Invoke(null, new object[] { x, TensorExecutionOptions.Scalar })!;
        var erfAuto = Tensor<double>.Erf(x);
        Assert.Equal(erfAuto.ToArray(), erfScalar.ToArray());
        var geluScalar = (Tensor<double>)geluOverload!.Invoke(null, new object[] { x, TensorExecutionOptions.Scalar })!;
        var geluAuto = Tensor<double>.Gelu(x);
        Assert.Equal(geluAuto.ToArray(), geluScalar.ToArray());
    }


}


