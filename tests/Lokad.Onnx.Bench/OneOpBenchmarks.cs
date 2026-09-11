namespace Lokad.Onnx.Bench;

using System;
using System.Collections.Generic;
using System.IO;

using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Order;
using BenchmarkDotNet.Running;

using Lokad.Onnx;

using Microsoft.ML.OnnxRuntime;
using OrtTensors = Microsoft.ML.OnnxRuntime.Tensors;

internal static class OneOpMicro
{
    internal static void RunOneOp(string[] args)
    {
        Console.WriteLine("Running one-op ORT session comparisons...");
        BenchmarkRunner.Run<OneOpBenchmarks>(DefaultConfig.Instance, args);
    }
}

/// <summary>
/// One-op session comparison: the same frozen one-op model file drives a
/// Lokad graph and a single-CPU ORT session on identical inputs, separating
/// operator and session overhead from kernel-only timings elsewhere.
/// Agreement runs before any timing; a breach fails the run.
/// </summary>
[InProcess]
[MemoryDiagnoser]
[IterationsColumn]
[GroupBenchmarksBy(BenchmarkLogicalGroupRule.ByCategory)]
[Orderer(methodOrderPolicy: BenchmarkDotNet.Order.MethodOrderPolicy.Declared)]
public class OneOpBenchmarks
{
    const int Seed = 20260910;
    const double Tolerance = 1e-4;

    sealed class Case
    {
        internal Case(string name, string inputName, ComputationalGraph graph, InferenceSession session, SessionOptions sessionOptions, ITensor lokadInput, OrtTensors.DenseTensor<float> ortInput, ExecutionOptions execOptions)
        {
            Name = name;
            InputName = inputName;
            Graph = graph;
            Session = session;
            SessionOptions = sessionOptions;
            LokadInput = lokadInput;
            OrtInput = ortInput;
            ExecOptions = execOptions;
        }

        internal string Name;
        internal string InputName;
        internal ComputationalGraph Graph;
        internal InferenceSession Session;
        internal SessionOptions SessionOptions;
        internal ITensor LokadInput;
        internal OrtTensors.DenseTensor<float> OrtInput;
        internal ExecutionOptions ExecOptions;
    }

    readonly List<Case> cases = new List<Case>();

    static string ModelPath(string kase) =>
        Path.Combine(global::Bench.FindRoot(), "tests", "Lokad.Onnx.Bench", "oneop", kase, "model.onnx");

    [GlobalSetup]
    public void Setup()
    {
        Add("matmul_384", "a", new int[] { 384, 384 });
        Add("softmax_attn", "x", new int[] { 12, 30, 30 });
        Add("layernorm_384", "x", new int[] { 257, 384 });
        Add("gelu_384", "x", new int[] { 257, 384 });
        Add("conv_3x3", "x", new int[] { 1, 64, 28, 28 });
        Add("gemm_1024x256", "a", new int[] { 1024, 256 });
        Add("gemm_512x4608", "a", new int[] { 512, 4608 });
        Add("conv_3x3_512", "x", new int[] { 1, 512, 14, 14 });
        Add("conv_1x1_1024", "x", new int[] { 1, 256, 14, 14 });
        Add("matmul_30x384x1536", "a", new int[] { 1, 30, 384 });
        Add("matmul_30x1536x384", "a", new int[] { 1, 30, 1536 });
        Add("matmul_30x384x384", "a", new int[] { 1, 30, 384 });
        Add("gelu_30x1536", "x", new int[] { 1, 30, 1536 });
        Add("transpose_30x12x32", "x", new int[] { 1, 30, 12, 32 });
        Add("gemm_4x768x2304", "a", new int[] { 4, 768 });
        Add("gemm_4x3072x768", "a", new int[] { 4, 3072 });
        Add("matmul_1x201x384x1536", "a", new int[] { 1, 201, 384 });
        Add("matmul_1x201x1536x384", "a", new int[] { 1, 201, 1536 });
        Add("matmul_1x201x384x384", "a", new int[] { 1, 201, 384 });
        Add("softmax_6x201x201", "x", new int[] { 6, 201, 201 });
        Add("gelu_1x201x1536", "x", new int[] { 1, 201, 1536 });
        Add("transpose_1x201x6x64", "x", new int[] { 1, 201, 6, 64 });
        foreach (var c in cases) VerifyAgreement(c);
        Console.WriteLine("OneOp agreement: all 22 cases match element-wise.");
    }

    void Add(string kase, string inputName, int[] dims)
    {
        string path = ModelPath(kase);
        if (!File.Exists(path)) throw new FileNotFoundException("one-op model missing: " + path);
        int n = 1;
        foreach (var d in dims) n *= d;
        var rnd = new Random(Seed);
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)rnd.NextDouble() * 2f - 1f;
        var lokadInput = new DenseTensor<float>(data, dims);
        var ortInput = new OrtTensors.DenseTensor<float>(data, dims);
        var graph = OnnxImport.Load(path);
        if (graph is null) throw new InvalidOperationException("one-op model failed to load: " + path);
        var so = global::Bench.CreateSingleCpuSessionOptions(1);
        var session = new InferenceSession(path, so);
        var execOptions = new ExecutionOptions(OptimizationMode.Speed, TensorExecutionOptions.Auto);
        cases.Add(new Case(kase, inputName, graph, session, so, lokadInput, ortInput, execOptions));
    }

    static float[] RunLokad(Case c)
    {
        if (!c.Graph.Execute(new ITensor[] { c.LokadInput }, true, ExecutionProvider.CPU, c.ExecOptions))
            throw new InvalidOperationException("Lokad execution failed for " + c.Name);
        return ((Tensor<float>)c.Graph.Outputs["y"]).ToArray();
    }

    static float[] RunOrt(Case c)
    {
        using (var results = c.Session.Run(new[] { NamedOnnxValue.CreateFromTensor(c.InputName, c.OrtInput) }, new[] { "y" }))
        {
            foreach (var r in results)
            {
                if (r.Name == "y") return r.AsTensor<float>().ToArray();
            }
        }
        throw new InvalidOperationException("ORT output missing for " + c.Name);
    }

    static void VerifyAgreement(Case c)
    {
        var lokad = RunLokad(c);
        var ort = RunOrt(c);
        if (lokad.Length != ort.Length) throw new InvalidOperationException(c.Name + ": shape mismatch.");
        double worst = 0;
        for (int i = 0; i < lokad.Length; i++)
        {
            double d = Math.Abs(lokad[i] - ort[i]) / (1.0 + Math.Abs(ort[i]));
            if (d > worst) worst = d;
        }
        Console.WriteLine("OneOp " + c.Name + ": agreement maxScaled=" + worst.ToString("E2"));
        if (worst > Tolerance) throw new InvalidOperationException(c.Name + ": agreement breach.");
    }

    [Benchmark(Description = "MatMul 384x384 - Lokad session")]
    [BenchmarkCategory("matmul")]
    public void LokadMatMul() => RunLokad(cases[0]);

    [Benchmark(Description = "MatMul 384x384 - ORT session")]
    [BenchmarkCategory("matmul")]
    public void OrtMatMul() => RunOrt(cases[0]);

    [Benchmark(Description = "Softmax 12x30x30 - Lokad session")]
    [BenchmarkCategory("softmax")]
    public void LokadSoftmax() => RunLokad(cases[1]);

    [Benchmark(Description = "Softmax 12x30x30 - ORT session")]
    [BenchmarkCategory("softmax")]
    public void OrtSoftmax() => RunOrt(cases[1]);

    [Benchmark(Description = "LayerNorm 257x384 - Lokad session")]
    [BenchmarkCategory("layernorm")]
    public void LokadLayerNorm() => RunLokad(cases[2]);

    [Benchmark(Description = "LayerNorm 257x384 - ORT session")]
    [BenchmarkCategory("layernorm")]
    public void OrtLayerNorm() => RunOrt(cases[2]);

    [Benchmark(Description = "Gelu 257x384 - Lokad session")]
    [BenchmarkCategory("gelu")]
    public void LokadGelu() => RunLokad(cases[3]);

    [Benchmark(Description = "Gelu 257x384 - ORT session")]
    [BenchmarkCategory("gelu")]
    public void OrtGelu() => RunOrt(cases[3]);

    [Benchmark(Description = "Conv 64ch 3x3 - Lokad session")]
    [BenchmarkCategory("conv")]
    public void LokadConv() => RunLokad(cases[4]);

    [Benchmark(Description = "Conv 64ch 3x3 - ORT session")]
    [BenchmarkCategory("conv")]
    public void OrtConv() => RunOrt(cases[4]);

    [Benchmark(Description = "Gemm 1024x256 @ 256x196 - Lokad session")]
    [BenchmarkCategory("gemm1024")]
    public void LokadGemm1024() => RunLokad(cases[5]);

    [Benchmark(Description = "Gemm 1024x256 @ 256x196 - ORT session")]
    [BenchmarkCategory("gemm1024")]
    public void OrtGemm1024() => RunOrt(cases[5]);

    [Benchmark(Description = "Gemm 512x4608 @ 4608x196 - Lokad session")]
    [BenchmarkCategory("gemm512")]
    public void LokadGemm512() => RunLokad(cases[6]);

    [Benchmark(Description = "Gemm 512x4608 @ 4608x196 - ORT session")]
    [BenchmarkCategory("gemm512")]
    public void OrtGemm512() => RunOrt(cases[6]);

    [Benchmark(Description = "Conv 512ch 3x3 over 1x512x14x14 - Lokad session")]
    [BenchmarkCategory("conv512")]
    public void LokadConv512() => RunLokad(cases[7]);

    [Benchmark(Description = "Conv 512ch 3x3 over 1x512x14x14 - ORT session")]
    [BenchmarkCategory("conv512")]
    public void OrtConv512() => RunOrt(cases[7]);

    [Benchmark(Description = "Conv 1x1 256to1024 over 1x256x14x14 - Lokad session")]
    [BenchmarkCategory("conv1024")]
    public void LokadConv1024() => RunLokad(cases[8]);

    [Benchmark(Description = "Conv 1x1 256to1024 over 1x256x14x14 - ORT session")]
    [BenchmarkCategory("conv1024")]
    public void OrtConv1024() => RunOrt(cases[8]);
    [Benchmark(Description = "MatMul 1x30x384 @ 384x1536 - Lokad session")]
    [BenchmarkCategory("mm30up")]
    public void LokadMm30Up() => RunLokad(cases[9]);

    [Benchmark(Description = "MatMul 1x30x384 @ 384x1536 - ORT session")]
    [BenchmarkCategory("mm30up")]
    public void OrtMm30Up() => RunOrt(cases[9]);

    [Benchmark(Description = "MatMul 1x30x1536 @ 1536x384 - Lokad session")]
    [BenchmarkCategory("mm30down")]
    public void LokadMm30Down() => RunLokad(cases[10]);

    [Benchmark(Description = "MatMul 1x30x1536 @ 1536x384 - ORT session")]
    [BenchmarkCategory("mm30down")]
    public void OrtMm30Down() => RunOrt(cases[10]);

    [Benchmark(Description = "MatMul 1x30x384 @ 384x384 - Lokad session")]
    [BenchmarkCategory("mm30qkv")]
    public void LokadMm30Qkv() => RunLokad(cases[11]);

    [Benchmark(Description = "MatMul 1x30x384 @ 384x384 - ORT session")]
    [BenchmarkCategory("mm30qkv")]
    public void OrtMm30Qkv() => RunOrt(cases[11]);

    [Benchmark(Description = "Gelu 1x30x1536 - Lokad session")]
    [BenchmarkCategory("gelu30")]
    public void LokadGelu30() => RunLokad(cases[12]);

    [Benchmark(Description = "Gelu 1x30x1536 - ORT session")]
    [BenchmarkCategory("gelu30")]
    public void OrtGelu30() => RunOrt(cases[12]);

    [Benchmark(Description = "Transpose 1x30x12x32 - Lokad session")]
    [BenchmarkCategory("tr30")]
    public void LokadTr30() => RunLokad(cases[13]);

    [Benchmark(Description = "Transpose 1x30x12x32 - ORT session")]
    [BenchmarkCategory("tr30")]
    public void OrtTr30() => RunOrt(cases[13]);

    [Benchmark(Description = "Gemm 4x768 @ 768x2304 - Lokad session")]
    [BenchmarkCategory("g432")]
    public void LokadGemm4x768() => RunLokad(cases[14]);

    [Benchmark(Description = "Gemm 4x768 @ 768x2304 - ORT session")]
    [BenchmarkCategory("g432")]
    public void OrtGemm4x768() => RunOrt(cases[14]);

    [Benchmark(Description = "Gemm 4x3072 @ 3072x768 - Lokad session")]
    [BenchmarkCategory("g4768")]
    public void LokadGemm4x3072() => RunLokad(cases[15]);

    [Benchmark(Description = "Gemm 4x3072 @ 3072x768 - ORT session")]
    [BenchmarkCategory("g4768")]
    public void OrtGemm4x3072() => RunOrt(cases[15]);

    [Benchmark(Description = "MatMul 1x201x384 @ 384x1536 - Lokad session")]
    [BenchmarkCategory("mm201up")]
    public void LokadMm201Up() => RunLokad(cases[16]);

    [Benchmark(Description = "MatMul 1x201x384 @ 384x1536 - ORT session")]
    [BenchmarkCategory("mm201up")]
    public void OrtMm201Up() => RunOrt(cases[16]);

    [Benchmark(Description = "MatMul 1x201x1536 @ 1536x384 - Lokad session")]
    [BenchmarkCategory("mm201down")]
    public void LokadMm201Down() => RunLokad(cases[17]);

    [Benchmark(Description = "MatMul 1x201x1536 @ 1536x384 - ORT session")]
    [BenchmarkCategory("mm201down")]
    public void OrtMm201Down() => RunOrt(cases[17]);

    [Benchmark(Description = "MatMul 1x201x384 @ 384x384 - Lokad session")]
    [BenchmarkCategory("mm201qkv")]
    public void LokadMm201Qkv() => RunLokad(cases[18]);

    [Benchmark(Description = "MatMul 1x201x384 @ 384x384 - ORT session")]
    [BenchmarkCategory("mm201qkv")]
    public void OrtMm201Qkv() => RunOrt(cases[18]);

    [Benchmark(Description = "Softmax 6x201x201 - Lokad session")]
    [BenchmarkCategory("sm201")]
    public void LokadSm201() => RunLokad(cases[19]);

    [Benchmark(Description = "Softmax 6x201x201 - ORT session")]
    [BenchmarkCategory("sm201")]
    public void OrtSm201() => RunOrt(cases[19]);

    [Benchmark(Description = "Gelu 1x201x1536 - Lokad session")]
    [BenchmarkCategory("gelu201")]
    public void LokadGelu201() => RunLokad(cases[20]);

    [Benchmark(Description = "Gelu 1x201x1536 - ORT session")]
    [BenchmarkCategory("gelu201")]
    public void OrtGelu201() => RunOrt(cases[20]);

    [Benchmark(Description = "Transpose 1x201x6x64 - Lokad session")]
    [BenchmarkCategory("tr201")]
    public void LokadTr201() => RunLokad(cases[21]);

    [Benchmark(Description = "Transpose 1x201x6x64 - ORT session")]
    [BenchmarkCategory("tr201")]
    public void OrtTr201() => RunOrt(cases[21]);
}
