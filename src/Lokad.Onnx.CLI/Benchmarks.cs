namespace Lokad.Onnx.CLI;

using System;
using System.Buffers;
using System.IO;
using System.Linq;
using System.Text.Json;

using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Running;

using static Lokad.Onnx.Text;
using static Lokad.Onnx.MathOps;

using Lokad.Onnx;

[RyuJitX64Job]
[IterationsColumn]
[MemoryDiagnoser]
[DisassemblyDiagnoser(printSource:true)]

public class MatMul2DBenchmarks : Runtime
{
    [GlobalSetup]
    public void Setup()
    {
        Initialize("Lokad.Onnx.CLI Benchmarks", "CLI", false, true, true);
    }

    [IterationSetup]
    public void IterationSetup()
    {
        t_384_384_a = Tensor<float>.Rand(384, 384);
        ah_1 = t_384_384_a.ToDenseTensor().Buffer.Pin();
        t_384_384_b = Tensor<float>.Rand(384, 384);
        bh_1 = t_384_384_b.ToDenseTensor().Buffer.Pin();
        ch = t_384_384_c.ToDenseTensor().Buffer.Pin();
        t_384_1536_a = Tensor<float>.Rand(384, 1536);
        ah_2 = t_384_1536_a.ToDenseTensor().Buffer.Pin();
        t_1536_384_b = Tensor<float>.Rand(1536, 384);
        bh_2 = t_1536_384_b.ToDenseTensor().Buffer.Pin();
    }

    [Benchmark(Description = "Multiply 2 384x384 matrices - managed", Baseline = true)]
    public void MatMul2D_1() =>
        mm_managed(384, 384, 384, t_384_384_a.ToDenseTensor().Buffer, t_384_384_a.ToDenseTensor().Buffer, t_384_384_c.ToDenseTensor().Buffer);

    [Benchmark(Description = "Multiply 2 384x384 matrices - unsafe")]
    public unsafe void MatMul2D_2() =>
       mm(384, 384, 384, (float*)ah_1.Pointer, (float*)bh_1.Pointer, (float*)ch.Pointer);

    [Benchmark(Description = "Multiply 2 384x384 matrices - managed simd")]
    public void MatMul2D_3() =>
        mm_vectorized(384, 384, 384, t_384_384_a.ToDenseTensor().Buffer, t_384_384_a.ToDenseTensor().Buffer, t_384_384_c.ToDenseTensor().Buffer);

    [Benchmark(Description = "Multiply 2 384x384 matrices - unsafe simd")]
    public unsafe void MatMul2D_4() =>
       mm_unsafe_vectorized(384, 384, 384, (float*)ah_1.Pointer, (float*)bh_1.Pointer, (float*)ch.Pointer);

    #region Fields
    Tensor<float> t_384_384_a = Tensor<float>.Zeros(0);
    Tensor<float> t_384_384_b = Tensor<float>.Zeros(0);
    Tensor<float> t_384_384_c = Tensor<float>.Zeros(384, 384);
    Tensor<float> t_384_1536_a = Tensor<float>.Zeros(0);
    Tensor<float> t_1536_384_b = Tensor<float>.Zeros(0);

    MemoryHandle ah_1 = new MemoryHandle();
    MemoryHandle bh_1 = new MemoryHandle();
    MemoryHandle ah_2 = new MemoryHandle();
    MemoryHandle bh_2 = new MemoryHandle();
    MemoryHandle ch = new MemoryHandle();
    #endregion
}

[InProcess]
[IterationsColumn]
[MemoryDiagnoser]
[GroupBenchmarksBy(BenchmarkLogicalGroupRule.ByCategory)]
public class TensorMatMulBenchmarks : Runtime
{
    [GlobalSetup]
    public void Setup()
    {
        t_384_384_a = Tensor<float>.Rand(384, 384);
        t_384_384_b = Tensor<float>.Rand(384, 384);
        t_384_1536_a = Tensor<float>.Rand(384, 1536);
        t_1536_384_b = Tensor<float>.Rand(1536, 384);
        t_3_4_384_384_a = Tensor<float>.Rand(4,1, 384, 384);
        t_3_4_384_384_b = Tensor<float>.Rand(4, 1, 384, 384);
    }

    [IterationSetup(Targets = ["MatMul_simd", "MatMul2_simd", "MatMul3_simd"])]
    public void EnableSimd() => HardwareConfig.UseSimd = true;
    
    [IterationSetup(Targets = ["MatMul", "MatMul2", "MatMul3"])]
    public void DisableSimd() => HardwareConfig.UseSimd = false;
   
    [Benchmark(Description = "Matrix multiply 2 384x384 tensors", Baseline = true)]
    [BenchmarkCategory("384x384")]
    public void MatMul() => Tensor<float>.MatMul(t_384_384_a, t_384_384_b);

    [Benchmark(Description = "Matrix multiply 2 384x384 tensors - simd")]
    [BenchmarkCategory("384x384")]
    public void MatMul_simd() => Tensor<float>.MatMul(t_384_384_a, t_384_384_b);

    [Benchmark(Description = "Matrix multiply 2 384x1536 tensors - simd")]
    [BenchmarkCategory("384x1536")]
    public void MatMul2_simd() => Tensor<float>.MatMul(t_384_1536_a, t_1536_384_b);

    [Benchmark(Description = "Matrix multiply 2 384x1536 tensors", Baseline = true)]
    [BenchmarkCategory("384x1536")]
    public void MatMul2() => Tensor<float>.MatMul(t_384_1536_a, t_1536_384_b);
    [BenchmarkCategory("3x4x384x384")]
    [Benchmark(Description = "Matrix multiply 2 3x4x384x384 tensors - simd")]
    public void MatMul3_simd() => Tensor<float>.MatMul(t_3_4_384_384_a, t_3_4_384_384_b);

    [Benchmark(Description = "Matrix multiply 2 3x4x384x384 tensors", Baseline = true)]
    [BenchmarkCategory("3x4x384x384")]
    public void MatMul3() => Tensor<float>.MatMul(t_3_4_384_384_a, t_3_4_384_384_b);

    #region Fields
    Tensor<float> t_384_384_a = Tensor<float>.Zeros(0);
    Tensor<float> t_384_384_b = Tensor<float>.Zeros(0);
    Tensor<float> t_384_1536_a = Tensor<float>.Zeros(0);
    Tensor<float> t_1536_384_b = Tensor<float>.Zeros(0);
    Tensor<float> t_3_4_384_384_a = Tensor<float>.Zeros(0);
    Tensor<float> t_3_4_384_384_b = Tensor<float>.Zeros(0);
    #endregion
}

[InProcess]
[IterationsColumn]
[MemoryDiagnoser]
[GroupBenchmarksBy(BenchmarkLogicalGroupRule.ByCategory)]
[Orderer(methodOrderPolicy: BenchmarkDotNet.Order.MethodOrderPolicy.Declared)]
public class TensorIndexingBenchmarks : Runtime
{
    [IterationSetup]
    public void Setup()
    {
        t_384_384_dense = Tensor<float>.Rand(384, 384);
        t_384_384_slice = t_384_384_dense[..];
        t_384_384_bcast = t_384_384_dense.PadLeft().BroadcastDim(0, 2);  
    }

    [Benchmark(Baseline = true, Description = "Multi-dim index into a 384x384 dense tensor")]
    [BenchmarkCategory("multidim")]
    public void MultiDimIndexDenseTensor()
    {
        var a = 0.0f;
        var di = t_384_384_dense.GetDimensionsIterator();
        foreach (var _ in di)
        {
            a += t_384_384_dense[_];
        }
    }

    [Benchmark(Description = "Multi-dim index into a 384x384 tensor slice")]
    [BenchmarkCategory("multidim")]
    public void MultidimIndexTensorSlice()
    {
        var a = 0.0f;
        var di = t_384_384_slice.GetDimensionsIterator();
        foreach (var _ in di)
        {
            a += t_384_384_slice[_];
        }
    }

    [Benchmark(Description = "Multi-dim index into a 2x384x384 broadcasted tensor")]
    [BenchmarkCategory("multidim")]
    public void MultidimIndexBroadcastedTensor()
    {
        var a = 0.0f;
        var di = t_384_384_slice.GetDimensionsIterator();
        foreach (var _ in di)
        {
            a += t_384_384_bcast[_];
        }
    }
    [Benchmark(Baseline = true, Description = "Scalar index into a 384x384 dense tensor")]
    [BenchmarkCategory("scalar")]
    public void GetValueDenseTensor()
    {
        var a = 0.0f;
        var di = t_384_384_dense.GetDimensionsIterator();
        for(int i = 0; i < t_384_384_dense.Length; i++)
        {
            a += t_384_384_dense.GetValue(i);
        }
    }

    [Benchmark(Description = "Scalar index into a 384x384 tensor slice")]
    [BenchmarkCategory("scalar")]
    public void GetValueSlice()
    {
        var a = 0.0f;
        for (int i = 0; i < t_384_384_slice.Length; i++)
        {
            a += t_384_384_slice.GetValue(i);
        }
    }

    [Benchmark(Description = "Scalar index into a 2x384x384 broadcasted tensor")]
    [BenchmarkCategory("scalar")]
    public void GetValueBroadcastedTensor()
    {
        var a = 0.0f;
        for (int i = 0; i < t_384_384_slice.Length; i++)
        {
            a += t_384_384_bcast.GetValue(i);
        }
    }
    #region Fields
    Tensor<float> t_384_384_dense = Tensor<float>.Zeros(0);
    Tensor<float> t_384_384_slice = Tensor<float>.Zeros(0);
    Tensor<float> t_384_384_bcast = Tensor<float>.Zeros(0);
    #endregion
}

[InProcess]
[IterationsColumn]
[GroupBenchmarksBy(BenchmarkLogicalGroupRule.ByCategory)]
[Orderer(methodOrderPolicy: BenchmarkDotNet.Order.MethodOrderPolicy.Declared)]
public class MultilingualEmbedded5SmallBenchmarks : Runtime
{
    [GlobalSetup()]
    public void Setup()
    {
        var op = Begin("Creating computational graph and tokenizing test data");
        graph = Model.Load(modelFile);
        var options = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true
        };
        TextData?[] textData = File.ReadAllLines(testDataFile).AsParallel().Select(t => JsonSerializer.Deserialize<TextData>(t, options)).ToArray();
        Random rnd = new Random();
        T20 = textData
            .AsParallel()
            .Where(t => t is not null && t!.Text.Length >= 21 && t!.Text[20] == ' ')
            .Select(t => t!.Text.Substring(0, 20)/*.Replace("\n", " ")*/)
            .OrderBy(x => rnd.Next())
            .ToArray();
        T200 = textData
            .AsParallel()
            .Where(t => t is not null && t!.Text.Length >= 201 && t!.Text[200] == ' ')
            .Select(t => t!.Text.Substring(0, 200)/*.Replace("\n", " ")*/)
            .OrderBy(x => rnd.Next())
            .ToArray();

        ui20_1 = GetTextTensors(T20[0], "me5s");
        ui20_10 = GetTextTensors(T20[1..11], "me5s");
        ui20_100 = GetTextTensors(T20[11..111], "me5s");
        op.Complete();
    }

    [IterationSetup(Targets = ["Benchmark20_1", "Benchmark20_10"])]
    public void SetupNoSimd()
    {
        graph!.Reset();
        HardwareConfig.UseSimd = false;
    }

    [IterationSetup(Targets = ["Benchmark20_1_simd", "Benchmark20_10_simd"])]
    public void SetupSimd()
    {
        graph!.Reset();
        HardwareConfig.UseSimd = true;
    }

    [Benchmark(Description="1 string of 20 chars", Baseline = true)]
    [BenchmarkCategory("1_20")]
    [WarmupCount(1)]
    [IterationCount(5)]
    public void Benchmark20_1() => graph!.Execute(ui20_1!, true);

    [Benchmark(Description = "1 string of 20 chars - simd")]
    [BenchmarkCategory("1_20")]
    [WarmupCount(1)]
    [IterationCount(5)]
    public void Benchmark20_1_simd() => graph!.Execute(ui20_1!, true);

    [Benchmark(Description = "10 strings of 20 chars")]
    [BenchmarkCategory("10_20")]
    [WarmupCount(1)]
    [IterationCount(2)]
    public void Benchmark20_10() => graph!.Execute(ui20_10!, true);

    [Benchmark(Description = "10 strings of 20 chars - simd")]
    [BenchmarkCategory("10_20")]
    [WarmupCount(1)]
    [IterationCount(2)]
    public void Benchmark20_10_simd() => graph!.Execute(ui20_10!, true);

    #region Fields
    string modelFile = Path.Combine(Runtime.AssemblyLocation, "benchmark-model.onnx");
    string testDataFile = Path.Combine(Runtime.AssemblyLocation, "train.jsonl");
    public static string[] T20 = Array.Empty<string>();
    public static string[] T200 = Array.Empty<string>();
    public static ComputationalGraph? graph;
    ITensor[]? ui20_1 = null;
    ITensor[]? ui20_10 = null;
    ITensor[]? ui20_100 = null;
    #endregion
}

internal class Benchmarks : Runtime
{
    internal static void RunMe5s()
    {
        var op = Begin("Preparing model and data for multilingual-embedded-5-small benchmark");
        var modelFile = Path.Combine(AssemblyLocation, "benchmark-model.onnx");
        var testDataFile = Path.Combine(AssemblyLocation, "train.jsonl");
        if (!File.Exists(modelFile))
        {
            if (!DownloadFile("benchmark-model.onnx", new Uri("https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/onnx/model.onnx?download=true"), modelFile))
            {
                Error("Could not download benchmark model file.");
                op.Abandon();
                return;
            }
        }

        if (!File.Exists(testDataFile))//
        {  //https://huggingface.co/datasets/mteb/quora/resolve/main/corpus.jsonl
            if (!DownloadFile("train.jsonl", new Uri("https://huggingface.co/datasets/mteb/amazon_reviews_multi/resolve/main/en/train.jsonl?download=true"), testDataFile))
            {
                Error("Could not download benchmark test data file.");
                op.Abandon();
                return;
            }
        }
        op.Complete();
        BenchmarkRunner.Run<MultilingualEmbedded5SmallBenchmarks>();
    }

    internal static void RunMatMul()
    {
        Info("Hardware config: {s}.", HardwareIntrinsics.GetFullInfo());
        BenchmarkRunner.Run<TensorMatMulBenchmarks>();
    }

    internal static void RunIndexing()
    {
        BenchmarkRunner.Run<TensorIndexingBenchmarks>();
    }

    internal static void RunMatMul2D(string[] args)
    {
        Info("SIMD hardware acceleration: {a}.", System.Numerics.Vector.IsHardwareAccelerated);
        Info("SIMD vector size: {v} bits.", System.Numerics.Vector<int>.Count * 4 * 8);
        Info("Creating new build of Lokad.Onnx solution to run and profile MatMul2D benchmark code...");
        BenchmarkRunner.Run<MatMul2DBenchmarks>(DefaultConfig.Instance, args);
    }
}

public partial class TextData
{
    public string Id { get; set; } = "";

    public string Text { get; set; } = "";

    public long Label { get; set; } = 0;

    public string Label_Text { get; set; } = "";
}

public class BenchmarkConfig : ManualConfig
{
}