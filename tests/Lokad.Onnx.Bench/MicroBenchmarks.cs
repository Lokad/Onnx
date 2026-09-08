namespace Lokad.Onnx.Bench;

using System;
using System.Buffers;
using System.Linq;

using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;

using Lokad.Onnx;

using static Lokad.Onnx.MathOps;
using static Lokad.Onnx.Runtime;

// Microbenchmark workloads moved from the CLI (BenchmarkDotNet travels with
// them). The corpus-dependent entries stay behind: their model and data files
// do not exist in the repo. Console writes replace CLI logging calls.

public class MatMul2DBenchmarks
{
    [GlobalSetup]
    public void Setup()
    {
    }

    [IterationSetup]
    public void IterationSetup()
    {
        // Same deterministic inputs and a freshly zeroed destination for every
        // compared variant; conversion runs once here, never in the timed body.
        var rnd = new Random(Seed);
        t_384_384_a = FillDeterministic(384, 384, rnd);
        t_384_384_b = FillDeterministic(384, 384, rnd);
        t_384_384_c = Tensor<float>.Zeros(384, 384);
        da = t_384_384_a.ToDenseTensor();
        db = t_384_384_b.ToDenseTensor();
        dc = t_384_384_c.ToDenseTensor();
        ah_1 = da.Buffer.Pin();
        bh_1 = db.Buffer.Pin();
        ch = dc.Buffer.Pin();
    }

    [IterationCleanup]
    public void IterationCleanup()
    {
        ah_1.Dispose();
        bh_1.Dispose();
        ch.Dispose();
    }

    [GlobalCleanup]
    public void VerifyAgreement()
    {
        // Recompute both paths on fresh destinations outside the timed loop and
        // require checksum agreement within float-reorder tolerance.
        var expected = Tensor<float>.Zeros(384, 384).ToDenseTensor();
        mm_managed(384, 384, 384, da.Buffer, db.Buffer, expected.Buffer);
        var actual = Tensor<float>.Zeros(384, 384).ToDenseTensor();
        unsafe
        {
            using var pa = da.Buffer.Pin();
            using var pb = db.Buffer.Pin();
            using var pc = actual.Buffer.Pin();
            mm_unsafe_vectorized_intrinsics(384, 384, 384, (float*)pa.Pointer, (float*)pb.Pointer, (float*)pc.Pointer);
        }
        float s1 = Checksum(expected);
        float s2 = Checksum(actual);
        float tolerance = 1e-3f * Math.Max(1f, Math.Abs(s1));
        Console.WriteLine("MatMul2D checksum agreement: managed=" + s1 + " intrinsics=" + s2 + ".");
        if (Math.Abs(s1 - s2) > tolerance)
            throw new InvalidOperationException($"MatMul2D variants disagree: managed={s1} intrinsics={s2}.");
    }


    [Benchmark(Description = "Multiply 2 384x384 matrices - managed")]
    public void MatMul2D_1() =>
        mm_managed(384, 384, 384, da.Buffer, db.Buffer, dc.Buffer);

    [Benchmark(Description = "Multiply 2 384x384 matrices - managed simd")]
    public void MatMul2D_3() =>
      mm_vectorized(384, 384, 384, da.Buffer, db.Buffer, dc.Buffer);

    [Benchmark(Description = "Multiply 2 384x384 matrices - unsafe")]
    public unsafe void MatMul2D_2() =>
       mm(384, 384, 384, (float*)ah_1.Pointer, (float*)bh_1.Pointer, (float*)ch.Pointer);
  
    [Benchmark(Description = "Multiply 2 384x384 matrices - unsafe simd")]
    public unsafe void MatMul2D_4() =>
       mm_unsafe_vectorized(384, 384, 384, (float*)ah_1.Pointer, (float*)bh_1.Pointer, (float*)ch.Pointer);

    [Benchmark(Description = "Multiply 2 384x384 matrices - unsafe simd intrinsics", Baseline = true)]
    public unsafe void MatMul2D_5() =>
      mm_unsafe_vectorized_intrinsics(384, 384, 384, (float*)ah_1.Pointer, (float*)bh_1.Pointer, (float*)ch.Pointer);

    [Benchmark(Description = "Multiply 2 384x384 matrices - unsafe simd intrinsics pointers 2x4")]
    public unsafe void MatMul2D_6() =>
      mm_unsafe_vectorized_intrinsics_2x4(384, 384, 384, (float*)ah_1.Pointer, (float*)bh_1.Pointer, (float*)ch.Pointer);
    static DenseTensor<float> FillDeterministic(int rows, int cols, Random rnd)
    {
        var t = Tensor<float>.Zeros(rows, cols).ToDenseTensor();
        for (int i = 0; i < t.Length; i++) t.SetValue(i, rnd.NextSingle());
        return t;
    }

    static float Checksum(DenseTensor<float> t)
    {
        float s = 0f;
        for (int i = 0; i < t.Length; i++) s += t.GetValue(i);
        return s;
    }

    #region Fields
    Tensor<float> t_384_384_a = Tensor<float>.Zeros(0);
    Tensor<float> t_384_384_b = Tensor<float>.Zeros(0);
    Tensor<float> t_384_384_c = Tensor<float>.Zeros(384, 384);
    const int Seed = 12345;
    DenseTensor<float> da = Tensor<float>.Zeros(0).ToDenseTensor();
    DenseTensor<float> db = Tensor<float>.Zeros(0).ToDenseTensor();
    DenseTensor<float> dc = Tensor<float>.Zeros(384, 384).ToDenseTensor();

    MemoryHandle ah_1 = new MemoryHandle();
    MemoryHandle bh_1 = new MemoryHandle();
    MemoryHandle ch = new MemoryHandle();
    #endregion
}

[InProcess]
[MemoryDiagnoser]
[IterationsColumn]
[GroupBenchmarksBy(BenchmarkLogicalGroupRule.ByCategory)]
[Orderer(methodOrderPolicy: BenchmarkDotNet.Order.MethodOrderPolicy.Declared)]
public class TensorMatMulBenchmarks
{
    [GlobalSetup]
    public void Setup()
    {
        t_384_384_a = Tensor<float>.Rand(384, 384);
        t_384_384_b = Tensor<float>.Rand(384, 384);
        t_384_1536_a = Tensor<float>.Rand(384, 1536);
        t_1536_384_b = Tensor<float>.Rand(1536, 384);
        t_6_384_384_a = Tensor<float>.Rand(6, 384, 384);
        t_6_384_384_b = Tensor<float>.Rand(6, 384, 384);
        t_3_4_384_384_a = Tensor<float>.Rand(3,4, 384, 384);
        t_3_4_384_384_b = Tensor<float>.Rand(3, 4, 384, 384);
    }

    [Benchmark(Description = "Matrix multiply 2 384x384 tensors", Baseline = true)]
    [BenchmarkCategory("384x384")]
    public void MatMul() => Tensor<float>.MatMul(t_384_384_a, t_384_384_b, TensorExecutionOptions.Scalar);

    [Benchmark(Description = "Matrix multiply 2 384x384 tensors - simd")]
    [BenchmarkCategory("384x384")]
    public void MatMul_simd() => Tensor<float>.MatMul(t_384_384_a, t_384_384_b, TensorExecutionOptions.Simd);

    [Benchmark(Description = "Matrix multiply 2 384x384 tensors - simd intrinsics")]
    [BenchmarkCategory("384x384")]
    public void MatMul_simd_intrinsics() => Tensor<float>.MatMul(t_384_384_a, t_384_384_b, TensorExecutionOptions.Intrinsics);

    [Benchmark(Description = "Matrix multiply 2 384x1536 tensors", Baseline = true)]
    [BenchmarkCategory("384x1536")]
    public void MatMul2() => Tensor<float>.MatMul(t_384_1536_a, t_1536_384_b, TensorExecutionOptions.Scalar);

    [Benchmark(Description = "Matrix multiply 2 384x1536 tensors - simd")]
    [BenchmarkCategory("384x1536")]
    public void MatMul2_simd() => Tensor<float>.MatMul(t_384_1536_a, t_1536_384_b, TensorExecutionOptions.Simd);

    [Benchmark(Description = "Matrix multiply 2 384x1536 tensors - simd intrinsics")]
    [BenchmarkCategory("384x1536")]
    public void MatMul2_simd_intrinsics() => Tensor<float>.MatMul(t_384_1536_a, t_1536_384_b, TensorExecutionOptions.Intrinsics);

    [Benchmark(Description = "Matrix multiply 2 6x384x384 tensors", Baseline = true)]
    [BenchmarkCategory("6x384x384")]
    public void MatMul3() => Tensor<float>.MatMul(t_6_384_384_a, t_6_384_384_b, TensorExecutionOptions.Scalar);

    [Benchmark(Description = "Matrix multiply 2 6x384x384 tensors - simd")]
    [BenchmarkCategory("6x384x384")]
    public void MatMul3_simd() => Tensor<float>.MatMul(t_6_384_384_a, t_6_384_384_b, TensorExecutionOptions.Simd);

    [Benchmark(Description = "Matrix multiply 2 6x384x384 tensors - simd intrinsics")]
    [BenchmarkCategory("6x384x384")]
    public void MatMul3_simd_intrinsics() => Tensor<float>.MatMul(t_6_384_384_a, t_6_384_384_b, TensorExecutionOptions.Intrinsics);

    [Benchmark(Description = "Matrix multiply 2 3x4x384x384 tensors", Baseline = true)]
    [BenchmarkCategory("3x4x384x384")]
    public void MatMul4() => Tensor<float>.MatMul(t_3_4_384_384_a, t_3_4_384_384_b, TensorExecutionOptions.Scalar);

    [Benchmark(Description = "Matrix multiply 2 3x4x384x384 tensors - simd")]
    [BenchmarkCategory("3x4x384x384")]
    public void MatMul4_simd() => Tensor<float>.MatMul(t_3_4_384_384_a, t_3_4_384_384_b, TensorExecutionOptions.Simd);

    [Benchmark(Description = "Matrix multiply 2 3x4x384x384 tensors - simd intrinsics")]
    [BenchmarkCategory("3x4x384x384")]
    public void MatMul4_simd_intrinsics() => Tensor<float>.MatMul(t_3_4_384_384_a, t_3_4_384_384_b, TensorExecutionOptions.Intrinsics);

    #region Fields
    Tensor<float> t_384_384_a = Tensor<float>.Zeros(0);
    Tensor<float> t_384_384_b = Tensor<float>.Zeros(0);
    Tensor<float> t_384_1536_a = Tensor<float>.Zeros(0);
    Tensor<float> t_1536_384_b = Tensor<float>.Zeros(0);
    Tensor<float> t_6_384_384_a = Tensor<float>.Zeros(0);
    Tensor<float> t_6_384_384_b = Tensor<float>.Zeros(0);
    Tensor<float> t_3_4_384_384_a = Tensor<float>.Zeros(0);
    Tensor<float> t_3_4_384_384_b = Tensor<float>.Zeros(0);
    #endregion
}

[InProcess]
[IterationsColumn]
[MemoryDiagnoser]
[GroupBenchmarksBy(BenchmarkLogicalGroupRule.ByCategory)]
[Orderer(methodOrderPolicy: BenchmarkDotNet.Order.MethodOrderPolicy.Declared)]
public class TensorIndexingBenchmarks
{
    [IterationSetup]
    public void Setup()
    {
        t_384_384_dense = Tensor<float>.Rand(384, 384);
        t_3_4_384_384_dense = Tensor<float>.Rand(3,4,384, 384);
        t_384_384_slice = t_384_384_dense[..];
        t_384_384_bcast = t_384_384_dense.PadLeft().BroadcastDim(0, 2);
        t_384_384_3_4_bcast = t_384_384_dense.PadLeft().PadLeft().BroadcastDim(0, 3).BroadcastDim(1,4);
    }

    [Benchmark(Description = "Multi-dim index into a 384x384 dense tensor")]
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
        var di = t_384_384_bcast.GetDimensionsIterator();
        foreach (var _ in di)
        {
            a += t_384_384_bcast[_];
        }
    }

    [Benchmark(Description = "Multi-dim index into a 3x4x384x384 dense tensor")]
    [BenchmarkCategory("multidim")]
    public void MultiDimIndexDenseTensor2()
    {
        var a = 0.0f;
        var di = t_3_4_384_384_dense.GetDimensionsIterator();
        foreach (var _ in di)
        {
            a += t_3_4_384_384_dense[_];
        }
    }

    [Benchmark(Description = "Multi-dim index into a 3x4x384x384 broadcasted tensor")]
    [BenchmarkCategory("multidim")]
    public void MultidimIndex34BroadcastedTensor()
    {
        var a = 0.0f;
        var di = t_384_384_3_4_bcast.GetDimensionsIterator();
        foreach (var _ in di)
        {
            a += t_384_384_3_4_bcast[_];
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

    [Benchmark(Description = "Scalar index into a 3x4x384x384 broadcasted tensor")]
    [BenchmarkCategory("scalar")]
    public void GetValue34BroadcastedTensor()
    {
        var a = 0.0f;
        for (int i = 0; i < t_384_384_3_4_bcast.Length; i++)
        {
            a += t_384_384_3_4_bcast.GetValue(i);
        }
    }

    #region Fields
    Tensor<float> t_384_384_dense = Tensor<float>.Zeros(0);
    Tensor<float> t_3_4_384_384_dense = Tensor<float>.Zeros(0);
    Tensor<float> t_384_384_slice = Tensor<float>.Zeros(0);
    Tensor<float> t_384_384_bcast = Tensor<float>.Zeros(0);
    Tensor<float> t_384_384_3_4_bcast = Tensor<float>.Zeros(0);
    int[] di = new int[2];

    #endregion
}

[InProcess]
[MemoryDiagnoser]
[IterationsColumn]
[GroupBenchmarksBy(BenchmarkLogicalGroupRule.ByCategory)]
[Orderer(methodOrderPolicy: BenchmarkDotNet.Order.MethodOrderPolicy.Declared)]
public class TensorOpBenchmarks
{
    [GlobalSetup]
    public void Setup()
    {
        sm_e5 = Tensor<float>.Rand(12, 30, 30);
        sm_dino = Tensor<float>.Rand(6, 257, 257);
        ln_x = Tensor<float>.Rand(257, 384);
        ln_s = Tensor<float>.Rand(384);
        ln_b = Tensor<float>.Rand(384);
        ew_a = Tensor<float>.Rand(257, 384);
        ew_b = Tensor<float>.Rand(257, 384);
        gath_t = Tensor<float>.Rand(1000, 384);
        gath_i = Tensor<int>.Zeros(30);
        for (int i = 0; i < 30; i++) gath_i[i] = (i * 7919) % 1000;
        tr_x = Tensor<float>.Rand(12, 30, 30);
        cc_a = Tensor<float>.Rand(201, 192);
        cc_b = Tensor<float>.Rand(201, 192);
        cv_stem_x = Tensor<float>.Rand(1, 3, 112, 112);
        cv_stem_w = Tensor<float>.Rand(64, 3, 7, 7);
        cv_33_x = Tensor<float>.Rand(1, 64, 28, 28);
        cv_33_w = Tensor<float>.Rand(64, 64, 3, 3);
        gm_a = Tensor<float>.Rand(4, 768);
        gm_b = Tensor<float>.Rand(768, 2304);
        gm_c = Tensor<float>.Rand(2304);
        th_x = Tensor<float>.Rand(4, 3072);
        sp_x = Tensor<float>.Rand(4, 12, 2304);
        sp_sizes = DenseTensor<long>.OfValues(new long[] { 768L, 768L, 768L });
        gap_x = Tensor<float>.Rand(1, 256, 14, 14);
    }

    [Benchmark(Description = "Softmax over 12x30x30 attention scores")]
    [BenchmarkCategory("softmax")]
    public void SoftmaxAttn() => Tensor<float>.Softmax(sm_e5, -1, null, 13);

    [Benchmark(Description = "Softmax over 12x30x30 attention scores - simd")]
    [BenchmarkCategory("softmax")]
    public void SoftmaxAttn_simd() => Tensor<float>.Softmax(sm_e5, -1, TensorExecutionOptions.Simd, 13);

    [Benchmark(Description = "Softmax over 12x30x30 attention scores - simd intrinsics")]
    [BenchmarkCategory("softmax")]
    public void SoftmaxAttn_simd_intrinsics() => Tensor<float>.Softmax(sm_e5, -1, TensorExecutionOptions.Intrinsics, 13);

    [Benchmark(Description = "Softmax over 6x257x257 attention scores")]
    [BenchmarkCategory("softmax")]
    public void SoftmaxDino() => Tensor<float>.Softmax(sm_dino, -1, null, 13);

    [Benchmark(Description = "Softmax over 6x257x257 attention scores - simd")]
    [BenchmarkCategory("softmax")]
    public void SoftmaxDino_simd() => Tensor<float>.Softmax(sm_dino, -1, TensorExecutionOptions.Simd, 13);

    [Benchmark(Description = "Softmax over 6x257x257 attention scores - simd intrinsics")]
    [BenchmarkCategory("softmax")]
    public void SoftmaxDino_simd_intrinsics() => Tensor<float>.Softmax(sm_dino, -1, TensorExecutionOptions.Intrinsics, 13);

    [Benchmark(Description = "LayerNormalization over 257x384")]
    [BenchmarkCategory("layernorm")]
    public void LayerNorm() => Tensor<float>.LayerNormalization(ln_x, ln_s, ln_b, -1, 1e-5f);

    [Benchmark(Description = "Erf over 257x384", Baseline = true)]
    [BenchmarkCategory("erf")]
    public void Erf() => Tensor<float>.Erf(ew_a, TensorExecutionOptions.Scalar);

    [Benchmark(Description = "Erf over 257x384 - simd")]
    [BenchmarkCategory("erf")]
    public void Erf_simd() => Tensor<float>.Erf(ew_a, TensorExecutionOptions.Simd);

    [Benchmark(Description = "Erf over 257x384 - simd intrinsics")]
    [BenchmarkCategory("erf")]
    public void Erf_simd_intrinsics() => Tensor<float>.Erf(ew_a, TensorExecutionOptions.Intrinsics);

    [Benchmark(Description = "Exact Gelu over 257x384", Baseline = true)]
    [BenchmarkCategory("gelu")]
    public void Gelu() => Tensor<float>.Gelu(ew_a, TensorExecutionOptions.Scalar);

    [Benchmark(Description = "Exact Gelu over 257x384 - simd")]
    [BenchmarkCategory("gelu")]
    public void Gelu_simd() => Tensor<float>.Gelu(ew_a, TensorExecutionOptions.Simd);

    [Benchmark(Description = "Exact Gelu over 257x384 - simd intrinsics")]
    [BenchmarkCategory("gelu")]
    public void Gelu_simd_intrinsics() => Tensor<float>.Gelu(ew_a, TensorExecutionOptions.Intrinsics);

    [Benchmark(Description = "Divide 257x384 tensors", Baseline = true)]
    [BenchmarkCategory("div")]
    public void Div() => Tensor<float>.Divide(ew_a, ew_b, TensorExecutionOptions.Scalar);

    [Benchmark(Description = "Divide 257x384 tensors - simd")]
    [BenchmarkCategory("div")]
    public void Div_simd() => Tensor<float>.Divide(ew_a, ew_b, TensorExecutionOptions.Simd);

    [Benchmark(Description = "Divide 257x384 tensors - simd intrinsics")]
    [BenchmarkCategory("div")]
    public void Div_simd_intrinsics() => Tensor<float>.Divide(ew_a, ew_b, TensorExecutionOptions.Intrinsics);

    [Benchmark(Description = "Add 257x384 tensors", Baseline = true)]
    [BenchmarkCategory("add")]
    public void Add() => Tensor<float>.Add(ew_a, ew_b, TensorExecutionOptions.Scalar);

    [Benchmark(Description = "Add 257x384 tensors - simd")]
    [BenchmarkCategory("add")]
    public void Add_simd() => Tensor<float>.Add(ew_a, ew_b, TensorExecutionOptions.Simd);

    [Benchmark(Description = "Add 257x384 tensors - simd intrinsics")]
    [BenchmarkCategory("add")]
    public void Add_simd_intrinsics() => Tensor<float>.Add(ew_a, ew_b, TensorExecutionOptions.Intrinsics);

    [Benchmark(Description = "Multiply 257x384 tensors", Baseline = true)]
    [BenchmarkCategory("mul")]
    public void Mul() => Tensor<float>.Multiply(ew_a, ew_b, TensorExecutionOptions.Scalar);

    [Benchmark(Description = "Multiply 257x384 tensors - simd")]
    [BenchmarkCategory("mul")]
    public void Mul_simd() => Tensor<float>.Multiply(ew_a, ew_b, TensorExecutionOptions.Simd);

    [Benchmark(Description = "Multiply 257x384 tensors - simd intrinsics")]
    [BenchmarkCategory("mul")]
    public void Mul_simd_intrinsics() => Tensor<float>.Multiply(ew_a, ew_b, TensorExecutionOptions.Intrinsics);

    [Benchmark(Description = "Gather 30 rows from 1000x384 table")]
    [BenchmarkCategory("gather")]
    public void Gather() => Tensor<float>.Gather(gath_t, gath_i, 0);

    [Benchmark(Description = "Transpose 12x30x30 over axes 0,2,1")]
    [BenchmarkCategory("transpose")]
    public void Transpose() => Tensor<float>.Transpose(tr_x, new[] { 0, 2, 1 });

    [Benchmark(Description = "Concat two 201x192 tensors on axis 1")]
    [BenchmarkCategory("concat")]
    public void Concat() => Tensor<float>.Concat(new[] { cc_a, cc_b }, 1);

    [Benchmark(Description = "Conv stem 7x7 s2 pad3 over 1x3x112x112")]
    [BenchmarkCategory("conv")]
    public void ConvStem() => CPUExecutionProvider.Conv(cv_stem_x, cv_stem_w, null, null, null, 1, new[] { 7, 7 }, new[] { 3, 3, 3, 3 }, new[] { 2, 2 }, null);

    [Benchmark(Description = "Conv 3x3 s1 pad1 over 1x64x28x28")]
    [BenchmarkCategory("conv")]
    public void ConvStage() => CPUExecutionProvider.Conv(cv_33_x, cv_33_w, null, null, null, 1, new[] { 3, 3 }, new[] { 1, 1, 1, 1 }, new[] { 1, 1 }, null);

    [Benchmark(Description = "Gemm 4x768 @ 768x2304 + bias (GPT-2 c_attn)")]
    [BenchmarkCategory("gemm")]
    public void GemmAttn() => CPUExecutionProvider.Gemm(gm_a, gm_b, gm_c, 1f, 1f, null, 0, 0);

    [Benchmark(Description = "Tanh over 4x3072 (GPT-2 gelu path)")]
    [BenchmarkCategory("tanh")]
    public void TanhAct() => CPUExecutionProvider.Tanh(th_x, null);

    [Benchmark(Description = "Split 4x12x2304 into 3 QKV parts")]
    [BenchmarkCategory("split")]
    public void SplitQkv() => CPUExecutionProvider.Split(sp_x, sp_sizes, 2, null, null, null, null);

    [Benchmark(Description = "GlobalAveragePool over 1x256x14x14")]
    [BenchmarkCategory("gap")]
    public void GlobalAvgPool() => CPUExecutionProvider.GlobalAveragePool(gap_x, null);

    #region Fields
    Tensor<float> sm_e5 = Tensor<float>.Zeros(0);
    Tensor<float> sm_dino = Tensor<float>.Zeros(0);
    Tensor<float> ln_x = Tensor<float>.Zeros(0);
    Tensor<float> ln_s = Tensor<float>.Zeros(0);
    Tensor<float> ln_b = Tensor<float>.Zeros(0);
    Tensor<float> ew_a = Tensor<float>.Zeros(0);
    Tensor<float> ew_b = Tensor<float>.Zeros(0);
    Tensor<float> gath_t = Tensor<float>.Zeros(0);
    Tensor<int> gath_i = Tensor<int>.Zeros(0);
    Tensor<float> tr_x = Tensor<float>.Zeros(0);
    Tensor<float> cc_a = Tensor<float>.Zeros(0);
    Tensor<float> cc_b = Tensor<float>.Zeros(0);
    Tensor<float> cv_stem_x = Tensor<float>.Zeros(0);
    Tensor<float> cv_stem_w = Tensor<float>.Zeros(0);
    Tensor<float> cv_33_x = Tensor<float>.Zeros(0);
    Tensor<float> cv_33_w = Tensor<float>.Zeros(0);
    Tensor<float> gm_a = Tensor<float>.Zeros(0);
    Tensor<float> gm_b = Tensor<float>.Zeros(0);
    Tensor<float> gm_c = Tensor<float>.Zeros(0);
    Tensor<float> th_x = Tensor<float>.Zeros(0);
    Tensor<float> sp_x = Tensor<float>.Zeros(0);
    Tensor<long> sp_sizes = Tensor<long>.Zeros(0);
    Tensor<float> gap_x = Tensor<float>.Zeros(0);
    #endregion
}

internal static class MicroBenchmarks
{
    internal static void RunMatMul2D(string[] args)
    {
        Console.WriteLine("Running matmul core benchmark...");
        Console.WriteLine("SIMD hardware acceleration: " + System.Numerics.Vector.IsHardwareAccelerated + ".");
        Console.WriteLine("SIMD vector size: " + (System.Numerics.Vector<int>.Count * 4 * 8) + " bits.");
        Console.WriteLine("SIMD supported intrinsics: " + HardwareIntrinsics.GetFullInfo() + ".");
        Console.WriteLine("Running MatMul2D benchmark code...");
        BenchmarkRunner.Run<MatMul2DBenchmarks>(DefaultConfig.Instance, args);
    }

    internal static void RunMatMul(string[] args)
    {
        Console.WriteLine("Running tensor matmul benchmark...");
        Console.WriteLine("SIMD hardware acceleration: " + System.Numerics.Vector.IsHardwareAccelerated + ".");
        Console.WriteLine("SIMD vector size: " + (System.Numerics.Vector<int>.Count * 4 * 8) + " bits.");
        Console.WriteLine("SIMD supported intrinsics: " + HardwareIntrinsics.GetFullInfo() + ".");
        BenchmarkRunner.Run<TensorMatMulBenchmarks>(DefaultConfig.Instance, args);
    }

    internal static void RunIndexing(string[] args)
    {
        BenchmarkRunner.Run<TensorIndexingBenchmarks>(DefaultConfig.Instance, args);
    }

    internal static void RunOps(string[] args)
    {
        Console.WriteLine("Running tensor op microbenchmarks...");
        Console.WriteLine("SIMD hardware acceleration: " + System.Numerics.Vector.IsHardwareAccelerated + ".");
        Console.WriteLine("SIMD vector size: " + (System.Numerics.Vector<int>.Count * 4 * 8) + " bits.");
        Console.WriteLine("SIMD supported intrinsics: " + HardwareIntrinsics.GetFullInfo() + ".");
        BenchmarkRunner.Run<TensorOpBenchmarks>(DefaultConfig.Instance, args);
    }
}
