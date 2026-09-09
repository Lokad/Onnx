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

    static bool agreementChecked;

    [GlobalCleanup]
    public void VerifyAgreement()
    {
        if (agreementChecked) return;
        agreementChecked = true;
        // Recompute every variant on fresh destinations outside the timed loop
        // and require element-wise agreement with the managed reference.
        var rnd = new Random(Seed);
        var xa = FillDeterministic(384, 384, rnd);
        var xb = FillDeterministic(384, 384, rnd);
        var expected = Tensor<float>.Zeros(384, 384).ToDenseTensor();
        ReferenceKernels.mm_managed(384, 384, 384, xa.Buffer, xb.Buffer, expected.Buffer);
        var reference = expected.ToArray();
        var dims = expected.Dimensions.ToArray();
        double tolerance = 1e-3;
        CheckVariant("managed-simd", dims, reference, RunManagedSimd(xa, xb), tolerance);
        CheckVariant("unsafe", dims, reference, RunUnsafe(xa, xb), tolerance);
        CheckVariant("unsafe-simd", dims, reference, RunUnsafeSimd(xa, xb), tolerance);
        CheckVariant("unsafe-simd-intrinsics", dims, reference, RunUnsafeIntrinsics(xa, xb), tolerance);
        CheckVariant("unsafe-simd-intrinsics-2x4", dims, reference, RunUnsafeIntrinsics2x4(xa, xb), tolerance);
        Console.WriteLine("MatMul2D agreement: all 6 variants match element-wise.");
    }

    static void CheckVariant(string name, int[] dims, float[] reference, float[] actual, double tolerance)
    {
        var agree = BenchValidate.RequireAgreement("MatMul2D " + name, dims, reference, dims, actual, tolerance);
        Console.WriteLine("MatMul2D " + name + ": worst scaled diff " + agree.scaled.ToString("E2") + " (abs " + agree.abs.ToString("E2") + ").");
    }

    static float[] RunManagedSimd(DenseTensor<float> xa, DenseTensor<float> xb)
    {
        var dest = Tensor<float>.Zeros(384, 384).ToDenseTensor();
        ReferenceKernels.mm_vectorized(384, 384, 384, xa.Buffer, xb.Buffer, dest.Buffer);
        return dest.ToArray();
    }

    static unsafe float[] RunUnsafe(DenseTensor<float> xa, DenseTensor<float> xb)
    {
        var dest = Tensor<float>.Zeros(384, 384).ToDenseTensor();
        using var pa = xa.Buffer.Pin();
        using var pb = xb.Buffer.Pin();
        using var pc = dest.Buffer.Pin();
        mm(384, 384, 384, (float*)pa.Pointer, (float*)pb.Pointer, (float*)pc.Pointer);
        return dest.ToArray();
    }

    static unsafe float[] RunUnsafeSimd(DenseTensor<float> xa, DenseTensor<float> xb)
    {
        var dest = Tensor<float>.Zeros(384, 384).ToDenseTensor();
        using var pa = xa.Buffer.Pin();
        using var pb = xb.Buffer.Pin();
        using var pc = dest.Buffer.Pin();
        mm_unsafe_vectorized(384, 384, 384, (float*)pa.Pointer, (float*)pb.Pointer, (float*)pc.Pointer);
        return dest.ToArray();
    }

    static unsafe float[] RunUnsafeIntrinsics(DenseTensor<float> xa, DenseTensor<float> xb)
    {
        var dest = Tensor<float>.Zeros(384, 384).ToDenseTensor();
        using var pa = xa.Buffer.Pin();
        using var pb = xb.Buffer.Pin();
        using var pc = dest.Buffer.Pin();
        mm_unsafe_vectorized_intrinsics(384, 384, 384, (float*)pa.Pointer, (float*)pb.Pointer, (float*)pc.Pointer);
        return dest.ToArray();
    }

    static unsafe float[] RunUnsafeIntrinsics2x4(DenseTensor<float> xa, DenseTensor<float> xb)
    {
        var dest = Tensor<float>.Zeros(384, 384).ToDenseTensor();
        using var pa = xa.Buffer.Pin();
        using var pb = xb.Buffer.Pin();
        using var pc = dest.Buffer.Pin();
        mm_unsafe_vectorized_intrinsics_2x4(384, 384, 384, (float*)pa.Pointer, (float*)pb.Pointer, (float*)pc.Pointer);
        return dest.ToArray();
    }


    [Benchmark(Description = "Multiply 2 384x384 matrices - managed")]
    public void MatMul2D_1() =>
        ReferenceKernels.mm_managed(384, 384, 384, da.Buffer, db.Buffer, dc.Buffer);

    [Benchmark(Description = "Multiply 2 384x384 matrices - managed simd")]
    public void MatMul2D_3() =>
      ReferenceKernels.mm_vectorized(384, 384, 384, da.Buffer, db.Buffer, dc.Buffer);

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

    static bool agreementChecked;

    [GlobalCleanup]
    public void VerifyAgreement()
    {
        if (agreementChecked) return;
        agreementChecked = true;
        // Every shape and mode recomputed on fresh seeded inputs outside the
        // timed loop, with simd and intrinsics agreeing element-wise with scalar.
        var rnd = new Random(12345);
        CheckShape("384x384", new[] { 384, 384 }, new[] { 384, 384 }, rnd);
        CheckShape("384x1536", new[] { 384, 1536 }, new[] { 1536, 384 }, rnd);
        CheckShape("6x384x384", new[] { 6, 384, 384 }, new[] { 6, 384, 384 }, rnd);
        CheckShape("3x4x384x384", new[] { 3, 4, 384, 384 }, new[] { 3, 4, 384, 384 }, rnd);
        Console.WriteLine("TensorMatMul agreement: all shapes and modes match element-wise.");
    }

    static void CheckShape(string name, int[] aDims, int[] bDims, Random rnd)
    {
        var a = FillRand(aDims, rnd);
        var b = FillRand(bDims, rnd);
        var expected = Tensor<float>.MatMul(a, b, TensorExecutionOptions.Scalar);
        var reference = expected.ToArray();
        var dims = expected.Dimensions.ToArray();
        var simd = Tensor<float>.MatMul(a, b, TensorExecutionOptions.Simd).ToArray();
        var intr = Tensor<float>.MatMul(a, b, TensorExecutionOptions.Intrinsics).ToArray();
        BenchValidate.RequireAgreement("TensorMatMul " + name + " simd", dims, reference, dims, simd, 1e-4);
        BenchValidate.RequireAgreement("TensorMatMul " + name + " intrinsics", dims, reference, dims, intr, 1e-4);
    }

    static Tensor<float> FillRand(int[] dims, Random rnd)
    {
        int length = 1;
        foreach (var d in dims) length *= d;
        var t = Tensor<float>.Zeros(dims).ToDenseTensor();
        for (int i = 0; i < length; i++) t.SetValue(i, (float)(rnd.NextDouble() * 2 - 1));
        return t;
    }

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

    static bool agreementChecked;

    [GlobalCleanup]
    public void VerifyAgreement()
    {
        if (agreementChecked) return;
        agreementChecked = true;
        // One small deterministic tensor viewed three ways: slice reads must
        // equal dense reads element-wise, broadcast channels must repeat the
        // dense values, and scalar reads must total the iterator reads.
        var dense = Tensor<float>.Zeros(4, 5).ToDenseTensor();
        for (int i = 0; i < 20; i++) dense.SetValue(i, i);
        var slice = dense[..];
        var bcast = dense.PadLeft().BroadcastDim(0, 2);
        BenchValidate.RequireAgreement("indexing slice", dense.Dimensions.ToArray(), dense.ToArray(),
            slice.Dimensions.ToArray(), slice.ToArray(), 0);
        var tiled = new float[40];
        var flat = dense.ToArray();
        for (int k = 0; k < 2; k++) Array.Copy(flat, 0, tiled, k * 20, 20);
        BenchValidate.RequireAgreement("indexing broadcast", new[] { 2, 4, 5 }, tiled,
            bcast.Dimensions.ToArray(), bcast.ToArray(), 0);
        double ScalarSum(Tensor<float> t)
        {
            double sum = 0;
            for (int i = 0; i < t.Length; i++) sum += t.GetValue(i);
            return sum;
        }
        double IteratorSum(Tensor<float> t)
        {
            double sum = 0;
            foreach (var coord in t.GetDimensionsIterator()) sum += t[coord];
            return sum;
        }
        if (ScalarSum(dense) != IteratorSum(dense)) throw new InvalidOperationException("indexing dense scalar/iterator sums disagree.");
        if (ScalarSum(slice) != IteratorSum(slice)) throw new InvalidOperationException("indexing slice scalar/iterator sums disagree.");
        if (ScalarSum(bcast) != IteratorSum(bcast)) throw new InvalidOperationException("indexing broadcast scalar/iterator sums disagree.");
        if (ScalarSum(bcast) != 2 * ScalarSum(dense)) throw new InvalidOperationException("indexing broadcast sum is not twice the dense sum.");
        Console.WriteLine("TensorIndexing agreement: views and access paths match element-wise.");
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

    static bool agreementChecked;

    [GlobalCleanup]
    public void VerifyAgreement()
    {
        if (agreementChecked) return;
        agreementChecked = true;
        var rnd = new Random(12345);
        CheckSoftmax("e5", new[] { 2, 30 }, rnd);
        CheckUnary("erf", rnd);
        CheckUnary("gelu", rnd);
        CheckBinary("div", rnd);
        CheckBinary("add", rnd);
        CheckBinary("mul", rnd);
        CheckLayerNorm(rnd);
        CheckGather(rnd);
        CheckTranspose(rnd);
        CheckConcat(rnd);
        CheckConv(rnd);
        CheckGemm(rnd);
        CheckTanh(rnd);
        CheckSplit(rnd);
        CheckGap(rnd);
        Console.WriteLine("TensorOp agreement done.");
    }

    static Tensor<float> Seeded(int[] dims, Random rnd, double scale)
    {
        int length = 1;
        foreach (var d in dims) length *= d;
        var t = Tensor<float>.Zeros(dims).ToDenseTensor();
        for (int i = 0; i < length; i++) t.SetValue(i, (float)(rnd.NextDouble() * 2 * scale - scale));
        return t;
    }

    static void CheckSoftmax(string name, int[] dims, Random rnd)
    {
        var x = Seeded(dims, rnd, 3.0);
        var reference = Tensor<float>.Softmax(x, -1, TensorExecutionOptions.Scalar, 13).ToArray();
        var shape = Tensor<float>.Softmax(x, -1, TensorExecutionOptions.Scalar, 13).Dimensions.ToArray();
        var simd = Tensor<float>.Softmax(x, -1, TensorExecutionOptions.Simd, 13).ToArray();
        var intr = Tensor<float>.Softmax(x, -1, TensorExecutionOptions.Intrinsics, 13).ToArray();
        BenchValidate.RequireAgreement("softmax " + name + " simd", shape, reference, shape, simd, 1e-4);
        BenchValidate.RequireAgreement("softmax " + name + " intrinsics", shape, reference, shape, intr, 1e-4);
    }

    static void CheckUnary(string name, Random rnd)
    {
        var x = Seeded(new[] { 4, 32 }, rnd, 3.0);
        Tensor<float> reference = name == "erf" ? Tensor<float>.Erf(x, TensorExecutionOptions.Scalar)
            : Tensor<float>.Gelu(x, TensorExecutionOptions.Scalar);
        var expected = reference.ToArray();
        var shape = reference.Dimensions.ToArray();
        Tensor<float> simd = name == "erf" ? Tensor<float>.Erf(x, TensorExecutionOptions.Simd)
            : Tensor<float>.Gelu(x, TensorExecutionOptions.Simd);
        Tensor<float> intr = name == "erf" ? Tensor<float>.Erf(x, TensorExecutionOptions.Intrinsics)
            : Tensor<float>.Gelu(x, TensorExecutionOptions.Intrinsics);
        BenchValidate.RequireAgreement(name + " simd", shape, expected, shape, simd.ToArray(), 1e-4);
        BenchValidate.RequireAgreement(name + " intrinsics", shape, expected, shape, intr.ToArray(), 1e-4);
    }

    static void CheckBinary(string name, Random rnd)
    {
        var a = Seeded(new[] { 4, 32 }, rnd, 3.0);
        var braw = Seeded(new[] { 4, 32 }, rnd, 3.0).ToArray();
        for (int i = 0; i < braw.Length; i++) braw[i] = Math.Abs(braw[i]) + 0.5f;
        var bd = Tensor<float>.Zeros(new[] { 4, 32 }).ToDenseTensor();
        for (int i = 0; i < braw.Length; i++) bd.SetValue(i, braw[i]);
        Tensor<float> reference = name == "div" ? Tensor<float>.Divide(a, bd, TensorExecutionOptions.Scalar)
            : name == "add" ? Tensor<float>.Add(a, bd, TensorExecutionOptions.Scalar)
            : Tensor<float>.Multiply(a, bd, TensorExecutionOptions.Scalar);
        var expected = reference.ToArray();
        var shape = reference.Dimensions.ToArray();
        Tensor<float> simd = name == "div" ? Tensor<float>.Divide(a, bd, TensorExecutionOptions.Simd)
            : name == "add" ? Tensor<float>.Add(a, bd, TensorExecutionOptions.Simd)
            : Tensor<float>.Multiply(a, bd, TensorExecutionOptions.Simd);
        Tensor<float> intr = name == "div" ? Tensor<float>.Divide(a, bd, TensorExecutionOptions.Intrinsics)
            : name == "add" ? Tensor<float>.Add(a, bd, TensorExecutionOptions.Intrinsics)
            : Tensor<float>.Multiply(a, bd, TensorExecutionOptions.Intrinsics);
        BenchValidate.RequireAgreement(name + " simd", shape, expected, shape, simd.ToArray(), 1e-4);
        BenchValidate.RequireAgreement(name + " intrinsics", shape, expected, shape, intr.ToArray(), 1e-4);
    }

    static void CheckLayerNorm(Random rnd)
    {
        var x = Seeded(new[] { 2, 8 }, rnd, 3.0);
        var s = Seeded(new[] { 8 }, rnd, 1.0);
        var b = Seeded(new[] { 8 }, rnd, 1.0);
        var actual = Tensor<float>.LayerNormalization(x, s, b, -1, 1e-5f).ToArray();
        var xa = x.ToArray();
        var sa = s.ToArray();
        var ba = b.ToArray();
        var expected = new float[16];
        for (int i = 0; i < 2; i++)
        {
            double mean = 0;
            for (int j = 0; j < 8; j++) mean += xa[i * 8 + j];
            mean /= 8;
            double variance = 0;
            for (int j = 0; j < 8; j++) variance += (xa[i * 8 + j] - mean) * (xa[i * 8 + j] - mean);
            variance /= 8;
            double inv = 1.0 / Math.Sqrt(variance + 1e-5);
            for (int j = 0; j < 8; j++) expected[i * 8 + j] = (float)((xa[i * 8 + j] - mean) * inv * sa[j] + ba[j]);
        }
        BenchValidate.RequireAgreement("layernorm", new[] { 2, 8 }, expected, new[] { 2, 8 }, actual, 1e-4);
    }

    static void CheckGather(Random rnd)
    {
        var t = Seeded(new[] { 50, 8 }, rnd, 3.0);
        var idx = new int[10];
        for (int i = 0; i < 10; i++) idx[i] = (i * 7919) % 50;
        var indices = DenseTensor<int>.OfValues(idx);
        var actual = Tensor<float>.Gather(t, indices, 0).ToArray();
        var ta = t.ToArray();
        var expected = new float[80];
        for (int i = 0; i < 10; i++)
            for (int j = 0; j < 8; j++) expected[i * 8 + j] = ta[idx[i] * 8 + j];
        BenchValidate.RequireAgreement("gather", new[] { 10, 8 }, expected, new[] { 10, 8 }, actual, 0);
    }

    static void CheckTranspose(Random rnd)
    {
        var x = Seeded(new[] { 4, 5, 6 }, rnd, 3.0);
        var actual = Tensor<float>.Transpose(x, new[] { 0, 2, 1 }).ToArray();
        var xa = x.ToArray();
        var expected = new float[120];
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 6; j++)
                for (int k = 0; k < 5; k++) expected[(i * 6 + j) * 5 + k] = xa[(i * 5 + k) * 6 + j];
        BenchValidate.RequireAgreement("transpose", new[] { 4, 6, 5 }, expected, new[] { 4, 6, 5 }, actual, 0);
    }

    static void CheckConcat(Random rnd)
    {
        var a = Seeded(new[] { 5, 7 }, rnd, 3.0);
        var b = Seeded(new[] { 5, 9 }, rnd, 3.0);
        var actual = Tensor<float>.Concat(new[] { a, b }, 1).ToArray();
        var aa = a.ToArray();
        var ba = b.ToArray();
        var expected = new float[80];
        for (int i = 0; i < 5; i++)
        {
            Array.Copy(aa, i * 7, expected, i * 16, 7);
            Array.Copy(ba, i * 9, expected, i * 16 + 7, 9);
        }
        BenchValidate.RequireAgreement("concat", new[] { 5, 16 }, expected, new[] { 5, 16 }, actual, 0);
    }

    static void CheckProvider(string name, Func<ExecutionOptions?, OpResult> run)
    {
        var scalar = run(ExecutionOptions.Scalar);
        if (scalar.Status != OpStatus.Success) throw new InvalidOperationException(name + ": scalar reference failed.");
        var def = run(null);
        if (def.Status != OpStatus.Success) throw new InvalidOperationException(name + ": default run failed.");
        var expected = (Tensor<float>)scalar.Outputs[0];
        var actual = (Tensor<float>)def.Outputs[0];
        BenchValidate.RequireAgreement(name, expected.Dimensions.ToArray(), expected.ToArray(),
            actual.Dimensions.ToArray(), actual.ToArray(), 1e-4);
    }

    static void CheckConv(Random rnd)
    {
        var x = Seeded(new[] { 1, 2, 5, 5 }, rnd, 1.0);
        var w = Seeded(new[] { 2, 2, 3, 3 }, rnd, 1.0);
        CheckProvider("conv", opt => CPUExecutionProvider.Conv(x, w, null, null, null, 1, new[] { 3, 3 }, new[] { 1, 1, 1, 1 }, new[] { 1, 1 }, opt));
    }

    static void CheckGemm(Random rnd)
    {
        var a = Seeded(new[] { 4, 32 }, rnd, 1.0);
        var b = Seeded(new[] { 32, 48 }, rnd, 1.0);
        var c = Seeded(new[] { 48 }, rnd, 1.0);
        CheckProvider("gemm", opt => CPUExecutionProvider.Gemm(a, b, c, 1f, 1f, opt, 0, 0));
    }

    static void CheckTanh(Random rnd)
    {
        var x = Seeded(new[] { 4, 64 }, rnd, 3.0);
        CheckProvider("tanh", opt => CPUExecutionProvider.Tanh(x, opt));
    }

    static void CheckSplit(Random rnd)
    {
        var x = Seeded(new[] { 2, 24 }, rnd, 3.0);
        var sizes = DenseTensor<long>.OfValues(new long[] { 8L, 8L, 8L });
        var scalar = CPUExecutionProvider.Split(x, sizes, 1, null, null, ExecutionOptions.Scalar, 3);
        if (scalar.Status != OpStatus.Success) throw new InvalidOperationException("split: scalar reference failed.");
        var def = CPUExecutionProvider.Split(x, sizes, 1, null, null, null, 3);
        if (def.Status != OpStatus.Success) throw new InvalidOperationException("split: default run failed.");
        if (scalar.Outputs.Length != 3 || def.Outputs.Length != 3) throw new InvalidOperationException("split: expected 3 outputs.");
        for (int i = 0; i < 3; i++)
        {
            var expected = (Tensor<float>)scalar.Outputs[i];
            var actual = (Tensor<float>)def.Outputs[i];
            BenchValidate.RequireAgreement("split" + i, expected.Dimensions.ToArray(), expected.ToArray(),
                actual.Dimensions.ToArray(), actual.ToArray(), 1e-4);
        }
    }

    static void CheckGap(Random rnd)
    {
        var x = Seeded(new[] { 1, 4, 6, 6 }, rnd, 3.0);
        CheckProvider("gap", opt => CPUExecutionProvider.GlobalAveragePool(x, opt));
    }


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
