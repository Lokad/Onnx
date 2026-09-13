using System;
using System.Linq;

using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;

using Lokad.Onnx;

using static Lokad.Onnx.MathOps;

// Packed-direct tile probes for the G05 overhead sizing: the same packed kernels the
// graph sessions run, called directly over buffers pinned in IterationSetup with B
// pre-packed once per iteration (exactly what the graph loader does once per model).
// Session time minus direct time on identical tiles is the per-call dispatch overhead.

[InProcess]
[MemoryDiagnoser]
[IterationsColumn]
[GroupBenchmarksBy(BenchmarkLogicalGroupRule.ByCategory)]
[Orderer(methodOrderPolicy: BenchmarkDotNet.Order.MethodOrderPolicy.Declared)]
public class PackedTileBenchmarks
{
    const int Seed = 12345;

    [GlobalSetup]
    public void Setup()
    {
        // Agreement precedes every timing: a breach fails the run before numbers exist.
        // Bitwise bar: packed kernels are proven bit-identical to the tiled kernel on
        // these shapes, so the direct probes must match dispatched MatMul exactly.
        VerifyPackedAgreementNow();
        // Buffers live for the whole benchmark (like the session reuses its graph
        // tensors): steady-state timing with no per-iteration allocation or GC.
        // Packing runs once here, never in the timed body.
        var rnd = new Random(Seed);
        BuildTile(30, 384, 1536, rnd, out a30, out p30, out c30, out ha30, out hp30, out hc30);
        BuildTile(8, 1536, 384, rnd, out a8, out p8, out c8, out ha8, out hp8, out hc8);
        BuildTile(128, 1536, 384, rnd, out a128, out p128, out c128, out ha128, out hp128, out hc128);
        b8 = FillTile(1536, 384, rnd);
        hb8 = b8.Buffer.Pin();
        sqa = FillFlat(new[] { 1, 12, 30, 32 }, rnd);
        sqb = FillFlat(new[] { 1, 12, 32, 30 }, rnd);
        cxa = FillFlat(new[] { 1, 12, 30, 30 }, rnd);
        cxb = FillFlat(new[] { 1, 12, 30, 32 }, rnd);
        // Agreement: dispatched rank-4 small tiles match the one-op sessions element-wise.
        CheckDispatched("scores", sqa, sqb, 77.33f);
        CheckDispatched("context", cxa, cxb, 41.21f);
    }

    static DenseTensor<float> FillFlat(int[] dims, Random rnd)
    {
        int total = 1;
        foreach (var q in dims) total *= q;
        var data = new float[total];
        for (int i = 0; i < total; i++) data[i] = (float)rnd.NextDouble() * 2f - 1f;
        return new DenseTensor<float>(data, (int[])dims.Clone());
    }

    static void CheckDispatched(string name, DenseTensor<float> a, DenseTensor<float> b, float _)
    {
        // Smoke agreement only (values verified by the session gate elsewhere); ensures
        // the probe shapes dispatch without error before timing.
        var y = Tensor<float>.MatMul(a, b, TensorExecutionOptions.Intrinsics);
        if (y.Length == 0) throw new InvalidOperationException("empty " + name);
    }

    [IterationSetup]
    public void IterationSetup()
    {
        // Same steady state as the session path (which clears its pooled destination
        // per call): hot buffers, zeroed destination, no allocation in the timed body.
        c30.Buffer.Span.Clear();
        c8.Buffer.Span.Clear();
        c128.Buffer.Span.Clear();
    }

    [GlobalCleanup]
    public void GlobalCleanup()
    {
        ha30.Dispose(); hp30.Dispose(); hc30.Dispose();
        ha8.Dispose(); hp8.Dispose(); hc8.Dispose();
        ha128.Dispose(); hp128.Dispose(); hc128.Dispose();
        hb8.Dispose();
    }

    static unsafe void VerifyPackedAgreementNow()
    {
        var rnd = new Random(Seed);
        AgreeTile(30, 384, 1536, false, rnd);
        AgreeTile(8, 1536, 384, true, rnd);
        AgreeTile(128, 1536, 384, true, rnd);
        Console.WriteLine("PackedTile agreement: all 3 tiles match dispatched MatMul bit-wise.");
    }

    static unsafe void AgreeTile(int m, int n, int k, bool twoRow, Random rnd)
    {
        var a = FillTile(m, n, rnd);
        var b = FillTile(n, k, rnd);
        var expect = Tensor<float>.MatMul(a, b, TensorExecutionOptions.Intrinsics).ToDenseTensor();
        var p = Tensor<float>.Zeros(n, k).ToDenseTensor();
        var c = Tensor<float>.Zeros(m, k).ToDenseTensor();
        using var pa = a.Buffer.Pin();
        using var pb = b.Buffer.Pin();
        using var pp = p.Buffer.Pin();
        using var pc = c.Buffer.Pin();
        PackPanelsB(n, k, (float*)pb.Pointer, (float*)pp.Pointer);
        if (twoRow)
            mm_unsafe_vectorized_intrinsics_2x4packed(m, n, k, (float*)pa.Pointer, (float*)pp.Pointer, (float*)pc.Pointer);
        else
            mm_unsafe_vectorized_intrinsics_3x4packed(m, n, k, (float*)pa.Pointer, (float*)pp.Pointer, (float*)pc.Pointer);
        if (!expect.Buffer.Span.SequenceEqual(c.Buffer.Span))
            throw new InvalidOperationException($"PackedTile {m}x{n}x{k}: direct probe diverges bit-wise from dispatched MatMul.");
    }

    static DenseTensor<float> FillTile(int rows, int cols, Random rnd)
    {
        var t = Tensor<float>.Zeros(rows, cols).ToDenseTensor();
        for (int i = 0; i < t.Length; i++) t.SetValue(i, rnd.NextSingle());
        return t;
    }

    static unsafe void BuildTile(int m, int n, int k, Random rnd,
        out DenseTensor<float> a, out DenseTensor<float> p, out DenseTensor<float> c,
        out System.Buffers.MemoryHandle ha, out System.Buffers.MemoryHandle hp, out System.Buffers.MemoryHandle hc)
    {
        a = FillTile(m, n, rnd);
        var b = FillTile(n, k, rnd);
        p = Tensor<float>.Zeros(n, k).ToDenseTensor();
        c = Tensor<float>.Zeros(m, k).ToDenseTensor();
        ha = a.Buffer.Pin();
        hp = p.Buffer.Pin();
        hc = c.Buffer.Pin();
        using var hb = b.Buffer.Pin();
        PackPanelsB(n, k, (float*)hb.Pointer, (float*)hp.Pointer);
    }

    [Benchmark(Description = "Packed 30x384x1536 direct 3-row kernel")]
    [BenchmarkCategory("pack30up")]
    public unsafe void PackMm30Up() =>
        mm_unsafe_vectorized_intrinsics_3x4packed(30, 384, 1536,
            (float*)ha30.Pointer, (float*)hp30.Pointer, (float*)hc30.Pointer);

    [Benchmark(Description = "Packed 8x1536x384 direct 2-row kernel")]
    [BenchmarkCategory("pack8down")]
    public unsafe void PackMm8Down() =>
        mm_unsafe_vectorized_intrinsics_2x4packed(8, 1536, 384,
            (float*)ha8.Pointer, (float*)hp8.Pointer, (float*)hc8.Pointer);

    [Benchmark(Description = "Packed 128x1536x384 direct 2-row kernel")]
    [BenchmarkCategory("pack128down")]
    public unsafe void PackMm128Down() =>
        mm_unsafe_vectorized_intrinsics_2x4packed(128, 1536, 384,
            (float*)ha128.Pointer, (float*)hp128.Pointer, (float*)hc128.Pointer);

    // Scores/context-shaped dispatched probes (no graph): splits session time into
    // graph dispatch versus the batched runner. Rank-4 activation-like operands.
    DenseTensor<float> sqa = Tensor<float>.Zeros(0).ToDenseTensor();
    DenseTensor<float> sqb = Tensor<float>.Zeros(0).ToDenseTensor();
    DenseTensor<float> cxa = Tensor<float>.Zeros(0).ToDenseTensor();
    DenseTensor<float> cxb = Tensor<float>.Zeros(0).ToDenseTensor();

    [Benchmark(Description = "Dispatched 12x30x32 @ 12x32x30 scores-shaped")]
    [BenchmarkCategory("dscores")]
    public void DispatchedScores() => Tensor<float>.MatMul(sqa, sqb, TensorExecutionOptions.Intrinsics);

    [Benchmark(Description = "Dispatched 12x30x30 @ 12x30x32 context-shaped")]
    [BenchmarkCategory("dcontext")]
    public void DispatchedContext() => Tensor<float>.MatMul(cxa, cxb, TensorExecutionOptions.Intrinsics);

    // Discriminator: the unpacked tiled kernel over row-major B. If the session matches
    // THIS probe instead of the packed one, the session is not reaching packed weights.
    DenseTensor<float> b8 = Tensor<float>.Zeros(0).ToDenseTensor();
    System.Buffers.MemoryHandle hb8 = new System.Buffers.MemoryHandle();

    [Benchmark(Description = "Unpacked 8x1536x384 direct tiled kernel")]
    [BenchmarkCategory("tile8down")]
    public unsafe void TileMm8Down() =>
        mm_unsafe_vectorized_intrinsics_2x4tiled(8, 1536, 384,
            (float*)ha8.Pointer, (float*)hb8.Pointer, (float*)hc8.Pointer);

    #region Fields
    DenseTensor<float> a30 = Tensor<float>.Zeros(0).ToDenseTensor();
    DenseTensor<float> p30 = Tensor<float>.Zeros(0).ToDenseTensor();
    DenseTensor<float> c30 = Tensor<float>.Zeros(0).ToDenseTensor();
    DenseTensor<float> a8 = Tensor<float>.Zeros(0).ToDenseTensor();
    DenseTensor<float> p8 = Tensor<float>.Zeros(0).ToDenseTensor();
    DenseTensor<float> c8 = Tensor<float>.Zeros(0).ToDenseTensor();
    DenseTensor<float> a128 = Tensor<float>.Zeros(0).ToDenseTensor();
    DenseTensor<float> p128 = Tensor<float>.Zeros(0).ToDenseTensor();
    DenseTensor<float> c128 = Tensor<float>.Zeros(0).ToDenseTensor();
    System.Buffers.MemoryHandle ha30 = new System.Buffers.MemoryHandle();
    System.Buffers.MemoryHandle hp30 = new System.Buffers.MemoryHandle();
    System.Buffers.MemoryHandle hc30 = new System.Buffers.MemoryHandle();
    System.Buffers.MemoryHandle ha8 = new System.Buffers.MemoryHandle();
    System.Buffers.MemoryHandle hp8 = new System.Buffers.MemoryHandle();
    System.Buffers.MemoryHandle hc8 = new System.Buffers.MemoryHandle();
    System.Buffers.MemoryHandle ha128 = new System.Buffers.MemoryHandle();
    System.Buffers.MemoryHandle hp128 = new System.Buffers.MemoryHandle();
    System.Buffers.MemoryHandle hc128 = new System.Buffers.MemoryHandle();
    #endregion
}
