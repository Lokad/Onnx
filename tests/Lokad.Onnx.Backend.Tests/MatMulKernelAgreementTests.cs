namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Guards the T02 rule: the raw production MatMul kernels must all compute
/// A-times-B (never A-times-A) and agree with an independent reference
/// within float-reorder tolerance, including accumulation onto a nonzero
/// destination. The naive managed/vectorized baselines live in the Bench
/// project; data here is deterministically seeded so failures reproduce.
/// </summary>
public class MatMulKernelAgreementTests
{
    const int N = 32;
    const int Seed = 777;

    static DenseTensor<float> Fill(int rows, int cols, Random rnd)
    {
        var t = Tensor<float>.Zeros(rows, cols).ToDenseTensor();
        for (int i = 0; i < t.Length; i++) t.SetValue(i, rnd.NextSingle());
        return t;
    }

    static float ReferenceSum(DenseTensor<float> a, DenseTensor<float> b, DenseTensor<float> c)
    {
        // Independent triple-loop reference: sum over C + A*B.
        float total = 0f;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
            {
                float acc = c.GetValue(i * N + j);
                for (int k = 0; k < N; k++) acc += a.GetValue(i * N + k) * b.GetValue(k * N + j);
                total += acc;
            }
        return total;
    }

    static float Sum(DenseTensor<float> t)
    {
        float s = 0f;
        for (int i = 0; i < t.Length; i++) s += t.GetValue(i);
        return s;
    }

    static void Agrees(float actual, float expected, string variant)
    {
        float tolerance = 1e-3f * Math.Max(1f, Math.Abs(expected));
        Assert.True(Math.Abs(actual - expected) <= tolerance,
            $"{variant} disagrees: {actual} vs reference {expected}.");
    }

    static unsafe void RunUnsafe(Action<IntPtr, IntPtr, IntPtr> kernel, DenseTensor<float> a, DenseTensor<float> b, DenseTensor<float> c)
    {
        using var pa = a.Buffer.Pin();
        using var pb = b.Buffer.Pin();
        using var pc = c.Buffer.Pin();
        kernel((IntPtr)pa.Pointer, (IntPtr)pb.Pointer, (IntPtr)pc.Pointer);
    }

    [Fact]
    public void DoubleBasic_MatchesOrt()
    {
        // ORT 1.29 double: [[19,22],[43,50]]. First double MatMul pin.
        var a = DenseTensor<double>.OfValues(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });
        var b = DenseTensor<double>.OfValues(new double[,] { { 5.0, 6.0 }, { 7.0, 8.0 } });
        var r = CPUExecutionProvider.MatMul(a, b, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = (Tensor<double>)r.Outputs![0];
        Assert.Equal(new int[] { 2, 2 }, y.Dimensions.ToArray());
        Assert.Equal(new double[] { 19.0, 22.0, 43.0, 50.0 }, y.ToArray());
    }

    [Fact]
    public void MixedDtype_InputsRejectedCleanly()
    {
        // ORT 1.29 refuses mixed-dtype MatMul at load; every other
        // binary kernel already fails descriptively - MatMul must too.
        var a = DenseTensor<float>.OfValues(new float[,] { { 1f, 2f }, { 3f, 4f } });
        var b = DenseTensor<int>.OfValues(new int[,] { { 1, 0 }, { 0, 1 } });
        var r = CPUExecutionProvider.MatMul(a, b, null, null);
        Assert.Equal(OpStatus.Failure, r.Status);
    }

    [Fact]
    public void OutOfScopeDtypes_RejectedCleanly()
    {
        // int8: ORT refuses int8 MatMul at load (not in MatMul-13), so both
        // sides reject. int64/uint32: ORT computes them, but int64/uint
        // MatMul kernels are out of scope, so these fail descriptively
        // instead of reaching a kernel cast.
        var a8 = DenseTensor<sbyte>.OfValues(new sbyte[,] { { 1, 2 }, { 3, 4 } });
        var b8 = DenseTensor<sbyte>.OfValues(new sbyte[,] { { 1, 0 }, { 0, 1 } });
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.MatMul(a8, b8, null, null).Status);
        var a64 = DenseTensor<long>.OfValues(new long[,] { { 1L, 2L }, { 3L, 4L } });
        var b64 = DenseTensor<long>.OfValues(new long[,] { { 1L, 0L }, { 0L, 1L } });
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.MatMul(a64, b64, null, null).Status);
        var au = DenseTensor<uint>.OfValues(new uint[,] { { 1u, 2u }, { 3u, 4u } });
        var bu = DenseTensor<uint>.OfValues(new uint[,] { { 1u, 0u }, { 0u, 1u } });
        Assert.Equal(OpStatus.Failure, CPUExecutionProvider.MatMul(au, bu, null, null).Status);
    }

    [Fact]
    public void Int32Overflow_WrapsLikeOrt()
    {
        // ORT 1.29 wraps int32 MatMul (unlike integer reductions, which
        // saturate): [max] x [2] -> [-2].
        var a = DenseTensor<int>.OfValues(new int[,] { { 2147483647 } });
        var b = DenseTensor<int>.OfValues(new int[,] { { 2 } });
        var r = CPUExecutionProvider.MatMul(a, b, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        Assert.Equal(new int[] { -2 }, ((Tensor<int>)r.Outputs![0]).ToArray());
    }

    [Fact]
    public unsafe void KernelsComputeABAgainstReference()
    {
        var rnd = new Random(Seed);
        var a = Fill(N, N, rnd);
        var b = Fill(N, N, rnd);
        Assert.NotEqual(Sum(a), Sum(b));
        var zero = Tensor<float>.Zeros(N, N).ToDenseTensor();
        float expected = ReferenceSum(a, b, zero);

        var c1 = Tensor<float>.Zeros(N, N).ToDenseTensor();
        RunUnsafe((pa, pb, pc) => MathOps.mm(N, N, N, (float*)pa, (float*)pb, (float*)pc), a, b, c1);
        Agrees(Sum(c1), expected, "unsafe");

        var c2 = Tensor<float>.Zeros(N, N).ToDenseTensor();
        RunUnsafe((pa, pb, pc) => MathOps.mm_unsafe_vectorized(N, N, N, (float*)pa, (float*)pb, (float*)pc), a, b, c2);
        Agrees(Sum(c2), expected, "unsafe-vectorized");

        var c3 = Tensor<float>.Zeros(N, N).ToDenseTensor();
        RunUnsafe((pa, pb, pc) => MathOps.mm_unsafe_vectorized_intrinsics(N, N, N, (float*)pa, (float*)pb, (float*)pc), a, b, c3);
        Agrees(Sum(c3), expected, "intrinsics");
        var c4 = Tensor<float>.Zeros(N, N).ToDenseTensor();
        RunUnsafe((pa, pb, pc) => MathOps.mm_unsafe_vectorized_intrinsics_2x4tiled(N, N, N, (float*)pa, (float*)pb, (float*)pc), a, b, c4);
        Agrees(Sum(c4), expected, "intrinsics-2x4tiled");
    }

    [Fact]
    public unsafe void KernelsAccumulateOntoNonzeroDestination()
    {
        var rnd = new Random(Seed);
        var a = Fill(N, N, rnd);
        var b = Fill(N, N, rnd);
        var c1 = Fill(N, N, rnd);
        var c2 = Tensor<float>.Zeros(N, N).ToDenseTensor();
        c1.Buffer.Span.CopyTo(c2.Buffer.Span);
        float expected = ReferenceSum(a, b, c1);

        RunUnsafe((pa, pb, pc) => MathOps.mm_unsafe_vectorized_intrinsics(N, N, N, (float*)pa, (float*)pb, (float*)pc), a, b, c2);
        Agrees(Sum(c2), expected, "intrinsics-accumulate");
        var c3 = Tensor<float>.Zeros(N, N).ToDenseTensor();
        c1.Buffer.Span.CopyTo(c3.Buffer.Span);
        RunUnsafe((pa, pb, pc) => MathOps.mm_unsafe_vectorized_intrinsics_2x4tiled(N, N, N, (float*)pa, (float*)pb, (float*)pc), a, b, c3);
        Agrees(Sum(c3), expected, "intrinsics-2x4tiled-accumulate");
    }

    [Fact]
    public unsafe void TiledMatchesUnrolledBitwiseOnTails()
    {
        var rnd = new Random(Seed);
        BitwiseEqual(6, 24, 20, rnd);
        BitwiseEqual(4, 24, 40, rnd);
        BitwiseEqual(4, 24, 44, rnd);
        BitwiseEqual(8, 40, 588, rnd);
    }

    static unsafe void BitwiseEqual(int m, int n, int k, Random rnd)
    {
        var a = FillRect(m, n, rnd);
        var b = FillRect(n, k, rnd);
        var c1 = Tensor<float>.Zeros(m, k).ToDenseTensor();
        var c2 = Tensor<float>.Zeros(m, k).ToDenseTensor();
        RunUnsafe((pa, pb, pc) => MathOps.mm_unsafe_vectorized_intrinsics_2x4(m, n, k, (float*)pa, (float*)pb, (float*)pc), a, b, c1);
        RunUnsafe((pa, pb, pc) => MathOps.mm_unsafe_vectorized_intrinsics_2x4tiled(m, n, k, (float*)pa, (float*)pb, (float*)pc), a, b, c2);
        Assert.True(c1.Buffer.Span.SequenceEqual(c2.Buffer.Span),
            $"tiled diverges bitwise from unrolled on {m}x{n}x{k}.");
        var d1 = FillRect(m, k, rnd);
        var d2 = Tensor<float>.Zeros(m, k).ToDenseTensor();
        d1.Buffer.Span.CopyTo(d2.Buffer.Span);
        RunUnsafe((pa, pb, pc) => MathOps.mm_unsafe_vectorized_intrinsics_2x4(m, n, k, (float*)pa, (float*)pb, (float*)pc), a, b, d1);
        RunUnsafe((pa, pb, pc) => MathOps.mm_unsafe_vectorized_intrinsics_2x4tiled(m, n, k, (float*)pa, (float*)pb, (float*)pc), a, b, d2);
        Assert.True(d1.Buffer.Span.SequenceEqual(d2.Buffer.Span),
            $"tiled diverges bitwise from unrolled on nonzero destination {m}x{n}x{k}.");
    }

    static unsafe void RunPacked(Action<IntPtr, IntPtr, IntPtr, IntPtr> kernel, DenseTensor<float> a, DenseTensor<float> b, DenseTensor<float> p, DenseTensor<float> c)
    {
        using var pa = a.Buffer.Pin();
        using var pb = b.Buffer.Pin();
        using var pp = p.Buffer.Pin();
        using var pc = c.Buffer.Pin();
        kernel((IntPtr)pa.Pointer, (IntPtr)pb.Pointer, (IntPtr)pp.Pointer, (IntPtr)pc.Pointer);
    }

    [Fact]
    public unsafe void PackedMatchesTiledBitwise()
    {
        var rnd = new Random(Seed);
        PackedEqual(8, 24, 20, rnd);
        PackedEqual(8, 24, 44, rnd);
        PackedEqual(8, 24, 96, rnd);
        PackedEqual(64, 24, 44, rnd);
        PackedEqual(64, 48, 3100, rnd);
        PackedEqual(64, 64, 3136, rnd);
    }

    static unsafe void PackedEqual(int m, int n, int k, Random rnd)
    {
        var a = FillRect(m, n, rnd);
        var b = FillRect(n, k, rnd);
        var c1 = Tensor<float>.Zeros(m, k).ToDenseTensor();
        RunUnsafe((pa, pb, pc) => MathOps.mm_unsafe_vectorized_intrinsics_2x4tiled(m, n, k, (float*)pa, (float*)pb, (float*)pc), a, b, c1);
        var p = Tensor<float>.Zeros(n, k).ToDenseTensor();
        var c2 = Tensor<float>.Zeros(m, k).ToDenseTensor();
        RunPacked((pa, pb, pp, pc) => { MathOps.PackPanelsB(n, k, (float*)pb, (float*)pp); MathOps.mm_unsafe_vectorized_intrinsics_2x4packed(m, n, k, (float*)pa, (float*)pp, (float*)pc); }, a, b, p, c2);
        Assert.True(c1.Buffer.Span.SequenceEqual(c2.Buffer.Span),
            $"packed diverges bitwise from tiled on {m}x{n}x{k}.");
        var d1 = FillRect(m, k, rnd);
        var d2 = Tensor<float>.Zeros(m, k).ToDenseTensor();
        d1.Buffer.Span.CopyTo(d2.Buffer.Span);
        RunUnsafe((pa, pb, pc) => MathOps.mm_unsafe_vectorized_intrinsics_2x4tiled(m, n, k, (float*)pa, (float*)pb, (float*)pc), a, b, d1);
        var q = Tensor<float>.Zeros(n, k).ToDenseTensor();
        RunPacked((pa, pb, pp, pc) => { MathOps.PackPanelsB(n, k, (float*)pb, (float*)pp); MathOps.mm_unsafe_vectorized_intrinsics_2x4packed(m, n, k, (float*)pa, (float*)pp, (float*)pc); }, a, b, q, d2);
        Assert.True(d1.Buffer.Span.SequenceEqual(d2.Buffer.Span),
            $"packed diverges bitwise from tiled on nonzero destination {m}x{n}x{k}.");
    }

    [Fact]
    public void DispatchedMatMulAgreesWithReference()
    {
        var rnd = new Random(Seed);
        DispatchedAgrees(30, 48, 80, rnd);
        DispatchedAgrees(64, 48, 80, rnd);
        DispatchedAgrees(64, 48, 3100, rnd);
    }

    static void DispatchedAgrees(int m, int n, int k, Random rnd)
    {
        var a = FillRect(m, n, rnd);
        var b = FillRect(n, k, rnd);
        var zero = Tensor<float>.Zeros(m, k).ToDenseTensor();
        float expected = ReferenceRectSum(a, b, zero, m, n, k);
        var got = Tensor<float>.MatMul2D(a, b);
        Assert.Equal(m, got.Dimensions[0]);
        Assert.Equal(k, got.Dimensions[1]);
        Agrees(SumRect(got.ToDenseTensor()), expected, "dispatched-" + m + "x" + n + "x" + k);
    }

    [Fact]
    public unsafe void TiledKernelAgreesOnTailShapes()
    {
        var rnd = new Random(Seed);
        TailAgrees(6, 24, 20, rnd);
        TailAgrees(4, 24, 40, rnd);
        TailAgrees(4, 24, 44, rnd);
    }

    static DenseTensor<float> FillRect(int rows, int cols, Random rnd)
    {
        var t = Tensor<float>.Zeros(rows, cols).ToDenseTensor();
        for (int i = 0; i < t.Length; i++) t.SetValue(i, rnd.NextSingle());
        return t;
    }

    static float ReferenceRectSum(DenseTensor<float> a, DenseTensor<float> b, DenseTensor<float> c, int m, int n, int k)
    {
        float total = 0f;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < k; j++)
            {
                float acc = c.GetValue(i * k + j);
                for (int l = 0; l < n; l++) acc += a.GetValue(i * n + l) * b.GetValue(l * k + j);
                total += acc;
            }
        return total;
    }

    static float SumRect(DenseTensor<float> t)
    {
        float s = 0f;
        for (int i = 0; i < t.Length; i++) s += t.GetValue(i);
        return s;
    }

    static unsafe void TailAgrees(int m, int n, int k, Random rnd)
    {
        var a = FillRect(m, n, rnd);
        var b = FillRect(n, k, rnd);
        var zero = Tensor<float>.Zeros(m, k).ToDenseTensor();
        float expected = ReferenceRectSum(a, b, zero, m, n, k);

        var c = Tensor<float>.Zeros(m, k).ToDenseTensor();
        RunUnsafe((pa, pb, pc) => MathOps.mm_unsafe_vectorized_intrinsics_2x4tiled(m, n, k, (float*)pa, (float*)pb, (float*)pc), a, b, c);
        Agrees(SumRect(c), expected, "intrinsics-2x4tiled-tail-" + m + "x" + n + "x" + k);
    }

    [Fact]
    public void NanInput_Propagates()
    {
        // ORT 1.29: [[nan,1],[1,1]] @ I -> [[nan,nan],[1,1]]. Guards the
        // blocked/SIMD FMA kernels against NaN-dropping rewrites.
        var a = DenseTensor<float>.OfValues(new float[,] { { float.NaN, 1f }, { 1f, 1f } });
        var b = DenseTensor<float>.OfValues(new float[,] { { 1f, 0f }, { 0f, 1f } });
        var r = CPUExecutionProvider.MatMul(a, b, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = ((Tensor<float>)r.Outputs[0]).ToArray();
        Assert.True(float.IsNaN(y[0]));
        Assert.True(float.IsNaN(y[1]));
        Assert.Equal(1f, y[2]);
        Assert.Equal(1f, y[3]);
    }

    [Fact]
    public void NanInputDouble_Propagates()
    {
        // ORT 1.29 double: [[nan,1],[1,1]] @ I -> [[nan,nan],[1,1]],
        // mirroring the float guard above through the double kernel.
        var a = DenseTensor<double>.OfValues(new double[,] { { double.NaN, 1.0 }, { 1.0, 1.0 } });
        var b = DenseTensor<double>.OfValues(new double[,] { { 1.0, 0.0 }, { 0.0, 1.0 } });
        var r = CPUExecutionProvider.MatMul(a, b, null, null);
        Assert.Equal(OpStatus.Success, r.Status);
        var y = ((Tensor<double>)r.Outputs[0]).ToArray();
        Assert.True(double.IsNaN(y[0]));
        Assert.True(double.IsNaN(y[1]));
        Assert.Equal(1.0, y[2]);
        Assert.Equal(1.0, y[3]);
    }
}
