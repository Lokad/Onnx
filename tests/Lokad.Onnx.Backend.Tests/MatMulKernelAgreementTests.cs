namespace Lokad.Onnx.Backend.Tests;

/// <summary>
/// Guards the T02 rule: the raw MatMul kernels compared by the benchmarks
/// must all compute A-times-B (never A-times-A) and agree with an independent
/// reference within float-reorder tolerance, including accumulation onto a
/// nonzero destination. Data is deterministically seeded so failures are
/// reproducible.
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
    public unsafe void KernelsComputeABAgainstReference()
    {
        var rnd = new Random(Seed);
        var a = Fill(N, N, rnd);
        var b = Fill(N, N, rnd);
        Assert.NotEqual(Sum(a), Sum(b));
        var zero = Tensor<float>.Zeros(N, N).ToDenseTensor();
        float expected = ReferenceSum(a, b, zero);

        var c1 = Tensor<float>.Zeros(N, N).ToDenseTensor();
        MathOps.mm_managed(N, N, N, a.Buffer, b.Buffer, c1.Buffer);
        Agrees(Sum(c1), expected, "managed");

        var c2 = Tensor<float>.Zeros(N, N).ToDenseTensor();
        MathOps.mm_vectorized(N, N, N, a.Buffer, b.Buffer, c2.Buffer);
        Agrees(Sum(c2), expected, "vectorized");

        var c3 = Tensor<float>.Zeros(N, N).ToDenseTensor();
        RunUnsafe((pa, pb, pc) => MathOps.mm(N, N, N, (float*)pa, (float*)pb, (float*)pc), a, b, c3);
        Agrees(Sum(c3), expected, "unsafe");

        var c4 = Tensor<float>.Zeros(N, N).ToDenseTensor();
        RunUnsafe((pa, pb, pc) => MathOps.mm_unsafe_vectorized(N, N, N, (float*)pa, (float*)pb, (float*)pc), a, b, c4);
        Agrees(Sum(c4), expected, "unsafe-vectorized");

        var c5 = Tensor<float>.Zeros(N, N).ToDenseTensor();
        RunUnsafe((pa, pb, pc) => MathOps.mm_unsafe_vectorized_intrinsics(N, N, N, (float*)pa, (float*)pb, (float*)pc), a, b, c5);
        Agrees(Sum(c5), expected, "intrinsics");
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

        MathOps.mm_managed(N, N, N, a.Buffer, b.Buffer, c1.Buffer);
        Agrees(Sum(c1), expected, "managed-accumulate");

        RunUnsafe((pa, pb, pc) => MathOps.mm_unsafe_vectorized_intrinsics(N, N, N, (float*)pa, (float*)pb, (float*)pc), a, b, c2);
        Agrees(Sum(c2), expected, "intrinsics-accumulate");
    }
}
