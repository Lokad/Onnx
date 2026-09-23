using System;
using System.Numerics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Diagnostics;

using System.Buffers;
using static Lokad.Onnx.MathOps;

namespace Lokad.Onnx;

public abstract partial class Tensor<T> where T : unmanaged
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static unsafe void RunIsolatedShortWideKernel(int m, int n, int k, float* x, float* y, float* output, TensorExecutionOptions options)
    {
        // Isolate wide projections at every packed row count; shared kernels stay unchanged.
        if (m >= 48 && n >= 1024 && k >= 1024
            && (long)n * k <= 67108864
            && options.UseSimd && options.UseIntrinsics && Fma.IsSupported)
        {
            RunIsolatedShortWidePackedRows(m, n, k, x, y, output, options);
            return;
        }
        RunFloatMatMulKernel(m, n, k, x, y, output, options);
    }

    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
    static unsafe void RunIsolatedShortWidePackedRows(int m, int n, int k, float* x, float* y, float* output, TensorExecutionOptions options)
    {
        bool threeRows = m % 3 == 0;
        int rows = threeRows ? m : m - m % 2;
        float[] packed = RentScratch<float>(n * k, options);
        try
        {
            fixed (float* pp = packed)
            {
                ShortWidePackPanelsB(n, k, y, pp);
                if (!AblationSwitches.EnablePackedAvx512Dynamic
                    || !TryPackedAvx512Rows(rows, n, k, x, pp, output))
                {
                    if (threeRows)
                        ShortWideMultiply3Rows(rows, n, k, x, pp, output);
                    else
                        ShortWideMultiply2Rows(rows, n, k, x, pp, output);
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(packed);
        }
        // The last odd row still reads the original operand, after scratch returns.
        if (rows != m)
            ShortWideMultiplyRemainder(1, n, k, x + rows * n, y, output + rows * k);
    }
}

public partial class MathOps
{
    // These copies serve only the guarded short-wide projection route.
    // Shared kernels retain their original bodies and compilation flags.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal unsafe static void ShortWidePackPanelsB(int N,
                          int K,
                          float* B,
                          float* P)
    {
        int blocked = K - (K % (4 * Vector256<float>.Count));
        for (int kb = 0; kb < blocked; kb += 4 * Vector256<float>.Count)
        {
            float* dst = P + (kb / (4 * Vector256<float>.Count)) * N * (4 * Vector256<float>.Count);
            for (int j = 0; j < N; j++)
            {
                float* src = B + j * K + kb;
                float* d = dst + j * (4 * Vector256<float>.Count);
                new ReadOnlySpan<float>(src, 32).CopyTo(new Span<float>(d, 32));
            }
        }
        int rem = K - blocked;
        float* tail = P + (blocked / (4 * Vector256<float>.Count)) * N * (4 * Vector256<float>.Count);
        for (int j = 0; j < N; j++)
        {
            float* src = B + j * K + blocked;
            float* dst = tail + j * rem;
            for (int k = 0; k < rem; k++) dst[k] = src[k];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal unsafe static void ShortWideMultiply2Rows(int M,
                          int N,
                          int K,
                          float* A,
                          float* P,
                          float* C)
    {
        if (M % 2 != 0)
            throw new ArgumentException(nameof(M));

        int panelWidth = 4 * Vector256<float>.Count;
        int blocked = K - (K % panelWidth);
        int tiles = blocked / panelWidth;

        // Kb panels lead so each panel is fetched once and stays L1/L2 resident
        // across every row group, exactly like the reference nest.
        for (int tb = 0; tb < tiles; tb++)
        {
            int kb = tb * panelWidth;
            float* panel = P + tb * N * panelWidth;
            for (int i = 0; i < M; i += 2)
            {
                float* a0 = A + i * N;
                float* a1 = a0 + N;
                float* b = panel;
                var Cpv1 = (Vector256<float>*)(C + i * K + kb);
                var Cpv2 = (Vector256<float>*)(C + (i + 1) * K + kb);
                Vector256<float> c00 = Cpv1[0];
                Vector256<float> c01 = Cpv1[1];
                Vector256<float> c02 = Cpv1[2];
                Vector256<float> c03 = Cpv1[3];
                Vector256<float> c10 = Cpv2[0];
                Vector256<float> c11 = Cpv2[1];
                Vector256<float> c12 = Cpv2[2];
                Vector256<float> c13 = Cpv2[3];
                for (int j = 0; j < N; ++j)
                {
                    var Bpv = (Vector256<float>*)b;
                    Vector256<float> bv0 = Bpv[0];
                    Vector256<float> bv1 = Bpv[1];
                    Vector256<float> bv2 = Bpv[2];
                    Vector256<float> bv3 = Bpv[3];
                    var av1 = Vector256.Create(*a0); a0 += 1;
                    var av2 = Vector256.Create(*a1); a1 += 1;
                    c00 = Fma.MultiplyAdd(bv0, av1, c00);
                    c10 = Fma.MultiplyAdd(bv0, av2, c10);
                    c01 = Fma.MultiplyAdd(bv1, av1, c01);
                    c11 = Fma.MultiplyAdd(bv1, av2, c11);
                    c02 = Fma.MultiplyAdd(bv2, av1, c02);
                    c12 = Fma.MultiplyAdd(bv2, av2, c12);
                    c03 = Fma.MultiplyAdd(bv3, av1, c03);
                    c13 = Fma.MultiplyAdd(bv3, av2, c13);
                    b += panelWidth;
                }
                Cpv1[0] = c00;
                Cpv1[1] = c01;
                Cpv1[2] = c02;
                Cpv1[3] = c03;
                Cpv2[0] = c10;
                Cpv2[1] = c11;
                Cpv2[2] = c12;
                Cpv2[3] = c13;
            }
        }

        int rem = K - blocked;
        if (rem > 0)
        {
            float* T = P + tiles * N * (4 * Vector256<float>.Count);
            int rv = rem / Vector256<float>.Count;
            for (int tt = 0; tt < rv; tt++)
            {
                for (int i = 0; i < M; i += 2)
                {
                    var Ap1 = A + i * N;
                    var Ap2 = Ap1 + N;
                    var rC1 = (Vector256<float>*)(C + i * K + blocked);
                    var rC2 = (Vector256<float>*)(C + (i + 1) * K + blocked);
                    Vector256<float> c1 = rC1[tt];
                    Vector256<float> c2 = rC2[tt];
                    for (int j = 0; j < N; ++j)
                    {
                        var Bpv = (Vector256<float>*)(T + j * rem + tt * Vector256<float>.Count);
                        c1 = Fma.MultiplyAdd(Bpv[0], Vector256.Create(Ap1[j]), c1);
                        c2 = Fma.MultiplyAdd(Bpv[0], Vector256.Create(Ap2[j]), c2);
                    }
                    rC1[tt] = c1;
                    rC2[tt] = c2;
                }
            }
            int vcols = rv * Vector256<float>.Count;
            // Narrow tail (fewer than 8 columns) accumulates in registers
            // instead of read-modify-writing C on every reduction step: the
            // per-element chain keeps the exact j-ascending mul-then-add order
            // of the loop it replaces, so results agree bit-wise while each C
            // row is loaded once and stored once.
            int tail = rem - vcols;
            if (tail > 0)
            for (int i = 0; i < M; i += 2)
            {
                var Ap1 = A + i * N;
                var Ap2 = Ap1 + N;
                var Cp1 = C + i * K + blocked + vcols;
                var Cp2 = Cp1 + K;
                // E94: masked-vector narrow tail. Same j-ascending mul-then-add
                // order per element as the scalar switch it replaces (MULPS then
                // ADDPS, never FMA), so results agree bit-wise. Integer maskmov
                // moves the bits (fault-suppressed, discarded by masked store);
                // arithmetic stays float. Scalar fallback preserves order.
                if (Avx2.IsSupported)
                {
                    Vector256<int> tmask = Vector256.Create(tail > 0 ? -1 : 0, tail > 1 ? -1 : 0, tail > 2 ? -1 : 0, tail > 3 ? -1 : 0, tail > 4 ? -1 : 0, tail > 5 ? -1 : 0, tail > 6 ? -1 : 0, 0);
                    Vector256<int> ci1 = Avx2.MaskLoad((int*)Cp1, tmask);
                    Vector256<int> ci2 = Avx2.MaskLoad((int*)Cp2, tmask);
                    Vector256<float> c1 = Unsafe.As<Vector256<int>, Vector256<float>>(ref ci1);
                    Vector256<float> c2 = Unsafe.As<Vector256<int>, Vector256<float>>(ref ci2);
                    for (int j = 0; j < N; ++j)
                    {
                        Vector256<int> bi = Avx2.MaskLoad((int*)(T + j * rem + vcols), tmask);
                        Vector256<float> bv = Unsafe.As<Vector256<int>, Vector256<float>>(ref bi);
                        var av1 = Vector256.Create(Ap1[j]);
                        var av2 = Vector256.Create(Ap2[j]);
                        c1 = c1 + av1 * bv;
                        c2 = c2 + av2 * bv;
                    }
                    Vector256<int> co1 = Unsafe.As<Vector256<float>, Vector256<int>>(ref c1);
                    Vector256<int> co2 = Unsafe.As<Vector256<float>, Vector256<int>>(ref c2);
                    Avx2.MaskStore((int*)Cp1, tmask, co1);
                    Avx2.MaskStore((int*)Cp2, tmask, co2);
                }
                else
                {
                    for (int j = 0; j < N; ++j)
                    {
                        float a1 = Ap1[j];
                        float a2 = Ap2[j];
                        var t = T + j * rem + vcols;
                        for (int k = 0; k < tail; k++) { Cp1[k] += a1 * t[k]; Cp2[k] += a2 * t[k]; }
                    }
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal unsafe static void ShortWideMultiply3Rows(int M,
                          int N,
                          int K,
                          float* A,
                          float* P,
                          float* C)
    {
        if (M % 3 != 0)
            throw new ArgumentException(nameof(M));

        int blocked = K - (K % (4 * Vector256<float>.Count));
        int tiles = blocked / (4 * Vector256<float>.Count);

        // Kb panels lead so each panel is fetched once and stays L1/L2 resident
        // across every row group; per-element FMA order matches the 2-row nest
        // exactly, so results agree bit-wise with the other packed kernels.
        for (int tb = 0; tb < tiles; tb++)
        {
            int kb = tb * (4 * Vector256<float>.Count);
            float* panel = P + tb * N * (4 * Vector256<float>.Count);
            for (int i = 0; i < M; i += 3)
            {
                var Ap1 = A + i * N;
                var Ap2 = Ap1 + N;
                var Ap3 = Ap2 + N;
                var Cpv1 = (Vector256<float>*)(C + i * K + kb);
                var Cpv2 = (Vector256<float>*)(C + (i + 1) * K + kb);
                var Cpv3 = (Vector256<float>*)(C + (i + 2) * K + kb);
                Vector256<float> c00 = Cpv1[0];
                Vector256<float> c01 = Cpv1[1];
                Vector256<float> c02 = Cpv1[2];
                Vector256<float> c03 = Cpv1[3];
                Vector256<float> c10 = Cpv2[0];
                Vector256<float> c11 = Cpv2[1];
                Vector256<float> c12 = Cpv2[2];
                Vector256<float> c13 = Cpv2[3];
                Vector256<float> c20 = Cpv3[0];
                Vector256<float> c21 = Cpv3[1];
                Vector256<float> c22 = Cpv3[2];
                Vector256<float> c23 = Cpv3[3];
                for (int j = 0; j < N; ++j)
                {
                    var av1 = Vector256.Create(Ap1[j]);
                    var av2 = Vector256.Create(Ap2[j]);
                    var av3 = Vector256.Create(Ap3[j]);
                    var Bpv = (Vector256<float>*)(panel + j * (4 * Vector256<float>.Count));
                    c00 = Fma.MultiplyAdd(Bpv[0], av1, c00);
                    c10 = Fma.MultiplyAdd(Bpv[0], av2, c10);
                    c20 = Fma.MultiplyAdd(Bpv[0], av3, c20);
                    c01 = Fma.MultiplyAdd(Bpv[1], av1, c01);
                    c11 = Fma.MultiplyAdd(Bpv[1], av2, c11);
                    c21 = Fma.MultiplyAdd(Bpv[1], av3, c21);
                    c02 = Fma.MultiplyAdd(Bpv[2], av1, c02);
                    c12 = Fma.MultiplyAdd(Bpv[2], av2, c12);
                    c22 = Fma.MultiplyAdd(Bpv[2], av3, c22);
                    c03 = Fma.MultiplyAdd(Bpv[3], av1, c03);
                    c13 = Fma.MultiplyAdd(Bpv[3], av2, c13);
                    c23 = Fma.MultiplyAdd(Bpv[3], av3, c23);
                }
                Cpv1[0] = c00;
                Cpv1[1] = c01;
                Cpv1[2] = c02;
                Cpv1[3] = c03;
                Cpv2[0] = c10;
                Cpv2[1] = c11;
                Cpv2[2] = c12;
                Cpv2[3] = c13;
                Cpv3[0] = c20;
                Cpv3[1] = c21;
                Cpv3[2] = c22;
                Cpv3[3] = c23;
            }
        }
        int rem = K - blocked;
        if (rem > 0)
        {
            float* T = P + tiles * N * (4 * Vector256<float>.Count);
            int rv = rem / Vector256<float>.Count;
            for (int tt = 0; tt < rv; tt++)
            {
                for (int i = 0; i < M; i += 3)
                {
                    var Ap1 = A + i * N;
                    var Ap2 = Ap1 + N;
                    var Ap3 = Ap2 + N;
                    var rC1 = (Vector256<float>*)(C + i * K + blocked);
                    var rC2 = (Vector256<float>*)(C + (i + 1) * K + blocked);
                    var rC3 = (Vector256<float>*)(C + (i + 2) * K + blocked);
                    Vector256<float> c1 = rC1[tt];
                    Vector256<float> c2 = rC2[tt];
                    Vector256<float> c3 = rC3[tt];
                    for (int j = 0; j < N; ++j)
                    {
                        var Bpv = (Vector256<float>*)(T + j * rem + tt * Vector256<float>.Count);
                        c1 = Fma.MultiplyAdd(Bpv[0], Vector256.Create(Ap1[j]), c1);
                        c2 = Fma.MultiplyAdd(Bpv[0], Vector256.Create(Ap2[j]), c2);
                        c3 = Fma.MultiplyAdd(Bpv[0], Vector256.Create(Ap3[j]), c3);
                    }
                    rC1[tt] = c1;
                    rC2[tt] = c2;
                    rC3[tt] = c3;
                }
            }
            int vcols = rv * Vector256<float>.Count;
            // Narrow tail (fewer than 8 columns) accumulates in registers
            // instead of read-modify-writing C on every reduction step: the
            // per-element chain keeps the exact j-ascending mul-then-add order
            // of the loop it replaces, so results agree bit-wise while each C
            // row is loaded once and stored once.
            int tail = rem - vcols;
            if (tail > 0)
            for (int i = 0; i < M; i += 3)
            {
                var Ap1 = A + i * N;
                var Ap2 = Ap1 + N;
                var Ap3 = Ap2 + N;
                var Cp1 = C + i * K + blocked + vcols;
                var Cp2 = Cp1 + K;
                var Cp3 = Cp2 + K;
                // E94: masked-vector narrow tail. Same j-ascending mul-then-add
                // order per element as the scalar switch it replaces (MULPS then
                // ADDPS, never FMA), so results agree bit-wise. Integer maskmov
                // moves the bits (fault-suppressed, discarded by masked store);
                // arithmetic stays float. Scalar fallback preserves order.
                if (Avx2.IsSupported)
                {
                    Vector256<int> tmask = Vector256.Create(tail > 0 ? -1 : 0, tail > 1 ? -1 : 0, tail > 2 ? -1 : 0, tail > 3 ? -1 : 0, tail > 4 ? -1 : 0, tail > 5 ? -1 : 0, tail > 6 ? -1 : 0, 0);
                    Vector256<int> ci1 = Avx2.MaskLoad((int*)Cp1, tmask);
                    Vector256<int> ci2 = Avx2.MaskLoad((int*)Cp2, tmask);
                    Vector256<int> ci3 = Avx2.MaskLoad((int*)Cp3, tmask);
                    Vector256<float> c1 = Unsafe.As<Vector256<int>, Vector256<float>>(ref ci1);
                    Vector256<float> c2 = Unsafe.As<Vector256<int>, Vector256<float>>(ref ci2);
                    Vector256<float> c3 = Unsafe.As<Vector256<int>, Vector256<float>>(ref ci3);
                    for (int j = 0; j < N; ++j)
                    {
                        Vector256<int> bi = Avx2.MaskLoad((int*)(T + j * rem + vcols), tmask);
                        Vector256<float> bv = Unsafe.As<Vector256<int>, Vector256<float>>(ref bi);
                        var av1 = Vector256.Create(Ap1[j]);
                        var av2 = Vector256.Create(Ap2[j]);
                        var av3 = Vector256.Create(Ap3[j]);
                        c1 = c1 + av1 * bv;
                        c2 = c2 + av2 * bv;
                        c3 = c3 + av3 * bv;
                    }
                    Vector256<int> co1 = Unsafe.As<Vector256<float>, Vector256<int>>(ref c1);
                    Vector256<int> co2 = Unsafe.As<Vector256<float>, Vector256<int>>(ref c2);
                    Vector256<int> co3 = Unsafe.As<Vector256<float>, Vector256<int>>(ref c3);
                    Avx2.MaskStore((int*)Cp1, tmask, co1);
                    Avx2.MaskStore((int*)Cp2, tmask, co2);
                    Avx2.MaskStore((int*)Cp3, tmask, co3);
                }
                else
                {
                    for (int j = 0; j < N; ++j)
                    {
                        float a1 = Ap1[j];
                        float a2 = Ap2[j];
                        float a3 = Ap3[j];
                        var t = T + j * rem + vcols;
                        for (int k = 0; k < tail; k++) { Cp1[k] += a1 * t[k]; Cp2[k] += a2 * t[k]; Cp3[k] += a3 * t[k]; }
                    }
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal unsafe static void ShortWideMultiplyRemainder(int M,
                          int N,
                          int K,
                          float* A,
                          float* B,
                          float* C)
    {

        for (int i = 0; i < M; i++)
        {
            var Ap = A + i * N;
            var Cp = C + i * K;
            for (int j = 0; j < N; ++j)
            {
                var a = Ap[j];
                var Bp = B + j * K;
                var av = Vector256.Create(a);
                int ceiling = (K / Vector256<float>.Count) * Vector256<float>.Count;
                var Bkv = MemoryMarshal.Cast<float, Vector256<float>>(new Span<float>(Bp, K));
                var Ckv = MemoryMarshal.Cast<float, Vector256<float>>(new Span<float>(Cp, K));
                for (int k = 0; k < Bkv.Length; k++)
                {
                    Ckv[k] = Fma.MultiplyAdd(Bkv[k], av, Ckv[k]);
                }
                for (int k = ceiling; k < K; k++)
                {
                    Cp[k] += a * Bp[k];
                }
            }
        }
    }

}
