// Generated isolated zero-seeded three-row consumer. Full reduction first,
// optional bias second, stores directly to the supplied final row stride.
using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx;

internal static class ConvDirectOutput
{
    public unsafe static void Multiply(int M,
                          int N,
                          int K,
                          float* A,
                          float* P,
                          float* C, int outputStride, float* Bias, bool hasBias)
    {
        if (K == 2)
        {
            if (Avx2.IsSupported)
            {
                MultiplyTwoColumns(M, N, A, P, C, outputStride, Bias, hasBias);
                return;
            }
        }

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
                var Cpv1 = (Vector256<float>*)(C + i * outputStride + kb);
                var Cpv2 = (Vector256<float>*)(C + (i + 1) * outputStride + kb);
                var Cpv3 = (Vector256<float>*)(C + (i + 2) * outputStride + kb);
                Vector256<float> c00 = Vector256<float>.Zero;
                Vector256<float> c01 = Vector256<float>.Zero;
                Vector256<float> c02 = Vector256<float>.Zero;
                Vector256<float> c03 = Vector256<float>.Zero;
                Vector256<float> c10 = Vector256<float>.Zero;
                Vector256<float> c11 = Vector256<float>.Zero;
                Vector256<float> c12 = Vector256<float>.Zero;
                Vector256<float> c13 = Vector256<float>.Zero;
                Vector256<float> c20 = Vector256<float>.Zero;
                Vector256<float> c21 = Vector256<float>.Zero;
                Vector256<float> c22 = Vector256<float>.Zero;
                Vector256<float> c23 = Vector256<float>.Zero;
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
                if (hasBias)
                {
                    var bias = Vector256.Create(Bias[i]);
                    c00 = AddBiasVector(c00, bias);
                    c01 = AddBiasVector(c01, bias);
                    c02 = AddBiasVector(c02, bias);
                    c03 = AddBiasVector(c03, bias);
                    bias = Vector256.Create(Bias[i + 1]);
                    c10 = AddBiasVector(c10, bias);
                    c11 = AddBiasVector(c11, bias);
                    c12 = AddBiasVector(c12, bias);
                    c13 = AddBiasVector(c13, bias);
                    bias = Vector256.Create(Bias[i + 2]);
                    c20 = AddBiasVector(c20, bias);
                    c21 = AddBiasVector(c21, bias);
                    c22 = AddBiasVector(c22, bias);
                    c23 = AddBiasVector(c23, bias);
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
                    var rC1 = (Vector256<float>*)(C + i * outputStride + blocked);
                    var rC2 = (Vector256<float>*)(C + (i + 1) * outputStride + blocked);
                    var rC3 = (Vector256<float>*)(C + (i + 2) * outputStride + blocked);
                    Vector256<float> c1 = Vector256<float>.Zero;
                    Vector256<float> c2 = Vector256<float>.Zero;
                    Vector256<float> c3 = Vector256<float>.Zero;
                    for (int j = 0; j < N; ++j)
                    {
                        var Bpv = (Vector256<float>*)(T + j * rem + tt * Vector256<float>.Count);
                        c1 = Fma.MultiplyAdd(Bpv[0], Vector256.Create(Ap1[j]), c1);
                        c2 = Fma.MultiplyAdd(Bpv[0], Vector256.Create(Ap2[j]), c2);
                        c3 = Fma.MultiplyAdd(Bpv[0], Vector256.Create(Ap3[j]), c3);
                    }
                    if (hasBias)
                    {
                        c1 = AddBiasVector(c1, Vector256.Create(Bias[i]));
                        c2 = AddBiasVector(c2, Vector256.Create(Bias[i + 1]));
                        c3 = AddBiasVector(c3, Vector256.Create(Bias[i + 2]));
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
                var Cp1 = C + i * outputStride + blocked + vcols;
                var Cp2 = Cp1 + outputStride;
                var Cp3 = Cp2 + outputStride;
                // E94: masked-vector narrow tail. Same j-ascending mul-then-add
                // order per element as the scalar switch it replaces (MULPS then
                // ADDPS, never FMA), so results agree bit-wise. Integer maskmov
                // moves the bits (fault-suppressed, discarded by masked store);
                // arithmetic stays float. Scalar fallback preserves order.
                if (Avx2.IsSupported)
                {
                    Vector256<int> tmask = Vector256.Create(tail > 0 ? -1 : 0, tail > 1 ? -1 : 0, tail > 2 ? -1 : 0, tail > 3 ? -1 : 0, tail > 4 ? -1 : 0, tail > 5 ? -1 : 0, tail > 6 ? -1 : 0, 0);
                    Vector256<int> ci1 = Vector256<int>.Zero;
                    Vector256<int> ci2 = Vector256<int>.Zero;
                    Vector256<int> ci3 = Vector256<int>.Zero;
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
                    if (hasBias)
                    {
                        c1 = AddBiasVector(c1, Vector256.Create(Bias[i]));
                        c2 = AddBiasVector(c2, Vector256.Create(Bias[i + 1]));
                        c3 = AddBiasVector(c3, Vector256.Create(Bias[i + 2]));
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
                    for (int k = 0; k < tail; k++) { Cp1[k] = 0f; Cp2[k] = 0f; Cp3[k] = 0f; }
                    for (int j = 0; j < N; ++j)
                    {
                        float a1 = Ap1[j];
                        float a2 = Ap2[j];
                        float a3 = Ap3[j];
                        var t = T + j * rem + vcols;
                        for (int k = 0; k < tail; k++) { Cp1[k] += a1 * t[k]; Cp2[k] += a2 * t[k]; Cp3[k] += a3 * t[k]; }
                    }
                    if (hasBias)
                        for (int k = 0; k < tail; k++)
                        {
                            // Match the existing scalar bias/copy epilogue when
                            // both operands are NaN: the bias payload wins.
                            Cp1[k] = AddBiasScalar(Cp1[k], Bias[i]);
                            Cp2[k] = AddBiasScalar(Cp2[k], Bias[i + 1]);
                            Cp3[k] = AddBiasScalar(Cp3[k], Bias[i + 2]);
                        }
                }
            }
        }
    }
    // The original scalar epilogue selects the bias NaN when both operands
    // are NaN. SIMD register allocation may otherwise reverse that choice.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Vector256<float> AddBiasVector(Vector256<float> value, Vector256<float> bias)
        => float.IsNaN(bias.GetElement(0)) ? Avx.Add(bias, Vector256<float>.Zero) : Avx.Add(value, bias);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static float AddBiasScalar(float value, float bias)
    {
        var b = Vector128.CreateScalar(bias);
        return float.IsNaN(bias) ? Sse.AddScalar(b, Vector128<float>.Zero).ToScalar()
            : Sse.AddScalar(Vector128.CreateScalar(value), b).ToScalar();
    }

    // Keep generic panel/output state out of this long two-column reduction.
    [MethodImpl(MethodImplOptions.NoInlining)]
    static unsafe void MultiplyTwoColumns(int M, int N, float* A, float* P,
        float* C, int outputStride, float* Bias, bool hasBias)
    {
        var mask = Vector256.Create(-1, -1, 0, 0, 0, 0, 0, 0);
        for (int i = 0; i < M; i += 3)
        {
            float* a1 = A + i * N;
            float* a2 = a1 + N;
            float* a3 = a2 + N;
            float* bp = P;
            var c1 = Vector256<float>.Zero;
            var c2 = Vector256<float>.Zero;
            var c3 = Vector256<float>.Zero;
            for (int j = 0; j < N; ++j)
            {
                var bv = Avx2.MaskLoad((int*)bp, mask).AsSingle();
                var av1 = Vector256.Create(*a1);
                var av2 = Vector256.Create(*a2);
                var av3 = Vector256.Create(*a3);
                c1 = c1 + av1 * bv;
                c2 = c2 + av2 * bv;
                c3 = c3 + av3 * bv;
                ++a1; ++a2; ++a3; bp += 2;
            }
            if (hasBias)
            {
                c1 = AddBiasVector(c1, Vector256.Create(Bias[i]));
                c2 = AddBiasVector(c2, Vector256.Create(Bias[i + 1]));
                c3 = AddBiasVector(c3, Vector256.Create(Bias[i + 2]));
            }
            float* cp = C + i * outputStride;
            Avx2.MaskStore((int*)cp, mask, c1.AsInt32());
            Avx2.MaskStore((int*)(cp + outputStride), mask, c2.AsInt32());
            Avx2.MaskStore((int*)(cp + 2 * outputStride), mask, c3.AsInt32());
        }
    }
}
