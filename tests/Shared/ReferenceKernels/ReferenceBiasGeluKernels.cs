using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using static Lokad.Onnx.MathOps;

namespace Lokad.Onnx.Tests.Support;

// Independent reference/experimental kernels shared by tests and benchmark tools.
// Relocated from the core without changing arithmetic or compilation attributes.
internal static class ReferenceBiasGeluKernels
{

    static float ScalarGelu(float v) => 0.5f * v * (1f + MathOps.Erf(v * 0.7071067811865476f));

    /// <summary>
    /// Fused bias-add plus exact GELU over spans: y[i] = gelu(x[i] + b[i % M])
    /// with the same per-element arithmetic as the two-node form (identical
    /// add, then the identical erf call and combine order). Rank-one bias of
    /// length M takes the vector path when M is a multiple of the vector
    /// width; anything else runs the scalar tail loop. Safe in place.
    /// </summary>
    /// <summary>
    /// BiasGelu pointer fast path: identical arithmetic and order to BiasGeluSpanFloat
    /// with raw vector pointers instead of a per-vector Slice plus Cast, removing
    /// bounds-check setup from the hot loop.
    /// </summary>
    internal static unsafe void BiasGeluSpanFloatPtr(ReadOnlySpan<float> xs, ReadOnlySpan<float> bias, Span<float> ys)
    {
        int w = Vector<float>.Count;
        var half = new Vector<float>(0.5f);
        var one = Vector<float>.One;
        var scale = new Vector<float>(0.7071067811865476f);
        int M = bias.Length;
        if (M <= 1 || M % w != 0 || xs.Length != ys.Length)
        {
            int soff = 0;
            for (int i = 0; i < xs.Length; i++)
            {
                ys[i] = ScalarGelu(xs[i] + bias[soff]);
                soff++;
                if (soff >= M) soff = 0;
            }
            return;
        }
        fixed (float* px = xs, py = ys, pb = bias)
        {
            var xvec = (Vector<float>*)px;
            var yvec = (Vector<float>*)py;
            var bvec = (Vector<float>*)pb;
            int nvec = xs.Length / w;
            int bvecs = M / w;
            int boff = 0;
            for (int i = 0; i < nvec; i++)
            {
                var tv = xvec[i] + bvec[boff];
                yvec[i] = half * tv * (one + ErfVector(scale * tv));
                if (++boff >= bvecs) boff = 0;
            }
            int tail = nvec * w;
            int toff = tail % M;
            for (int i = tail; i < xs.Length; i++)
            {
                ys[i] = ScalarGelu(xs[i] + bias[toff]);
                toff++;
                if (toff >= M) toff = 0;
            }
        }
    }

    /// <summary>
    /// E72 twin: two-way unrolled BiasGelu pointer loop. Same per-lane arithmetic
    /// and order as BiasGeluSpanFloatPtr, two independent vectors per iteration
    /// for instruction-level parallelism. Test-reachable only; no dispatch yet.
    /// </summary>
    internal static unsafe void BiasGeluSpanFloatPtr2x(ReadOnlySpan<float> xs, ReadOnlySpan<float> bias, Span<float> ys)
    {
        int w = Vector<float>.Count;
        var half = new Vector<float>(0.5f);
        var one = Vector<float>.One;
        var scale = new Vector<float>(0.7071067811865476f);
        int M = bias.Length;
        if (M <= 1 || M % w != 0 || xs.Length != ys.Length)
        {
            int soff = 0;
            for (int i = 0; i < xs.Length; i++)
            {
                ys[i] = ScalarGelu(xs[i] + bias[soff]);
                soff++;
                if (soff >= M) soff = 0;
            }
            return;
        }
        fixed (float* px = xs, py = ys, pb = bias)
        {
            var xvec = (Vector<float>*)px;
            var yvec = (Vector<float>*)py;
            var bvec = (Vector<float>*)pb;
            int nvec = xs.Length / w;
            int bvecs = M / w;
            int boff = 0;
            int pairs = nvec / 2;
            for (int i = 0; i < pairs; i++)
            {
                int b0 = boff;
                int b1 = b0 + 1;
                if (b1 >= bvecs) b1 = 0;
                var tv0 = xvec[2 * i] + bvec[b0];
                var tv1 = xvec[2 * i + 1] + bvec[b1];
                yvec[2 * i] = half * tv0 * (one + ErfVector(scale * tv0));
                yvec[2 * i + 1] = half * tv1 * (one + ErfVector(scale * tv1));
                boff = b1 + 1;
                if (boff >= bvecs) boff = 0;
            }
            for (int i = pairs * 2; i < nvec; i++)
            {
                var tv = xvec[i] + bvec[boff];
                yvec[i] = half * tv * (one + ErfVector(scale * tv));
                if (++boff >= bvecs) boff = 0;
            }
            int tail = nvec * w;
            int toff = tail % M;
            for (int i = tail; i < xs.Length; i++)
            {
                ys[i] = ScalarGelu(xs[i] + bias[toff]);
                toff++;
                if (toff >= M) toff = 0;
            }
        }
    }
    internal static void BiasGeluSpanFloat(ReadOnlySpan<float> xs, ReadOnlySpan<float> bias, Span<float> ys)
    {
        int w = Vector<float>.Count;
        var half = new Vector<float>(0.5f);
        var one = Vector<float>.One;
        var scale = new Vector<float>(0.7071067811865476f);
        int M = bias.Length;
        if (M > 1 && M % w == 0 && xs.Length == ys.Length)
        {
            var xvec = MemoryMarshal.Cast<float, Vector<float>>(xs);
            var yvec = MemoryMarshal.Cast<float, Vector<float>>(ys);
            int boff = 0;
            for (int i = 0; i < yvec.Length; i++)
            {
                var bv = MemoryMarshal.Cast<float, Vector<float>>(bias.Slice(boff, w))[0];
                var tv = xvec[i] + bv;
                yvec[i] = half * tv * (one + ErfVector(scale * tv));
                boff += w;
                if (boff >= M) boff -= M;
            }
            int tail = yvec.Length * w;
            int toff = tail % M;
            for (int i = tail; i < xs.Length; i++)
            {
                ys[i] = ScalarGelu(xs[i] + bias[toff]);
                toff++;
                if (toff >= M) toff = 0;
            }
            return;
        }
        int soff = 0;
        for (int i = 0; i < xs.Length; i++)
        {
            ys[i] = ScalarGelu(xs[i] + bias[soff]);
            soff++;
            if (soff >= M) soff = 0;
        }
    }
}
