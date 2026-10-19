using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using static Lokad.Onnx.MathOps;

namespace Lokad.Onnx.Tests.Support;

// Independent reference/experimental kernels shared by tests and benchmark tools.
// Relocated from the core without changing arithmetic or compilation attributes.
internal static class ReferenceExpKernels
{
    // Exact lane arithmetic of ExpVector, widened without reassociating its polynomial.
    // Kept separate from the Estrin exponential, whose evaluation order differs.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector512<float> ExpVectorHorner512(Vector512<float> v)
    {
        var isFinite = Vector512.Equals(v, v);
        var x = Vector512.Min(Vector512.Max(v, Vector512.Create(-88.722839f)), Vector512.Create(88.722839f));
        var scaled = x * Vector512.Create(1.44269504088896341f);
        var shifted = Vector512.ConditionalSelect(Vector512.GreaterThanOrEqual(scaled, Vector512<float>.Zero),
            scaled + Vector512.Create(0.5f), scaled - Vector512.Create(0.5f));
        var n = Vector512.ConvertToInt32(shifted);
        var clamped = Vector512.Min(Vector512.Max(n, Vector512.Create(-126)), Vector512.Create(127));
        var nf = Vector512.ConvertToSingle(clamped);
        var r = Vector512.FusedMultiplyAdd(nf, Vector512.Create(-0.693359375f), x);
        r = Vector512.FusedMultiplyAdd(nf, Vector512.Create(2.12194440e-4f), r);
        var p = Vector512.Create(1f / 5040f);
        p = Vector512.FusedMultiplyAdd(p, r, Vector512.Create(1f / 720f));
        p = Vector512.FusedMultiplyAdd(p, r, Vector512.Create(1f / 120f));
        p = Vector512.FusedMultiplyAdd(p, r, Vector512.Create(1f / 24f));
        p = Vector512.FusedMultiplyAdd(p, r, Vector512.Create(1f / 6f));
        p = Vector512.FusedMultiplyAdd(p, r, Vector512.Create(0.5f));
        p = Vector512.FusedMultiplyAdd(p, r, Vector512<float>.One);
        p = Vector512.FusedMultiplyAdd(p, r, Vector512<float>.One);
        var scale = Vector512.AsSingle(Vector512.ShiftLeft(clamped + Vector512.Create(127), 23));
        var y = p * scale;
        y = Vector512.ConditionalSelect(Vector512.GreaterThan(v, Vector512.Create(88.722839f)),
            Vector512.Create(float.PositiveInfinity), y);
        y = Vector512.ConditionalSelect(Vector512.LessThan(v, Vector512.Create(-88.722839f)), Vector512<float>.Zero, y);
        return Vector512.ConditionalSelect(isFinite, y, Vector512.Create(float.NaN));
    }

    /// <summary>Span driver for the 512-bit exponential probe: 16-wide vector core with a
    /// scalar MathF.Exp tail (same tail rule as the span-softmax scalar tails).</summary>
    internal static unsafe void ExpSpan512(ReadOnlySpan<float> xs, Span<float> ys)
    {
        int w = Vector512<float>.Count;
        int nvec = xs.Length / w;
        fixed (float* px = xs, py = ys)
        {
            var xv = (Vector512<float>*)px;
            var yv = (Vector512<float>*)py;
            for (int i = 0; i < nvec; i++) yv[i] = ExpVector512(xv[i]);
            for (int i = nvec * w; i < xs.Length; i++) ys[i] = MathF.Exp(xs[i]);
        }
    }
}
