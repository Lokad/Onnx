using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx;

public abstract partial class Tensor<T>
{
    // Four independent erf streams preserve the exact per-lane arithmetic of
    // ErfVectorInline. Interleaving exposes FMA concurrency without helper calls.
    // The target JIT uses the additional AVX-512 vector registers for this body.
    internal static bool TryBiasGeluSpanFloatInterleaved(ReadOnlySpan<float> xs, ReadOnlySpan<float> bias, Span<float> ys)
    {
        if (!Avx512F.IsSupported || !Fma.IsSupported || Vector<float>.Count != 8
            || bias.Length <= 1 || bias.Length % 32 != 0
            || xs.Length != ys.Length || xs.Length % bias.Length != 0)
            return false;
        BiasGeluSpanFloatInterleaved(xs, bias, ys);
        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.NoInlining)]
    internal static unsafe void BiasGeluSpanFloatInterleaved(ReadOnlySpan<float> xs, ReadOnlySpan<float> bias, Span<float> ys)
    {
        int w = Vector<float>.Count, group = w * 4, m = bias.Length;
        if (m <= 1 || m % group != 0 || xs.Length != ys.Length || xs.Length % m != 0) { BiasGeluSpanFloatInline(xs, bias, ys); return; }
        var half = new Vector<float>(0.5f); var one = new Vector<float>(1f); var scale = new Vector<float>(0.7071067811865476f);
        fixed (float* px = xs, pb = bias, py = ys)
        {
            var x = (Vector<float>*)px; var b = (Vector<float>*)pb; var y = (Vector<float>*)py;
            int n = xs.Length / w, bn = m / w, bi = 0;
            for (int i = 0; i < n; i += 4)
            {
                var tv0 = x[i + 0] + b[bi + 0]; var v0 = scale * tv0;
                var tv1 = x[i + 1] + b[bi + 1]; var v1 = scale * tv1;
                var tv2 = x[i + 2] + b[bi + 2]; var v2 = scale * tv2;
                var tv3 = x[i + 3] + b[bi + 3]; var v3 = scale * tv3;
                var negZero0 = new Vector<float>(-0.0f);
                var negZero1 = new Vector<float>(-0.0f);
                var negZero2 = new Vector<float>(-0.0f);
                var negZero3 = new Vector<float>(-0.0f);
                var signBits0 = Vector.BitwiseAnd(v0, negZero0);
                var signBits1 = Vector.BitwiseAnd(v1, negZero1);
                var signBits2 = Vector.BitwiseAnd(v2, negZero2);
                var signBits3 = Vector.BitwiseAnd(v3, negZero3);
                var ax0 = Vector.BitwiseAnd(Vector.OnesComplement(negZero0), v0);
                var ax1 = Vector.BitwiseAnd(Vector.OnesComplement(negZero1), v1);
                var ax2 = Vector.BitwiseAnd(Vector.OnesComplement(negZero2), v2);
                var ax3 = Vector.BitwiseAnd(Vector.OnesComplement(negZero3), v3);
                ax0 = Vector.ConditionalSelect(Vector.GreaterThan(ax0, new Vector<float>(3.925f)), new Vector<float>(3.925f), ax0);
                ax1 = Vector.ConditionalSelect(Vector.GreaterThan(ax1, new Vector<float>(3.925f)), new Vector<float>(3.925f), ax1);
                ax2 = Vector.ConditionalSelect(Vector.GreaterThan(ax2, new Vector<float>(3.925f)), new Vector<float>(3.925f), ax2);
                ax3 = Vector.ConditionalSelect(Vector.GreaterThan(ax3, new Vector<float>(3.925f)), new Vector<float>(3.925f), ax3);
                var sq0 = ax0 * ax0;
                var sq1 = ax1 * ax1;
                var sq2 = ax2 * ax2;
                var sq3 = ax3 * ax3;
                var rs0 = new Vector<float>(-5.99104969e-4f);
                var rs1 = new Vector<float>(-5.99104969e-4f);
                var rs2 = new Vector<float>(-5.99104969e-4f);
                var rs3 = new Vector<float>(-5.99104969e-4f);
                rs0 = Vector.FusedMultiplyAdd(rs0, sq0, new Vector<float>(4.99339588e-3f));
                rs1 = Vector.FusedMultiplyAdd(rs1, sq1, new Vector<float>(4.99339588e-3f));
                rs2 = Vector.FusedMultiplyAdd(rs2, sq2, new Vector<float>(4.99339588e-3f));
                rs3 = Vector.FusedMultiplyAdd(rs3, sq3, new Vector<float>(4.99339588e-3f));
                rs0 = Vector.FusedMultiplyAdd(rs0, sq0, new Vector<float>(-2.67667342e-2f));
                rs1 = Vector.FusedMultiplyAdd(rs1, sq1, new Vector<float>(-2.67667342e-2f));
                rs2 = Vector.FusedMultiplyAdd(rs2, sq2, new Vector<float>(-2.67667342e-2f));
                rs3 = Vector.FusedMultiplyAdd(rs3, sq3, new Vector<float>(-2.67667342e-2f));
                rs0 = Vector.FusedMultiplyAdd(rs0, sq0, new Vector<float>(1.12818025e-1f));
                rs1 = Vector.FusedMultiplyAdd(rs1, sq1, new Vector<float>(1.12818025e-1f));
                rs2 = Vector.FusedMultiplyAdd(rs2, sq2, new Vector<float>(1.12818025e-1f));
                rs3 = Vector.FusedMultiplyAdd(rs3, sq3, new Vector<float>(1.12818025e-1f));
                rs0 = Vector.FusedMultiplyAdd(rs0, sq0, new Vector<float>(-3.76124859e-1f));
                rs1 = Vector.FusedMultiplyAdd(rs1, sq1, new Vector<float>(-3.76124859e-1f));
                rs2 = Vector.FusedMultiplyAdd(rs2, sq2, new Vector<float>(-3.76124859e-1f));
                rs3 = Vector.FusedMultiplyAdd(rs3, sq3, new Vector<float>(-3.76124859e-1f));
                rs0 = Vector.FusedMultiplyAdd(rs0, sq0, new Vector<float>(1.28379151e-1f));
                rs1 = Vector.FusedMultiplyAdd(rs1, sq1, new Vector<float>(1.28379151e-1f));
                rs2 = Vector.FusedMultiplyAdd(rs2, sq2, new Vector<float>(1.28379151e-1f));
                rs3 = Vector.FusedMultiplyAdd(rs3, sq3, new Vector<float>(1.28379151e-1f));
                rs0 = Vector.FusedMultiplyAdd(rs0, ax0, ax0);
                rs1 = Vector.FusedMultiplyAdd(rs1, ax1, ax1);
                rs2 = Vector.FusedMultiplyAdd(rs2, ax2, ax2);
                rs3 = Vector.FusedMultiplyAdd(rs3, ax3, ax3);
                var big0 = Vector.GreaterThan(ax0, new Vector<float>(0.921875f));
                var big1 = Vector.GreaterThan(ax1, new Vector<float>(0.921875f));
                var big2 = Vector.GreaterThan(ax2, new Vector<float>(0.921875f));
                var big3 = Vector.GreaterThan(ax3, new Vector<float>(0.921875f));
                rs0 = Vector.ConditionalSelect(big0, Vector<float>.Zero, rs0);
                rs1 = Vector.ConditionalSelect(big1, Vector<float>.Zero, rs1);
                rs2 = Vector.ConditionalSelect(big2, Vector<float>.Zero, rs2);
                rs3 = Vector.ConditionalSelect(big3, Vector<float>.Zero, rs3);
                var ab0 = Vector.ConditionalSelect(big0, ax0, Vector<float>.Zero);
                var ab1 = Vector.ConditionalSelect(big1, ax1, Vector<float>.Zero);
                var ab2 = Vector.ConditionalSelect(big2, ax2, Vector<float>.Zero);
                var ab3 = Vector.ConditionalSelect(big3, ax3, Vector<float>.Zero);
                var rb0 = new Vector<float>(1.72948930e-5f);
                var rb1 = new Vector<float>(1.72948930e-5f);
                var rb2 = new Vector<float>(1.72948930e-5f);
                var rb3 = new Vector<float>(1.72948930e-5f);
                rb0 = Vector.FusedMultiplyAdd(rb0, ab0, new Vector<float>(-3.83208680e-4f));
                rb1 = Vector.FusedMultiplyAdd(rb1, ab1, new Vector<float>(-3.83208680e-4f));
                rb2 = Vector.FusedMultiplyAdd(rb2, ab2, new Vector<float>(-3.83208680e-4f));
                rb3 = Vector.FusedMultiplyAdd(rb3, ab3, new Vector<float>(-3.83208680e-4f));
                rb0 = Vector.FusedMultiplyAdd(rb0, ab0, new Vector<float>(3.88393435e-3f));
                rb1 = Vector.FusedMultiplyAdd(rb1, ab1, new Vector<float>(3.88393435e-3f));
                rb2 = Vector.FusedMultiplyAdd(rb2, ab2, new Vector<float>(3.88393435e-3f));
                rb3 = Vector.FusedMultiplyAdd(rb3, ab3, new Vector<float>(3.88393435e-3f));
                rb0 = Vector.FusedMultiplyAdd(rb0, ab0, new Vector<float>(-2.42545605e-2f));
                rb1 = Vector.FusedMultiplyAdd(rb1, ab1, new Vector<float>(-2.42545605e-2f));
                rb2 = Vector.FusedMultiplyAdd(rb2, ab2, new Vector<float>(-2.42545605e-2f));
                rb3 = Vector.FusedMultiplyAdd(rb3, ab3, new Vector<float>(-2.42545605e-2f));
                rb0 = Vector.FusedMultiplyAdd(rb0, ab0, new Vector<float>(1.06777847e-1f));
                rb1 = Vector.FusedMultiplyAdd(rb1, ab1, new Vector<float>(1.06777847e-1f));
                rb2 = Vector.FusedMultiplyAdd(rb2, ab2, new Vector<float>(1.06777847e-1f));
                rb3 = Vector.FusedMultiplyAdd(rb3, ab3, new Vector<float>(1.06777847e-1f));
                rb0 = Vector.FusedMultiplyAdd(rb0, ab0, new Vector<float>(6.34846687e-1f));
                rb1 = Vector.FusedMultiplyAdd(rb1, ab1, new Vector<float>(6.34846687e-1f));
                rb2 = Vector.FusedMultiplyAdd(rb2, ab2, new Vector<float>(6.34846687e-1f));
                rb3 = Vector.FusedMultiplyAdd(rb3, ab3, new Vector<float>(6.34846687e-1f));
                rb0 = Vector.FusedMultiplyAdd(rb0, ab0, new Vector<float>(1.28717512e-1f));
                rb1 = Vector.FusedMultiplyAdd(rb1, ab1, new Vector<float>(1.28717512e-1f));
                rb2 = Vector.FusedMultiplyAdd(rb2, ab2, new Vector<float>(1.28717512e-1f));
                rb3 = Vector.FusedMultiplyAdd(rb3, ab3, new Vector<float>(1.28717512e-1f));
                rb0 = Vector.FusedMultiplyAdd(rb0, ab0, ab0);
                rb1 = Vector.FusedMultiplyAdd(rb1, ab1, ab1);
                rb2 = Vector.FusedMultiplyAdd(rb2, ab2, ab2);
                rb3 = Vector.FusedMultiplyAdd(rb3, ab3, ab3);
                var t0 = Vector<float>.Zero - rb0;
                var t1 = Vector<float>.Zero - rb1;
                var t2 = Vector<float>.Zero - rb2;
                var t3 = Vector<float>.Zero - rb3;
                t0 = Vector.ConditionalSelect(Vector.LessThan(t0, new Vector<float>(-88.3762626647949f)), new Vector<float>(-88.3762626647949f), t0);
                t1 = Vector.ConditionalSelect(Vector.LessThan(t1, new Vector<float>(-88.3762626647949f)), new Vector<float>(-88.3762626647949f), t1);
                t2 = Vector.ConditionalSelect(Vector.LessThan(t2, new Vector<float>(-88.3762626647949f)), new Vector<float>(-88.3762626647949f), t2);
                t3 = Vector.ConditionalSelect(Vector.LessThan(t3, new Vector<float>(-88.3762626647949f)), new Vector<float>(-88.3762626647949f), t3);
                var r0 = Vector.FusedMultiplyAdd(new Vector<float>(1.44269504088896341f), t0, new Vector<float>(12582912.0f));
                var r1 = Vector.FusedMultiplyAdd(new Vector<float>(1.44269504088896341f), t1, new Vector<float>(12582912.0f));
                var r2 = Vector.FusedMultiplyAdd(new Vector<float>(1.44269504088896341f), t2, new Vector<float>(12582912.0f));
                var r3 = Vector.FusedMultiplyAdd(new Vector<float>(1.44269504088896341f), t3, new Vector<float>(12582912.0f));
                r0 = r0 - new Vector<float>(12582912.0f);
                r1 = r1 - new Vector<float>(12582912.0f);
                r2 = r2 - new Vector<float>(12582912.0f);
                r3 = r3 - new Vector<float>(12582912.0f);
                var fx0 = Vector.FusedMultiplyAdd(r0, new Vector<float>(-6.93145752e-1f), t0);
                var fx1 = Vector.FusedMultiplyAdd(r1, new Vector<float>(-6.93145752e-1f), t1);
                var fx2 = Vector.FusedMultiplyAdd(r2, new Vector<float>(-6.93145752e-1f), t2);
                var fx3 = Vector.FusedMultiplyAdd(r3, new Vector<float>(-6.93145752e-1f), t3);
                fx0 = Vector.FusedMultiplyAdd(r0, new Vector<float>(-1.42860677e-6f), fx0);
                fx1 = Vector.FusedMultiplyAdd(r1, new Vector<float>(-1.42860677e-6f), fx1);
                fx2 = Vector.FusedMultiplyAdd(r2, new Vector<float>(-1.42860677e-6f), fx2);
                fx3 = Vector.FusedMultiplyAdd(r3, new Vector<float>(-1.42860677e-6f), fx3);
                var y0 = new Vector<float>(1.38319808e-3f);
                var y1 = new Vector<float>(1.38319808e-3f);
                var y2 = new Vector<float>(1.38319808e-3f);
                var y3 = new Vector<float>(1.38319808e-3f);
                y0 = Vector.FusedMultiplyAdd(y0, fx0, new Vector<float>(8.37550033e-3f));
                y1 = Vector.FusedMultiplyAdd(y1, fx1, new Vector<float>(8.37550033e-3f));
                y2 = Vector.FusedMultiplyAdd(y2, fx2, new Vector<float>(8.37550033e-3f));
                y3 = Vector.FusedMultiplyAdd(y3, fx3, new Vector<float>(8.37550033e-3f));
                y0 = Vector.FusedMultiplyAdd(y0, fx0, new Vector<float>(4.16689515e-2f));
                y1 = Vector.FusedMultiplyAdd(y1, fx1, new Vector<float>(4.16689515e-2f));
                y2 = Vector.FusedMultiplyAdd(y2, fx2, new Vector<float>(4.16689515e-2f));
                y3 = Vector.FusedMultiplyAdd(y3, fx3, new Vector<float>(4.16689515e-2f));
                y0 = Vector.FusedMultiplyAdd(y0, fx0, new Vector<float>(1.66664466e-1f));
                y1 = Vector.FusedMultiplyAdd(y1, fx1, new Vector<float>(1.66664466e-1f));
                y2 = Vector.FusedMultiplyAdd(y2, fx2, new Vector<float>(1.66664466e-1f));
                y3 = Vector.FusedMultiplyAdd(y3, fx3, new Vector<float>(1.66664466e-1f));
                y0 = Vector.FusedMultiplyAdd(y0, fx0, new Vector<float>(4.99999851e-1f));
                y1 = Vector.FusedMultiplyAdd(y1, fx1, new Vector<float>(4.99999851e-1f));
                y2 = Vector.FusedMultiplyAdd(y2, fx2, new Vector<float>(4.99999851e-1f));
                y3 = Vector.FusedMultiplyAdd(y3, fx3, new Vector<float>(4.99999851e-1f));
                y0 = Vector.FusedMultiplyAdd(y0, fx0, Vector<float>.One);
                y1 = Vector.FusedMultiplyAdd(y1, fx1, Vector<float>.One);
                y2 = Vector.FusedMultiplyAdd(y2, fx2, Vector<float>.One);
                y3 = Vector.FusedMultiplyAdd(y3, fx3, Vector<float>.One);
                y0 = Vector.FusedMultiplyAdd(y0, fx0, Vector<float>.One);
                y1 = Vector.FusedMultiplyAdd(y1, fx1, Vector<float>.One);
                y2 = Vector.FusedMultiplyAdd(y2, fx2, Vector<float>.One);
                y3 = Vector.FusedMultiplyAdd(y3, fx3, Vector<float>.One);
                var ri0 = Vector.ConvertToInt32(r0);
                var ri1 = Vector.ConvertToInt32(r1);
                var ri2 = Vector.ConvertToInt32(r2);
                var ri3 = Vector.ConvertToInt32(r3);
                ri0 = Vector.Min(Vector.Max(ri0, new Vector<int>(-126)), new Vector<int>(127));
                ri1 = Vector.Min(Vector.Max(ri1, new Vector<int>(-126)), new Vector<int>(127));
                ri2 = Vector.Min(Vector.Max(ri2, new Vector<int>(-126)), new Vector<int>(127));
                ri3 = Vector.Min(Vector.Max(ri3, new Vector<int>(-126)), new Vector<int>(127));
                y0 = y0 * Vector.AsVectorSingle(Vector.ShiftLeft(ri0 + new Vector<int>(127), 23));
                y1 = y1 * Vector.AsVectorSingle(Vector.ShiftLeft(ri1 + new Vector<int>(127), 23));
                y2 = y2 * Vector.AsVectorSingle(Vector.ShiftLeft(ri2 + new Vector<int>(127), 23));
                y3 = y3 * Vector.AsVectorSingle(Vector.ShiftLeft(ri3 + new Vector<int>(127), 23));
                y0 = Vector<float>.One - y0;
                y1 = Vector<float>.One - y1;
                y2 = Vector<float>.One - y2;
                y3 = Vector<float>.One - y3;
                y0 = Vector.BitwiseOr(rs0, y0);
                y1 = Vector.BitwiseOr(rs1, y1);
                y2 = Vector.BitwiseOr(rs2, y2);
                y3 = Vector.BitwiseOr(rs3, y3);
                y[i + 0] = half * tv0 * (one + (Vector.BitwiseOr(y0, signBits0)));
                y[i + 1] = half * tv1 * (one + (Vector.BitwiseOr(y1, signBits1)));
                y[i + 2] = half * tv2 * (one + (Vector.BitwiseOr(y2, signBits2)));
                y[i + 3] = half * tv3 * (one + (Vector.BitwiseOr(y3, signBits3)));
                bi += 4; if (bi >= bn) bi = 0;
            }
        }
    }
}
