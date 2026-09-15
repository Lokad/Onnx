using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;

using Lokad.Onnx;
using Microsoft.ML.OnnxRuntime;

// P03/M1 prototype: one complete channel-blocked ResNet bottleneck region, kept outside
// the product graph executor. Blocked activations lay out as [N, Cb, H, W, 8] with the
// 8-channel block innermost so one AVX2 vector holds one block. The region covers entry
// reorder, 1x1 reduction, 3x3 spatial convolution, 1x1 expansion, residual add fused with
// Relu at the final store, and exit reorder. Filters are prepacked once with a guard that
// trips when the source array is replaced, so stale packs can never price wrong weights.
internal static class BlockedBottleneck
{
    internal const int Bc = 8;

    static int BIndex(int n, int cb, int h, int w, int b, int Cb, int H, int W)
        => ((((n * Cb) + cb) * H + h) * W + w) * Bc + b;

    static void ToBlockedInto(float[] src, float[] dst, int N, int C, int H, int W, int Cb)
    {
        for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++)
            dst[BIndex(n, c / Bc, h, w, c % Bc, Cb, H, W)] = src[(((n * C) + c) * H + h) * W + w];
    }

    static float[] ToBlocked(float[] src, int N, int C, int H, int W, int Cb)
    {
        var dst = new float[N * Cb * H * W * Bc];
        ToBlockedInto(src, dst, N, C, H, W, Cb);
        return dst;
    }

    static float[] FromBlocked(float[] src, int N, int C, int H, int W, int Cb)
    {
        var dst = new float[N * C * H * W];
        for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
        for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++)
            dst[(((n * C) + c) * H + h) * W + w] = src[BIndex(n, c / Bc, h, w, c % Bc, Cb, H, W)];
        return dst;
    }

    internal sealed class BlockedFilter
    {
        internal readonly float[] Packed;
        internal readonly int Kt;
        internal readonly int Ct;
        internal readonly int KH;
        internal readonly int KW;
        readonly float[] Source;

        BlockedFilter(float[] packed, int kt, int ct, int kh, int kw, float[] source)
        {
            Packed = packed;
            Kt = kt;
            Ct = ct;
            KH = kh;
            KW = kw;
            Source = source;
        }

        internal bool Check(float[] src) => ReferenceEquals(src, Source) && src.Length == Source.Length;

        internal static BlockedFilter Pack(float[] src, int K, int C, int KH, int KW)
        {
            int Kt = (K + Bc - 1) / Bc;
            int Ct = (C + Bc - 1) / Bc;
            var p = new float[Kt * Ct * KH * KW * Bc * Bc];
            for (int k = 0; k < K; k++)
            for (int c = 0; c < C; c++)
            for (int kh = 0; kh < KH; kh++)
            for (int kw = 0; kw < KW; kw++)
                p[(((((k / Bc) * Ct) + (c / Bc)) * KH + kh) * KW + kw) * Bc * Bc + ((c % Bc) * Bc) + (k % Bc)]
                    = src[(((k * C) + c) * KH + kh) * KW + kw];
            return new BlockedFilter(p, Kt, Ct, KH, KW, src);
        }
    }

    internal struct RegionCost
    {
        internal double ReorderMs;
        internal double KernelMs;
        internal double Pw1Ms;
        internal double Sp3Ms;
        internal double Pw2Ms;
    }

    internal sealed class RegionWorkspace
    {
        int N, C, R, H, W;
        internal float[] Xb, Rb, T1, T2, Yb;

        internal RegionWorkspace()
        {
            Xb = Array.Empty<float>();
            Rb = Array.Empty<float>();
            T1 = Array.Empty<float>();
            T2 = Array.Empty<float>();
            Yb = Array.Empty<float>();
        }

        internal void Ensure(int n, int c, int r, int h, int w)
        {
            if (N == n && C == c && R == r && H == h && W == w && Xb.Length > 0) return;
            N = n;
            C = c;
            R = r;
            H = h;
            W = w;
            int CbC = (c + Bc - 1) / Bc;
            int CbR = (r + Bc - 1) / Bc;
            Xb = new float[n * CbC * h * w * Bc];
            Rb = new float[n * CbC * h * w * Bc];
            T1 = new float[n * CbR * h * w * Bc];
            T2 = new float[n * CbR * h * w * Bc];
            Yb = new float[n * CbC * h * w * Bc];
        }

        internal int Bytes => (Xb.Length + Rb.Length + T1.Length + T2.Length + Yb.Length) * 4;
    }

    static void PointwiseBlocked(float[] x, BlockedFilter w, float[]? bias, float[] y,
        int N, int C, int K, int CbIn, int CbOut, int H, int W, bool relu)
    {
        Span<float> acc = stackalloc float[Bc];
        if (w.KH != 1 || w.KW != 1) throw new ArgumentException("Pointwise kernel needs a 1x1 pack.");
        for (int n = 0; n < N; n++)
        for (int h = 0; h < H; h++)
        for (int wpos = 0; wpos < W; wpos++)
        for (int kb = 0; kb < w.Kt; kb++)
        {
            int kCount = Math.Min(Bc, K - (kb * Bc));
            for (int bi = 0; bi < kCount; bi++) acc[bi] = bias == null ? 0f : bias[(kb * Bc) + bi];
            for (int cb = 0; cb < w.Ct; cb++)
            {
                int cCount = Math.Min(Bc, C - (cb * Bc));
                int xBase = BIndex(n, cb, h, wpos, 0, CbIn, H, W);
                int wBase = ((kb * w.Ct) + cb) * Bc * Bc;
                if (kCount == Bc && Vector<float>.Count == Bc && cCount == Bc)
                {
                    var q0 = new Vector<float>(acc);
                    var q1 = Vector<float>.Zero;
                    var q2 = Vector<float>.Zero;
                    var q3 = Vector<float>.Zero;
                    for (int ci = 0; ci < Bc; ci += 4)
                    {
                        float x0 = x[xBase + ci];
                        float x1 = x[xBase + ci + 1];
                        float x2 = x[xBase + ci + 2];
                        float x3 = x[xBase + ci + 3];
                        q0 = Vector.FusedMultiplyAdd(new Vector<float>(w.Packed, wBase + (ci * Bc)), new Vector<float>(x0), q0);
                        q1 = Vector.FusedMultiplyAdd(new Vector<float>(w.Packed, wBase + ((ci + 1) * Bc)), new Vector<float>(x1), q1);
                        q2 = Vector.FusedMultiplyAdd(new Vector<float>(w.Packed, wBase + ((ci + 2) * Bc)), new Vector<float>(x2), q2);
                        q3 = Vector.FusedMultiplyAdd(new Vector<float>(w.Packed, wBase + ((ci + 3) * Bc)), new Vector<float>(x3), q3);
                    }
                    q0 += q1 + q2 + q3;
                    q0.CopyTo(acc);
                }
                else
                    if (kCount == Bc && Vector<float>.Count == Bc)
                    {
                        var accV0 = new Vector<float>(acc);
                        var accV1 = Vector<float>.Zero;
                        int ci = 0;
                        int cPairs = cCount & ~1;
                        for (; ci < cPairs; ci += 2)
                        {
                            float x0 = x[xBase + ci];
                            float x1 = x[xBase + ci + 1];
                            accV0 = Vector.FusedMultiplyAdd(new Vector<float>(w.Packed, wBase + (ci * Bc)), new Vector<float>(x0), accV0);
                            accV1 = Vector.FusedMultiplyAdd(new Vector<float>(w.Packed, wBase + ((ci + 1) * Bc)), new Vector<float>(x1), accV1);
                        }
                        for (; ci < cCount; ci++)
                        {
                            float xv = x[xBase + ci];
                            accV0 = Vector.FusedMultiplyAdd(new Vector<float>(w.Packed, wBase + (ci * Bc)), new Vector<float>(xv), accV0);
                        }
                        accV0 += accV1;
                        accV0.CopyTo(acc);
                    }
                else
                {
                    for (int ci = 0; ci < cCount; ci++)
                    {
                        float xv = x[xBase + ci];
                        int wRow = wBase + (ci * Bc);
                        int bi = 0;
                        int step = Vector<float>.Count;
                        var xvV = new Vector<float>(xv);
                        for (; bi + step <= kCount; bi += step)
                        {
                            var av = new Vector<float>(acc.Slice(bi));
                            av = Vector.FusedMultiplyAdd(new Vector<float>(w.Packed, wRow + bi), xvV, av);
                            av.CopyTo(acc.Slice(bi));
                        }
                        for (; bi < kCount; bi++) acc[bi] += w.Packed[wRow + bi] * xv;
                    }
                }
            }
            int yBase = BIndex(n, kb, h, wpos, 0, CbOut, H, W);
            for (int bi = 0; bi < kCount; bi++)
            {
                float v = acc[bi];
                if (relu && v < 0f) v = 0f;
                y[yBase + bi] = v;
            }
        }
    }

    static void Spatial3x3Blocked(float[] x, BlockedFilter w, float[]? bias, float[] y,
        int N, int C, int K, int CbIn, int CbOut, int H, int W, bool relu)
    {
        Span<float> acc = stackalloc float[Bc];
        if (w.KH != 3 || w.KW != 3) throw new ArgumentException("Spatial kernel needs a 3x3 pack.");
        for (int n = 0; n < N; n++)
        for (int h = 0; h < H; h++)
        for (int wpos = 0; wpos < W; wpos++)
        for (int kb = 0; kb < w.Kt; kb++)
        {
            int kCount = Math.Min(Bc, K - (kb * Bc));
            for (int bi = 0; bi < kCount; bi++) acc[bi] = bias == null ? 0f : bias[(kb * Bc) + bi];
            for (int cb = 0; cb < w.Ct; cb++)
            for (int kh = 0; kh < 3; kh++)
            for (int kw = 0; kw < 3; kw++)
            {
                int ih = h + kh - 1;
                int iw = wpos + kw - 1;
                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                int cCount = Math.Min(Bc, C - (cb * Bc));
                int xBase = BIndex(n, cb, ih, iw, 0, CbIn, H, W);
                int wBase = ((((kb * w.Ct) + cb) * 3 + kh) * 3 + kw) * Bc * Bc;
                if (kCount == Bc && Vector<float>.Count == Bc && cCount == Bc)
                {
                    var q0 = new Vector<float>(acc);
                    var q1 = Vector<float>.Zero;
                    var q2 = Vector<float>.Zero;
                    var q3 = Vector<float>.Zero;
                    for (int ci = 0; ci < Bc; ci += 4)
                    {
                        float x0 = x[xBase + ci];
                        float x1 = x[xBase + ci + 1];
                        float x2 = x[xBase + ci + 2];
                        float x3 = x[xBase + ci + 3];
                        q0 = Vector.FusedMultiplyAdd(new Vector<float>(w.Packed, wBase + (ci * Bc)), new Vector<float>(x0), q0);
                        q1 = Vector.FusedMultiplyAdd(new Vector<float>(w.Packed, wBase + ((ci + 1) * Bc)), new Vector<float>(x1), q1);
                        q2 = Vector.FusedMultiplyAdd(new Vector<float>(w.Packed, wBase + ((ci + 2) * Bc)), new Vector<float>(x2), q2);
                        q3 = Vector.FusedMultiplyAdd(new Vector<float>(w.Packed, wBase + ((ci + 3) * Bc)), new Vector<float>(x3), q3);
                    }
                    q0 += q1 + q2 + q3;
                    q0.CopyTo(acc);
                }
                else
                    if (kCount == Bc && Vector<float>.Count == Bc)
                    {
                        var accV0 = new Vector<float>(acc);
                        var accV1 = Vector<float>.Zero;
                        int ci = 0;
                        int cPairs = cCount & ~1;
                        for (; ci < cPairs; ci += 2)
                        {
                            float x0 = x[xBase + ci];
                            float x1 = x[xBase + ci + 1];
                            accV0 = Vector.FusedMultiplyAdd(new Vector<float>(w.Packed, wBase + (ci * Bc)), new Vector<float>(x0), accV0);
                            accV1 = Vector.FusedMultiplyAdd(new Vector<float>(w.Packed, wBase + ((ci + 1) * Bc)), new Vector<float>(x1), accV1);
                        }
                        for (; ci < cCount; ci++)
                        {
                            float xv = x[xBase + ci];
                            accV0 = Vector.FusedMultiplyAdd(new Vector<float>(w.Packed, wBase + (ci * Bc)), new Vector<float>(xv), accV0);
                        }
                        accV0 += accV1;
                        accV0.CopyTo(acc);
                    }
                else
                {
                    for (int ci = 0; ci < cCount; ci++)
                    {
                        float xv = x[xBase + ci];
                        int wRow = wBase + (ci * Bc);
                        int bi = 0;
                        int step = Vector<float>.Count;
                        var xvV = new Vector<float>(xv);
                        for (; bi + step <= kCount; bi += step)
                        {
                            var av = new Vector<float>(acc.Slice(bi));
                            av = Vector.FusedMultiplyAdd(new Vector<float>(w.Packed, wRow + bi), xvV, av);
                            av.CopyTo(acc.Slice(bi));
                        }
                        for (; bi < kCount; bi++) acc[bi] += w.Packed[wRow + bi] * xv;
                    }
                }
            }
            int yBase = BIndex(n, kb, h, wpos, 0, CbOut, H, W);
            for (int bi = 0; bi < kCount; bi++)
            {
                float v = acc[bi];
                if (relu && v < 0f) v = 0f;
                y[yBase + bi] = v;
            }
        }
    }

    static void PointwiseResidualBlocked(float[] x, BlockedFilter w, float[]? bias, float[] resid,
        float[] y, int N, int C, int K, int CbIn, int CbOut, int H, int W)
    {
        Span<float> acc = stackalloc float[Bc];
        if (w.KH != 1 || w.KW != 1) throw new ArgumentException("Pointwise kernel needs a 1x1 pack.");
        for (int n = 0; n < N; n++)
        for (int h = 0; h < H; h++)
        for (int wpos = 0; wpos < W; wpos++)
        for (int kb = 0; kb < w.Kt; kb++)
        {
            int kCount = Math.Min(Bc, K - (kb * Bc));
            for (int bi = 0; bi < kCount; bi++) acc[bi] = bias == null ? 0f : bias[(kb * Bc) + bi];
            for (int cb = 0; cb < w.Ct; cb++)
            {
                int cCount = Math.Min(Bc, C - (cb * Bc));
                int xBase = BIndex(n, cb, h, wpos, 0, CbIn, H, W);
                int wBase = ((kb * w.Ct) + cb) * Bc * Bc;
                if (kCount == Bc && Vector<float>.Count == Bc && cCount == Bc)
                {
                    var q0 = new Vector<float>(acc);
                    var q1 = Vector<float>.Zero;
                    var q2 = Vector<float>.Zero;
                    var q3 = Vector<float>.Zero;
                    for (int ci = 0; ci < Bc; ci += 4)
                    {
                        float x0 = x[xBase + ci];
                        float x1 = x[xBase + ci + 1];
                        float x2 = x[xBase + ci + 2];
                        float x3 = x[xBase + ci + 3];
                        q0 = Vector.FusedMultiplyAdd(new Vector<float>(w.Packed, wBase + (ci * Bc)), new Vector<float>(x0), q0);
                        q1 = Vector.FusedMultiplyAdd(new Vector<float>(w.Packed, wBase + ((ci + 1) * Bc)), new Vector<float>(x1), q1);
                        q2 = Vector.FusedMultiplyAdd(new Vector<float>(w.Packed, wBase + ((ci + 2) * Bc)), new Vector<float>(x2), q2);
                        q3 = Vector.FusedMultiplyAdd(new Vector<float>(w.Packed, wBase + ((ci + 3) * Bc)), new Vector<float>(x3), q3);
                    }
                    q0 += q1 + q2 + q3;
                    q0.CopyTo(acc);
                }
                else
                    if (kCount == Bc && Vector<float>.Count == Bc)
                    {
                        var accV0 = new Vector<float>(acc);
                        var accV1 = Vector<float>.Zero;
                        int ci = 0;
                        int cPairs = cCount & ~1;
                        for (; ci < cPairs; ci += 2)
                        {
                            float x0 = x[xBase + ci];
                            float x1 = x[xBase + ci + 1];
                            accV0 = Vector.FusedMultiplyAdd(new Vector<float>(w.Packed, wBase + (ci * Bc)), new Vector<float>(x0), accV0);
                            accV1 = Vector.FusedMultiplyAdd(new Vector<float>(w.Packed, wBase + ((ci + 1) * Bc)), new Vector<float>(x1), accV1);
                        }
                        for (; ci < cCount; ci++)
                        {
                            float xv = x[xBase + ci];
                            accV0 = Vector.FusedMultiplyAdd(new Vector<float>(w.Packed, wBase + (ci * Bc)), new Vector<float>(xv), accV0);
                        }
                        accV0 += accV1;
                        accV0.CopyTo(acc);
                    }
                else
                {
                    for (int ci = 0; ci < cCount; ci++)
                    {
                        float xv = x[xBase + ci];
                        int wRow = wBase + (ci * Bc);
                        int bi = 0;
                        int step = Vector<float>.Count;
                        var xvV = new Vector<float>(xv);
                        for (; bi + step <= kCount; bi += step)
                        {
                            var av = new Vector<float>(acc.Slice(bi));
                            av = Vector.FusedMultiplyAdd(new Vector<float>(w.Packed, wRow + bi), xvV, av);
                            av.CopyTo(acc.Slice(bi));
                        }
                        for (; bi < kCount; bi++) acc[bi] += w.Packed[wRow + bi] * xv;
                    }
                }
            }
            int yBase = BIndex(n, kb, h, wpos, 0, CbOut, H, W);
            for (int bi = 0; bi < kCount; bi++)
            {
                float v = acc[bi] + resid[yBase + bi];
                if (v < 0f) v = 0f;
                y[yBase + bi] = v;
            }
        }
    }

    internal static float[]? RunRegion(float[] x,
        float[] f1, BlockedFilter w1, float[]? b1,
        float[] f2, BlockedFilter w2, float[]? b2,
        float[] f3, BlockedFilter w3, float[]? b3,
        float[] resid, int N, int C, int R, int H, int W,
        RegionWorkspace? ws, out int scratchBytes, out RegionCost cost)
    {
        scratchBytes = 0;
        cost = default;
        if (w1 == null || w2 == null || w3 == null) return null;
        if (!w1.Check(f1) || !w2.Check(f2) || !w3.Check(f3)) return null;
        int CbC = (C + Bc - 1) / Bc;
        int CbR = (R + Bc - 1) / Bc;
        float[] xb, rb, t1, t2, yb;
        if (ws == null)
        {
            xb = new float[N * CbC * H * W * Bc];
            rb = new float[N * CbC * H * W * Bc];
            t1 = new float[N * CbR * H * W * Bc];
            t2 = new float[N * CbR * H * W * Bc];
            yb = new float[N * CbC * H * W * Bc];
        }
        else
        {
            ws.Ensure(N, C, R, H, W);
            xb = ws.Xb;
            rb = ws.Rb;
            t1 = ws.T1;
            t2 = ws.T2;
            yb = ws.Yb;
        }
        scratchBytes = (t1.Length + t2.Length + yb.Length + xb.Length + rb.Length) * 4;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        ToBlockedInto(x, xb, N, C, H, W, CbC);
        ToBlockedInto(resid, rb, N, C, H, W, CbC);
        sw.Stop();
        double reorder = sw.Elapsed.TotalMilliseconds;
        sw.Restart();
        PointwiseBlocked(xb, w1, b1, t1, N, C, R, CbC, CbR, H, W, true);
        sw.Stop();
        double pw1 = sw.Elapsed.TotalMilliseconds;
        sw.Restart();
        Spatial3x3Blocked(t1, w2, b2, t2, N, R, R, CbR, CbR, H, W, true);
        sw.Stop();
        double sp3 = sw.Elapsed.TotalMilliseconds;
        sw.Restart();
        PointwiseResidualBlocked(t2, w3, b3, rb, yb, N, R, C, CbR, CbC, H, W);
        sw.Stop();
        double pw2 = sw.Elapsed.TotalMilliseconds;
        double kernel = pw1 + sp3 + pw2;
        sw.Restart();
        float[] y = FromBlocked(yb, N, C, H, W, CbC);
        sw.Stop();
        cost = new RegionCost { ReorderMs = reorder + sw.Elapsed.TotalMilliseconds, KernelMs = kernel, Pw1Ms = pw1, Sp3Ms = sp3, Pw2Ms = pw2 };
        return y;
    }

    static float[] RunLegacy(float[] x,
        float[] f1, float[]? b1, float[] f2, float[]? b2, float[] f3, float[]? b3,
        float[] resid, int N, int C, int R, int H, int W)
    {
        var opts = TensorExecutionOptions.Auto with { MaxDegreeOfParallelism = 1 };
        var X = DenseTensor<float>.OfValues(x.AsSpan(), new int[] { N, C, H, W });
        var W1 = DenseTensor<float>.OfValues(f1.AsSpan(), new int[] { R, C, 1, 1 });
        var W2 = DenseTensor<float>.OfValues(f2.AsSpan(), new int[] { R, R, 3, 3 });
        var W3 = DenseTensor<float>.OfValues(f3.AsSpan(), new int[] { C, R, 1, 1 });
        var B1 = b1 == null ? null : DenseTensor<float>.OfValues(b1.AsSpan(), new int[] { R });
        var B2 = b2 == null ? null : DenseTensor<float>.OfValues(b2.AsSpan(), new int[] { R });
        var B3 = b3 == null ? null : DenseTensor<float>.OfValues(b3.AsSpan(), new int[] { C });
        var Rs = DenseTensor<float>.OfValues(resid.AsSpan(), new int[] { N, C, H, W });
        var y1 = Tensor<float>.Conv2D(X, W1, 1, new int[] { 0, 0, 0, 0 }, B1, new int[] { 1, 1 }, new int[] { 1, 1 }, null, opts);
        var y2 = Tensor<float>.Conv2D(Tensor<float>.Relu(y1), W2, 1, new int[] { 1, 1, 1, 1 }, B2, new int[] { 3, 3 }, new int[] { 1, 1 }, null, opts);
        var y3 = Tensor<float>.Conv2D(Tensor<float>.Relu(y2), W3, 1, new int[] { 0, 0, 0, 0 }, B3, new int[] { 1, 1 }, new int[] { 1, 1 }, null, opts);
        var o = Tensor<float>.Relu(Tensor<float>.Add(y3, Rs));
        return o.ToDenseTensor().Buffer.ToArray();
    }

    static float[] Rand(int n, Random rnd, float s)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rnd.NextDouble() * 2.0 - 1.0) * s;
        return a;
    }

    static double MaxScaled(float[] cand, float[] refer)
    {
        double worst = 0;
        for (int i = 0; i < cand.Length; i++)
            worst = Math.Max(worst, Math.Abs(cand[i] - refer[i]) / (1.0 + Math.Abs(refer[i])));
        return worst;
    }

    static int CheckCase(string name, int N, int C, int R, int H, int W, bool biased)
    {
        var rnd = new Random(1234 + N + C + R + H + W);
        float[] x = Rand(N * C * H * W, rnd, 1f);
        float[] f1 = Rand(R * C, rnd, 0.4f);
        float[] f2 = Rand(R * R * 9, rnd, 0.2f);
        float[] f3 = Rand(C * R, rnd, 0.4f);
        float[]? b1 = biased ? Rand(R, rnd, 0.5f) : null;
        float[]? b2 = biased ? Rand(R, rnd, 0.5f) : null;
        float[]? b3 = biased ? Rand(C, rnd, 0.5f) : null;
        float[] resid = Rand(N * C * H * W, rnd, 1f);
        var w1 = BlockedFilter.Pack(f1, R, C, 1, 1);
        var w2 = BlockedFilter.Pack(f2, R, R, 3, 3);
        var w3 = BlockedFilter.Pack(f3, C, R, 1, 1);
        float[] refer = RunLegacy(x, f1, b1, f2, b2, f3, b3, resid, N, C, R, H, W);
        float[]? cand = RunRegion(x, f1, w1, b1, f2, w2, b2, f3, w3, b3, resid, N, C, R, H, W, null, out int scratch, out RegionCost _);
        if (cand == null) { Console.WriteLine("case " + name + ": guard tripped on fresh packs (unexpected). FAIL"); return 1; }
        double ms = MaxScaled(cand, refer);
        bool ok = ms <= 1e-4;
        Console.WriteLine("case " + name + ": maxScaled=" + ms.ToString("E2") + " scratchBytes=" + scratch + (ok ? " PASS" : " FAIL"));
        return ok ? 0 : 1;
    }

    static int CheckGuardTrip()
    {
        var rnd = new Random(77);
        float[] x = Rand(16 * 1 * 1 * 4, rnd, 1f);
        float[] f1 = Rand(8 * 16, rnd, 0.4f);
        float[] f2 = Rand(8 * 8 * 9, rnd, 0.2f);
        float[] f3 = Rand(16 * 8, rnd, 0.4f);
        float[] b = Rand(8, rnd, 0.5f);
        float[] bc = Rand(16, rnd, 0.5f);
        float[] resid = Rand(16 * 1 * 1 * 4, rnd, 1f);
        var w1 = BlockedFilter.Pack(f1, 8, 16, 1, 1);
        var w2 = BlockedFilter.Pack(f2, 8, 8, 3, 3);
        var w3 = BlockedFilter.Pack(f3, 16, 8, 1, 1);
        float[] swapped = (float[])f1.Clone();
        float[]? got = RunRegion(x, swapped, w1, b, f2, w2, b, f3, w3, bc, resid, 1, 16, 8, 1, 4, null, out int _, out RegionCost _);
        bool ok = got == null;
        Console.WriteLine("case guard-trip: replaced source " + (ok ? "tripped PASS" : "missed FAIL"));
        return ok ? 0 : 1;
    }

    internal static int RunConvBlock(string root, string[] args)
    {
        if (args.Length == 1 && args[0] == "verify")
        {
            int rc = 0;
            rc |= CheckCase("A-target-bias", 1, 256, 64, 56, 56, true);
            rc |= CheckCase("B-target-nobias", 1, 256, 64, 56, 56, false);
            rc |= CheckCase("C-tail", 1, 24, 10, 5, 5, true);
            rc |= CheckCase("D-degenerate", 1, 16, 8, 1, 1, true);
            rc |= CheckGuardTrip();
            Console.WriteLine(rc == 0 ? "convblock verify: all cases agree." : "convblock verify: FAILED.");
            return rc;
        }
        if (args.Length >= 1 && args[0] == "run")
        {
            int reps = 9;
            for (int i = 1; i < args.Length; i++)
            {
                if (args[i] == "--reps" && i + 1 < args.Length && int.TryParse(args[i + 1], out int k) && k >= 1) { reps = k; i++; }
                else { Console.WriteLine("usage: Bench convblock run [--reps K]"); return 2; }
            }
            return RunTable(root, reps);
        }
        Console.WriteLine("usage: Bench convblock verify|run [--reps K]");
        return 2;
    }

    static float[] InitArray(ComputationalGraph graph, string name)
    {
        var t = (Tensor<float>)graph.Initializers[name];
        return t.ToDenseTensor().Buffer.ToArray();
    }

    static double MedianOf(double[] v)
    {
        var s = (double[])v.Clone();
        Array.Sort(s);
        return s[s.Length / 2];
    }

    static double BestOf(double[] v)
    {
        double b = v[0];
        foreach (double d in v) b = Math.Min(b, d);
        return b;
    }

    static int RunTable(string root, int reps)
    {
        string model = Path.Combine(root, "artifacts", "p03-region", "bottleneck.onnx");
        if (!File.Exists(model)) { Console.WriteLine("missing region model: " + model); return 1; }
        const int N = 1, C = 256, R = 64, H = 56, W = 56;
        var graph = OnnxImport.Load(model);
        if (graph == null) { Console.WriteLine("region model failed to load."); return 1; }
        float[] f1 = InitArray(graph, "W1");
        float[] b1 = InitArray(graph, "B1");
        float[] f2 = InitArray(graph, "W2");
        float[] b2 = InitArray(graph, "B2");
        float[] f3 = InitArray(graph, "W3");
        float[] b3 = InitArray(graph, "B3");
        var rnd = new Random(20260915);
        float[] x = Rand(N * C * H * W, rnd, 1f);
        using var so = Bench.CreateSingleCpuSessionOptions(1);
        using var session = new InferenceSession(model, so);
        string[] inNames = session.InputMetadata.Keys.ToArray();
        string[] outNames = session.OutputMetadata.Keys.ToArray();
        var xt = DenseTensor<float>.OfValues(x.AsSpan(), new int[] { N, C, H, W });
        var named = new Dictionary<string, ITensor>(StringComparer.Ordinal);
        named[inNames[0]] = xt;
        var ortInputs = Bench.BuildOrtInputs(named, inNames);
        try
        {
            float[] legacy = RunLegacy(x, f1, b1, f2, b2, f3, b3, x, N, C, R, H, W);
            var w1 = BlockedFilter.Pack(f1, R, C, 1, 1);
            var w2 = BlockedFilter.Pack(f2, R, R, 3, 3);
            var w3 = BlockedFilter.Pack(f3, C, R, 1, 1);
            var ws = new RegionWorkspace();
            float[]? blocked = RunRegion(x, f1, w1, b1, f2, w2, b2, f3, w3, b3, x, N, C, R, H, W, ws, out int scratch, out RegionCost _);
            if (blocked == null) { Console.WriteLine("blocked leg guard tripped on fresh packs. FAIL"); return 1; }
            float[] ort;
            using (var ro = new RunOptions())
            using (var o = session.Run(ro, ortInputs, outNames)) { ort = o[0].GetTensorDataAsSpan<float>().ToArray(); }
            double bl = MaxScaled(blocked, legacy);
            double ol = MaxScaled(ort, legacy);
            Console.WriteLine("agreement: blocked-vs-legacy=" + bl.ToString("E2") + " ort-vs-legacy=" + ol.ToString("E2"));
            if (bl > 1e-4 || ol > 1e-4) { Console.WriteLine("convblock run: agreement FAILED."); return 1; }
            var tL = new double[reps];
            var tB = new double[reps];
            var tO = new double[reps];
            var tR = new double[reps];
            var tK = new double[reps];
            var tP1 = new double[reps];
            var tS3 = new double[reps];
            var tP2 = new double[reps];
            var sw = new System.Diagnostics.Stopwatch();
            using (var ro = new RunOptions())
            for (int r = 0; r < reps; r++)
            {
                sw.Restart(); RunLegacy(x, f1, b1, f2, b2, f3, b3, x, N, C, R, H, W); sw.Stop(); tL[r] = sw.Elapsed.TotalMilliseconds;
                RegionCost c;
                sw.Restart(); RunRegion(x, f1, w1, b1, f2, w2, b2, f3, w3, b3, x, N, C, R, H, W, ws, out int _, out c); sw.Stop(); tB[r] = sw.Elapsed.TotalMilliseconds; tR[r] = c.ReorderMs; tK[r] = c.KernelMs; tP1[r] = c.Pw1Ms; tS3[r] = c.Sp3Ms; tP2[r] = c.Pw2Ms;
                sw.Restart(); using (var o = session.Run(ro, ortInputs, outNames)) { } sw.Stop(); tO[r] = sw.Elapsed.TotalMilliseconds;
            }
            long packBytes = (long)(w1.Packed.Length + w2.Packed.Length + w3.Packed.Length) * 4;
            Console.WriteLine("hardware: VectorCount=" + Vector<float>.Count + " accelerated=" + Vector.IsHardwareAccelerated);
            Console.WriteLine("region 1x1-64x256 + 3x3-64x64 + 1x1-256x64 at 56x56, reps=" + reps);
            Console.WriteLine("legacy  best=" + BestOf(tL).ToString("F2") + "ms median=" + MedianOf(tL).ToString("F2") + "ms");
            Console.WriteLine("blocked best=" + BestOf(tB).ToString("F2") + "ms median=" + MedianOf(tB).ToString("F2") + "ms");
            Console.WriteLine("ort     best=" + BestOf(tO).ToString("F2") + "ms median=" + MedianOf(tO).ToString("F2") + "ms");
            Console.WriteLine("bytes: scratch=" + scratch + " prepacked=" + packBytes);
            Console.WriteLine("split (steady-state workspace): reorder median=" + MedianOf(tR).ToString("F2") + "ms kernel median=" + MedianOf(tK).ToString("F2") + "ms [pw1=" + MedianOf(tP1).ToString("F2") + " sp3=" + MedianOf(tS3).ToString("F2") + " pw2=" + MedianOf(tP2).ToString("F2") + "]");
            return 0;
        }
        finally { foreach (var v in ortInputs.Values) v.Dispose(); }
    }
}
