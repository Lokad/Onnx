using System;
using System.Numerics;
using Lokad.Onnx;
// P07/M2 prototype: transposed-B micro-kernel outside the product.
// Computes C[m,n] = sum_k a[m*K+k] * b[n*K+k] with b stored row-major [N,K],
// i.e. the scores contraction without materializing the K transpose view.
// Both k-streams stay contiguous, which the generic-stride kernels do not
// exploit (see the P05 90x reversed-view penalty). Scalar tail covers any K;
// NaN ordering follows the loop and is owned by the existing kernels, so the
// twin gate stays finite like the P05 lab.
static class P07TransposedRead
{
    internal static float[] MmTransposedB4(float[] a, float[] b, int M, int K, int N)
    {
        // v2: four n-lanes share each A vector, keeping four FMA chains live.
        var c = new float[M * N];
        int step = Vector<float>.Count;
        int n4 = (N / 4) * 4;
        for (int m = 0; m < M; m++)
        {
            int n = 0;
            for (; n < n4; n += 4)
            {
                var acc0 = Vector<float>.Zero;
                var acc1 = Vector<float>.Zero;
                var acc2 = Vector<float>.Zero;
                var acc3 = Vector<float>.Zero;
                int k = 0;
                for (; k + step <= K; k += step)
                {
                    var av = new Vector<float>(a, m * K + k);
                    acc0 = Vector.FusedMultiplyAdd(av, new Vector<float>(b, (n + 0) * K + k), acc0);
                    acc1 = Vector.FusedMultiplyAdd(av, new Vector<float>(b, (n + 1) * K + k), acc1);
                    acc2 = Vector.FusedMultiplyAdd(av, new Vector<float>(b, (n + 2) * K + k), acc2);
                    acc3 = Vector.FusedMultiplyAdd(av, new Vector<float>(b, (n + 3) * K + k), acc3);
                }
                float s0 = Vector.Sum(acc0), s1 = Vector.Sum(acc1), s2 = Vector.Sum(acc2), s3 = Vector.Sum(acc3);
                for (; k < K; k++)
                {
                    float x = a[m * K + k];
                    s0 += x * b[(n + 0) * K + k];
                    s1 += x * b[(n + 1) * K + k];
                    s2 += x * b[(n + 2) * K + k];
                    s3 += x * b[(n + 3) * K + k];
                }
                c[m * N + n] = s0; c[m * N + n + 1] = s1; c[m * N + n + 2] = s2; c[m * N + n + 3] = s3;
            }
            for (; n < N; n++)
            {
                var acc = Vector<float>.Zero;
                int k = 0;
                for (; k + step <= K; k += step)
                    acc = Vector.FusedMultiplyAdd(new Vector<float>(a, m * K + k), new Vector<float>(b, n * K + k), acc);
                float s = Vector.Sum(acc);
                for (; k < K; k++) s += a[m * K + k] * b[n * K + k];
                c[m * N + n] = s;
            }
        }
        return c;
    }    internal static float[] MmTransposedB82(float[] a, float[] b, int M, int K, int N)
    {
        // v3: 2 m-rows by 4 n-lanes share each k-step (8 FMA chains); the B
        // panel streams once per m-pair instead of once per row. Per-dot op
        // order matches v2 exactly, so twins must agree bitwise.
        var c = new float[M * N];
        int step = Vector<float>.Count;
        int m2 = (M / 2) * 2;
        int n4 = (N / 4) * 4;
        for (int m = 0; m < m2; m += 2)
        {
            int n = 0;
            for (; n < n4; n += 4)
            {
                var a00 = Vector<float>.Zero; var a01 = Vector<float>.Zero;
                var a02 = Vector<float>.Zero; var a03 = Vector<float>.Zero;
                var a10 = Vector<float>.Zero; var a11 = Vector<float>.Zero;
                var a12 = Vector<float>.Zero; var a13 = Vector<float>.Zero;
                int k = 0;
                for (; k + step <= K; k += step)
                {
                    var av0 = new Vector<float>(a, m * K + k);
                    var av1 = new Vector<float>(a, (m + 1) * K + k);
                    var b0 = new Vector<float>(b, (n + 0) * K + k);
                    var b1 = new Vector<float>(b, (n + 1) * K + k);
                    var b2 = new Vector<float>(b, (n + 2) * K + k);
                    var b3 = new Vector<float>(b, (n + 3) * K + k);
                    a00 = Vector.FusedMultiplyAdd(av0, b0, a00);
                    a01 = Vector.FusedMultiplyAdd(av0, b1, a01);
                    a02 = Vector.FusedMultiplyAdd(av0, b2, a02);
                    a03 = Vector.FusedMultiplyAdd(av0, b3, a03);
                    a10 = Vector.FusedMultiplyAdd(av1, b0, a10);
                    a11 = Vector.FusedMultiplyAdd(av1, b1, a11);
                    a12 = Vector.FusedMultiplyAdd(av1, b2, a12);
                    a13 = Vector.FusedMultiplyAdd(av1, b3, a13);
                }
                float s00 = Vector.Sum(a00), s01 = Vector.Sum(a01), s02 = Vector.Sum(a02), s03 = Vector.Sum(a03);
                float s10 = Vector.Sum(a10), s11 = Vector.Sum(a11), s12 = Vector.Sum(a12), s13 = Vector.Sum(a13);
                for (; k < K; k++)
                {
                    float x0 = a[m * K + k], x1 = a[(m + 1) * K + k];
                    s00 += x0 * b[(n + 0) * K + k]; s01 += x0 * b[(n + 1) * K + k];
                    s02 += x0 * b[(n + 2) * K + k]; s03 += x0 * b[(n + 3) * K + k];
                    s10 += x1 * b[(n + 0) * K + k]; s11 += x1 * b[(n + 1) * K + k];
                    s12 += x1 * b[(n + 2) * K + k]; s13 += x1 * b[(n + 3) * K + k];
                }
                c[m * N + n] = s00; c[m * N + n + 1] = s01; c[m * N + n + 2] = s02; c[m * N + n + 3] = s03;
                c[(m + 1) * N + n] = s10; c[(m + 1) * N + n + 1] = s11; c[(m + 1) * N + n + 2] = s12; c[(m + 1) * N + n + 3] = s13;
            }
            for (; n < N; n++)
            {
                for (int dm = 0; dm < 2; dm++)
                {
                    var acc = Vector<float>.Zero;
                    int k = 0;
                    for (; k + step <= K; k += step)
                        acc = Vector.FusedMultiplyAdd(new Vector<float>(a, (m + dm) * K + k), new Vector<float>(b, n * K + k), acc);
                    float s = Vector.Sum(acc);
                    for (; k < K; k++) s += a[(m + dm) * K + k] * b[n * K + k];
                    c[(m + dm) * N + n] = s;
                }
            }
        }
        for (int m = m2; m < M; m++)
        {
            int n = 0;
            for (; n < n4; n += 4)
            {
                var acc0 = Vector<float>.Zero; var acc1 = Vector<float>.Zero;
                var acc2 = Vector<float>.Zero; var acc3 = Vector<float>.Zero;
                int k = 0;
                for (; k + step <= K; k += step)
                {
                    var av = new Vector<float>(a, m * K + k);
                    acc0 = Vector.FusedMultiplyAdd(av, new Vector<float>(b, (n + 0) * K + k), acc0);
                    acc1 = Vector.FusedMultiplyAdd(av, new Vector<float>(b, (n + 1) * K + k), acc1);
                    acc2 = Vector.FusedMultiplyAdd(av, new Vector<float>(b, (n + 2) * K + k), acc2);
                    acc3 = Vector.FusedMultiplyAdd(av, new Vector<float>(b, (n + 3) * K + k), acc3);
                }
                float s0 = Vector.Sum(acc0), s1 = Vector.Sum(acc1), s2 = Vector.Sum(acc2), s3 = Vector.Sum(acc3);
                for (; k < K; k++)
                {
                    float x = a[m * K + k];
                    s0 += x * b[(n + 0) * K + k]; s1 += x * b[(n + 1) * K + k];
                    s2 += x * b[(n + 2) * K + k]; s3 += x * b[(n + 3) * K + k];
                }
                c[m * N + n] = s0; c[m * N + n + 1] = s1; c[m * N + n + 2] = s2; c[m * N + n + 3] = s3;
            }
            for (; n < N; n++)
            {
                var acc = Vector<float>.Zero;
                int k = 0;
                for (; k + step <= K; k += step)
                    acc = Vector.FusedMultiplyAdd(new Vector<float>(a, m * K + k), new Vector<float>(b, n * K + k), acc);
                float s = Vector.Sum(acc);
                for (; k < K; k++) s += a[m * K + k] * b[n * K + k];
                c[m * N + n] = s;
            }
        }
        return c;
    }
    internal static unsafe float[] MmTransposedB82u(float[] a, float[] b, int M, int K, int N)
    {
        // v4a: unsafe-pointer port of v3 (H4: per-step bounds checks and
        // array-based vector ctors). Identical FMA and tail order per dot,
        // so results must agree bitwise with v2/v3, not just at 1e-4.
        var c = new float[M * N];
        int step = Vector<float>.Count;
        int m2 = (M / 2) * 2;
        int n4 = (N / 4) * 4;
        fixed (float* ap = a, bp = b, cp = c)
        {
            for (int m = 0; m < m2; m += 2)
            {
                int n = 0;
                for (; n < n4; n += 4)
                {
                    var a00 = Vector<float>.Zero; var a01 = Vector<float>.Zero;
                    var a02 = Vector<float>.Zero; var a03 = Vector<float>.Zero;
                    var a10 = Vector<float>.Zero; var a11 = Vector<float>.Zero;
                    var a12 = Vector<float>.Zero; var a13 = Vector<float>.Zero;
                    float* a0r = ap + m * K;
                    float* a1r = ap + (m + 1) * K;
                    float* b0r = bp + (n + 0) * K;
                    float* b1r = bp + (n + 1) * K;
                    float* b2r = bp + (n + 2) * K;
                    float* b3r = bp + (n + 3) * K;
                    int k = 0;
                    for (; k + step <= K; k += step)
                    {
                        var av0 = new Vector<float>(new ReadOnlySpan<float>(a0r + k, step));
                        var av1 = new Vector<float>(new ReadOnlySpan<float>(a1r + k, step));
                        var b0 = new Vector<float>(new ReadOnlySpan<float>(b0r + k, step));
                        var b1 = new Vector<float>(new ReadOnlySpan<float>(b1r + k, step));
                        var b2 = new Vector<float>(new ReadOnlySpan<float>(b2r + k, step));
                        var b3 = new Vector<float>(new ReadOnlySpan<float>(b3r + k, step));
                        a00 = Vector.FusedMultiplyAdd(av0, b0, a00);
                        a01 = Vector.FusedMultiplyAdd(av0, b1, a01);
                        a02 = Vector.FusedMultiplyAdd(av0, b2, a02);
                        a03 = Vector.FusedMultiplyAdd(av0, b3, a03);
                        a10 = Vector.FusedMultiplyAdd(av1, b0, a10);
                        a11 = Vector.FusedMultiplyAdd(av1, b1, a11);
                        a12 = Vector.FusedMultiplyAdd(av1, b2, a12);
                        a13 = Vector.FusedMultiplyAdd(av1, b3, a13);
                    }
                    float s00 = Vector.Sum(a00), s01 = Vector.Sum(a01), s02 = Vector.Sum(a02), s03 = Vector.Sum(a03);
                    float s10 = Vector.Sum(a10), s11 = Vector.Sum(a11), s12 = Vector.Sum(a12), s13 = Vector.Sum(a13);
                    for (; k < K; k++)
                    {
                        float x0 = a0r[k], x1 = a1r[k];
                        s00 += x0 * b0r[k]; s01 += x0 * b1r[k];
                        s02 += x0 * b2r[k]; s03 += x0 * b3r[k];
                        s10 += x1 * b0r[k]; s11 += x1 * b1r[k];
                        s12 += x1 * b2r[k]; s13 += x1 * b3r[k];
                    }
                    cp[m * N + n] = s00; cp[m * N + n + 1] = s01; cp[m * N + n + 2] = s02; cp[m * N + n + 3] = s03;
                    cp[(m + 1) * N + n] = s10; cp[(m + 1) * N + n + 1] = s11; cp[(m + 1) * N + n + 2] = s12; cp[(m + 1) * N + n + 3] = s13;
                }
                for (; n < N; n++)
                {
                    float* bnr = bp + n * K;
                    for (int dm = 0; dm < 2; dm++)
                    {
                        float* amr = ap + (m + dm) * K;
                        var acc = Vector<float>.Zero;
                        int k = 0;
                        for (; k + step <= K; k += step)
                            acc = Vector.FusedMultiplyAdd(new Vector<float>(new ReadOnlySpan<float>(amr + k, step)), new Vector<float>(new ReadOnlySpan<float>(bnr + k, step)), acc);
                        float st = Vector.Sum(acc);
                        for (; k < K; k++) st += amr[k] * bnr[k];
                        cp[(m + dm) * N + n] = st;
                    }
                }
            }
            for (int m = m2; m < M; m++)
            {
                float* amr = ap + m * K;
                int n = 0;
                for (; n < n4; n += 4)
                {
                    var acc0 = Vector<float>.Zero; var acc1 = Vector<float>.Zero;
                    var acc2 = Vector<float>.Zero; var acc3 = Vector<float>.Zero;
                    float* b0r = bp + (n + 0) * K;
                    float* b1r = bp + (n + 1) * K;
                    float* b2r = bp + (n + 2) * K;
                    float* b3r = bp + (n + 3) * K;
                    int k = 0;
                    for (; k + step <= K; k += step)
                    {
                        var av = new Vector<float>(new ReadOnlySpan<float>(amr + k, step));
                        acc0 = Vector.FusedMultiplyAdd(av, new Vector<float>(new ReadOnlySpan<float>(b0r + k, step)), acc0);
                        acc1 = Vector.FusedMultiplyAdd(av, new Vector<float>(new ReadOnlySpan<float>(b1r + k, step)), acc1);
                        acc2 = Vector.FusedMultiplyAdd(av, new Vector<float>(new ReadOnlySpan<float>(b2r + k, step)), acc2);
                        acc3 = Vector.FusedMultiplyAdd(av, new Vector<float>(new ReadOnlySpan<float>(b3r + k, step)), acc3);
                    }
                    float s0 = Vector.Sum(acc0), s1 = Vector.Sum(acc1), s2 = Vector.Sum(acc2), s3 = Vector.Sum(acc3);
                    for (; k < K; k++)
                    {
                        float x = amr[k];
                        s0 += x * b0r[k]; s1 += x * b1r[k];
                        s2 += x * b2r[k]; s3 += x * b3r[k];
                    }
                    cp[m * N + n] = s0; cp[m * N + n + 1] = s1; cp[m * N + n + 2] = s2; cp[m * N + n + 3] = s3;
                }
                for (; n < N; n++)
                {
                    float* bnr = bp + n * K;
                    var acc = Vector<float>.Zero;
                    int k = 0;
                    for (; k + step <= K; k += step)
                        acc = Vector.FusedMultiplyAdd(new Vector<float>(new ReadOnlySpan<float>(amr + k, step)), new Vector<float>(new ReadOnlySpan<float>(bnr + k, step)), acc);
                    float st = Vector.Sum(acc);
                    for (; k < K; k++) st += amr[k] * bnr[k];
                    cp[m * N + n] = st;
                }
            }
        }
        return c;
    }
    internal static float[] MmTransposedB(float[] a, float[] b, int M, int K, int N)
    {
        var c = new float[M * N];
        int step = Vector<float>.Count;
        for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
        {
            var acc = Vector<float>.Zero;
            int k = 0;
            for (; k + step <= K; k += step)
                acc = Vector.FusedMultiplyAdd(new Vector<float>(a, m * K + k), new Vector<float>(b, n * K + k), acc);
            float s = Vector.Sum(acc);
            for (; k < K; k++) s += a[m * K + k] * b[n * K + k];
            c[m * N + n] = s;
        }
        return c;
    }
    static float[] Rand(int n, Random rnd)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)rnd.NextDouble() * 2f - 1f;
        return a;
    }
    static int CheckTr(string name, int M, int K, int N, int seed)
    {
        // Reference runs the production path: reversed-stride view of a
        // row-major [N,K] buffer through Tensor MatMul, densify included.
        var rnd = new Random(seed);
        float[] a = Rand(M * K, rnd);
        float[] bRowMajor = Rand(N * K, rnd);
        var da = DenseTensor<float>.OfValues(a.AsSpan(), new int[] { M, K });
        var view = new DenseTensor<float>(new Memory<float>(bRowMajor), new int[] { K, N }, true);
        float[] refer = Tensor<float>.MatMul(da, view).ToDenseTensor().Buffer.ToArray();
        float[] cand = MmTransposedB(a, bRowMajor, M, K, N);
        float[] cand4 = MmTransposedB4(a, bRowMajor, M, K, N);
        float[] cand82 = MmTransposedB82(a, bRowMajor, M, K, N);
        double worst = 0;
        for (int i = 0; i < refer.Length; i++)
        {
            double ds = Math.Abs(cand[i] - refer[i]) / (1.0 + Math.Abs(refer[i]));
            if (ds > worst) worst = ds;
        }
        double worst4 = 0;
        for (int i = 0; i < refer.Length; i++)
        {
            double ds4 = Math.Abs(cand4[i] - refer[i]) / (1.0 + Math.Abs(refer[i]));
            if (ds4 > worst4) worst4 = ds4;
        }
        bool ok4 = worst4 <= 1e-4;
        Console.WriteLine("case " + name + " v2 maxScaled=" + worst4.ToString("E2") + (ok4 ? " PASS" : " FAIL"));

        double worst82 = 0;
        for (int i = 0; i < refer.Length; i++)
        {
            double d82 = System.Math.Abs(cand82[i] - refer[i]) / (1.0 + System.Math.Abs(refer[i]));
            if (d82 > worst82) worst82 = d82;
        }
        bool ok82 = worst82 <= 1e-4;
        Console.WriteLine("case " + name + " v3 maxScaled=" + worst82.ToString("E2") + (ok82 ? " PASS" : " FAIL"));
        float[] cand82u = MmTransposedB82u(a, bRowMajor, M, K, N);
        double worst82u = 0;
        for (int i = 0; i < refer.Length; i++)
        {
            double d82u = System.Math.Abs(cand82u[i] - refer[i]) / (1.0 + System.Math.Abs(refer[i]));
            if (d82u > worst82u) worst82u = d82u;
        }
        bool ok82u = worst82u <= 1e-4;
        bool bitU = cand82u.AsSpan().SequenceEqual(cand4);
        Console.WriteLine("case " + name + " v4a maxScaled=" + worst82u.ToString("E2") + " bitwise-v2=" + bitU + (ok82u && bitU ? " PASS" : " FAIL"));
        bool ok = worst <= 1e-4 && ok4 && ok82 && ok82u && cand82u.AsSpan().SequenceEqual(cand4);
        Console.WriteLine("case " + name + ": maxScaled=" + worst.ToString("E2") + (ok ? " PASS" : " FAIL"));
        return ok ? 0 : 1;
    }
    internal static int RunTrRead(string[] args)
    {
        if (args.Length >= 1 && args[0] == "bench")
        {
            int reps = 9;
            for (int i = 1; i < args.Length; i++)
            {
                if (args[i] == "--reps" && i + 1 < args.Length && int.TryParse(args[i + 1], out int k) && k >= 1) { reps = k; i++; }
                else { Console.WriteLine("usage: Bench trread verify|bench [--reps K]"); return 2; }
            }
            return RunBench(reps);
        }
        if (args.Length != 1 || args[0] != "verify")
        {
            Console.WriteLine("usage: Bench trread verify|bench [--reps K]");
            return 2;
        }
        int rc = 0;
        rc |= CheckTr("e5key-360x32x30", 360, 32, 30, 201);
        rc |= CheckTr("dinokey-1206x64x201", 1206, 64, 201, 202);
        rc |= CheckTr("gptkey-144x64x4", 144, 64, 4, 203);
        rc |= CheckTr("tail-7x13x11", 7, 13, 11, 204);
        rc |= CheckTr("narrow-48x20x30", 48, 20, 30, 205);
        Console.WriteLine(rc == 0 ? "trread verify: all cases agree." : "trread verify: FAILED.");
        return rc;
    }
    static void BenchShape(string name, int M, int K, int N, int reps)
    {
        var rnd = new Random(999);
        float[] a = Rand(M * K, rnd);
        float[] bRowMajor = Rand(N * K, rnd);
        var da = DenseTensor<float>.OfValues(a.AsSpan(), new int[] { M, K });
        var view = new DenseTensor<float>(new Memory<float>(bRowMajor), new int[] { K, N }, true);
        Tensor<float>.MatMul(da, view);
        MmTransposedB(a, bRowMajor, M, K, N);
        var tR = new double[reps];
        var tV1 = new double[reps];
        var tV2 = new double[reps];
        var tV3 = new double[reps];
        var tV4 = new double[reps];
        var sw = new System.Diagnostics.Stopwatch();
        for (int r = 0; r < reps; r++)
        {
            sw.Restart(); Tensor<float>.MatMul(da, view); sw.Stop(); tR[r] = sw.Elapsed.TotalMilliseconds;
            sw.Restart(); MmTransposedB(a, bRowMajor, M, K, N); sw.Stop(); tV1[r] = sw.Elapsed.TotalMilliseconds;
            sw.Restart(); MmTransposedB4(a, bRowMajor, M, K, N); sw.Stop(); tV2[r] = sw.Elapsed.TotalMilliseconds;
            sw.Restart(); MmTransposedB82(a, bRowMajor, M, K, N); sw.Stop(); tV3[r] = sw.Elapsed.TotalMilliseconds;
            sw.Restart(); MmTransposedB82u(a, bRowMajor, M, K, N); sw.Stop(); tV4[r] = sw.Elapsed.TotalMilliseconds;
        }
        Array.Sort(tR);
        Array.Sort(tV1);
        Array.Sort(tV2);
        Array.Sort(tV3);
        Array.Sort(tV4);
        Console.WriteLine(name + " ref(densify+matmul) best=" + tR[0].ToString("F2") + "ms median=" + tR[reps / 2].ToString("F2") + "ms");
        Console.WriteLine(name + " trread-v1 best=" + tV1[0].ToString("F2") + "ms median=" + tV1[reps / 2].ToString("F2") + "ms");
        Console.WriteLine(name + " trread-v3 best=" + tV3[0].ToString("F2") + "ms median=" + tV3[reps / 2].ToString("F2") + "ms");
        Console.WriteLine(name + " trread-v2 best=" + tV2[0].ToString("F2") + "ms median=" + tV2[reps / 2].ToString("F2") + "ms");
        Console.WriteLine(name + " trread-v4a best=" + tV4[0].ToString("F2") + "ms median=" + tV4[reps / 2].ToString("F2") + "ms");
    }
    static int RunBench(int reps)
    {
        Console.WriteLine("hardware: VectorCount=" + Vector<float>.Count + " accelerated=" + Vector.IsHardwareAccelerated);
        BenchShape("e5key-360x32x30", 360, 32, 30, reps);
        BenchShape("dinokey-1206x64x201", 1206, 64, 201, reps);
        BenchShape("gptkey-144x64x4", 144, 64, 4, reps);
        return 0;
    }
}