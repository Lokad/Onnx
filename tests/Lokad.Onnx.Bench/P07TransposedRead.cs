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
        double worst = 0;
        for (int i = 0; i < refer.Length; i++)
        {
            double ds = Math.Abs(cand[i] - refer[i]) / (1.0 + Math.Abs(refer[i]));
            if (ds > worst) worst = ds;
        }
        bool ok = worst <= 1e-4;
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
        var tB = new double[reps];
        var sw = new System.Diagnostics.Stopwatch();
        for (int r = 0; r < reps; r++)
        {
            sw.Restart(); Tensor<float>.MatMul(da, view); sw.Stop(); tR[r] = sw.Elapsed.TotalMilliseconds;
            sw.Restart(); MmTransposedB(a, bRowMajor, M, K, N); sw.Stop(); tB[r] = sw.Elapsed.TotalMilliseconds;
        }
        Array.Sort(tR);
        Array.Sort(tB);
        Console.WriteLine(name + " ref(densify+matmul) best=" + tR[0].ToString("F2") + "ms median=" + tR[reps / 2].ToString("F2") + "ms");
        Console.WriteLine(name + " trread best=" + tB[0].ToString("F2") + "ms median=" + tB[reps / 2].ToString("F2") + "ms");
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