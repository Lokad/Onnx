using System;
using System.Numerics;
using System.Runtime.InteropServices;
using Lokad.Onnx;
// P05/M1 prototype: K-blocked GEMM against full-reduction panels, outside the product.
// Pack tiles B into (Kn reduction rows) by (Kb output cols) blocks, zero-padded at edges:
// padding is arithmetically neutral on finite values but not NaN-preserving, so
// agreement legs stay finite and NaN behavior is owned by the existing kernels.
static class P05GemmBlocked
{
    internal const int Kb = 32;
    internal const int Kn = 256;
    internal static float[] PackBK(float[] b, int N, int K, out int nKo, out int nNo)
    {
        nKo = (K + Kb - 1) / Kb;
        nNo = (N + Kn - 1) / Kn;
        var p = new float[nKo * nNo * Kn * Kb];
        for (int ko = 0; ko < nKo; ko++)
        for (int no = 0; no < nNo; no++)
        for (int jj = 0; jj < Kn; jj++)
        for (int kk = 0; kk < Kb; kk++)
        {
            int jr = no * Kn + jj;
            int kr = ko * Kb + kk;
            p[((ko * nNo) + no) * Kn * Kb + jj * Kb + kk] = (jr < N && kr < K) ? b[jr * K + kr] : 0f;
        }
        return p;
    }

    internal static float[] MmBlocked(float[] a, float[] packed, int M, int N, int K, int nKo, int nNo)
    {
        if ((M & 1) != 0) throw new ArgumentException("Blocked prototype needs even M.");
        if (Kb % Vector<float>.Count != 0) throw new ArgumentException("Blocked prototype needs Kb split by the vector width.");
        var c = new float[M * K];
        int step = Vector<float>.Count;
        int kbV = Kb / step;
        var wp = MemoryMarshal.Cast<float, Vector<float>>(packed.AsSpan());
        var acc0 = new Vector<float>[kbV];
        var acc1 = new Vector<float>[kbV];
        var tail0 = new float[Kb];
        var tail1 = new float[Kb];
        for (int i = 0; i < M; i += 2)
        for (int ko = 0; ko < nKo; ko++)
        {
            int kbC = Math.Min(Kb, K - ko * Kb);
            Array.Clear(acc0, 0, kbV);
            Array.Clear(acc1, 0, kbV);
            Array.Clear(tail0, 0, Kb);
            Array.Clear(tail1, 0, Kb);
            for (int no = 0; no < nNo; no++)
            {
                int knC = Math.Min(Kn, N - no * Kn);
                int tile = ((ko * nNo) + no) * Kn * Kb;
                for (int j = 0; j < knC; j++)
                {
                    float x0 = a[i * N + no * Kn + j];
                    float x1 = a[(i + 1) * N + no * Kn + j];
                    var x0V = new Vector<float>(x0);
                    var x1V = new Vector<float>(x1);
                    int wvi = tile / step + j * (Kb / step);
                    for (int k = 0; k * step + step <= kbC; k++)
                    {
                        acc0[k] = Vector.FusedMultiplyAdd(wp[wvi + k], x0V, acc0[k]);
                        acc1[k] = Vector.FusedMultiplyAdd(wp[wvi + k], x1V, acc1[k]);
                    }
                    for (int kk = (kbC / step) * step; kk < kbC; kk++)
                    {
                        float w = packed[tile + j * Kb + kk];
                        tail0[kk] += w * x0;
                        tail1[kk] += w * x1;
                    }
                }
            }
            for (int k = 0; k * step + step <= kbC; k++)
            {
                acc0[k].CopyTo(c, i * K + ko * Kb + k * step);
                acc1[k].CopyTo(c, (i + 1) * K + ko * Kb + k * step);
            }
            for (int kk = (kbC / step) * step; kk < kbC; kk++)
            {
                c[i * K + ko * Kb + kk] = tail0[kk];
                c[(i + 1) * K + ko * Kb + kk] = tail1[kk];
            }
        }
        return c;
    }
    static float[] RandG(int n, Random rnd)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)rnd.NextDouble() * 2f - 1f;
        return a;
    }
    static int CheckGemm(string name, int M, int N, int K, int seed)
    {
        var rnd = new Random(seed);
        float[] a = RandG(M * N, rnd);
        float[] b = RandG(N * K, rnd);
        var da = DenseTensor<float>.OfValues(a.AsSpan(), new int[] { M, N });
        var db = DenseTensor<float>.OfValues(b.AsSpan(), new int[] { N, K });
        float[] refer = Tensor<float>.MatMul(da, db).ToDenseTensor().Buffer.ToArray();
        float[] packed = PackBK(b, N, K, out int nKo, out int nNo);
        float[] cand = MmBlocked(a, packed, M, N, K, nKo, nNo);
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
    internal static int RunGemmBlock(string[] args)
    {
        if (args.Length >= 1 && args[0] == "bench")
        {
            int reps = 9;
            for (int i = 1; i < args.Length; i++)
            {
                if (args[i] == "--reps" && i + 1 < args.Length && int.TryParse(args[i + 1], out int k) && k >= 1) { reps = k; i++; }
                else { Console.WriteLine("usage: Bench gemmblock bench [--reps K]"); return 2; }
            }
            return RunBench(reps);
        }
        if (args.Length == 1 && args[0] == "stride")
        {
            return StrideTwin();
        }
        if (args.Length != 1 || args[0] != "verify")
        {
            Console.WriteLine("usage: Bench gemmblock verify|bench|stride");
            return 2;
        }
        int rc = 0;
        rc |= CheckGemm("e5-8tok-mlp", 8, 384, 1536, 101);
        rc |= CheckGemm("e5-30tok-mlp", 30, 384, 1536, 102);
        rc |= CheckGemm("e5-128tok-mlp", 128, 384, 1536, 103);
        rc |= CheckGemm("gpt2-32tok-mlp", 32, 768, 3072, 104);
        rc |= CheckGemm("gpt2-8tok-mlp", 8, 768, 3072, 105);
        rc |= CheckGemm("tail-6x100x70", 6, 100, 70, 106);
        Console.WriteLine(rc == 0 ? "gemmblock verify: all cases agree." : "gemmblock verify: FAILED.");
        return rc;
    }
    static void BenchShape(string name, int M, int N, int K, int reps)
    {
        var rnd = new Random(999);
        float[] a = RandG(M * N, rnd);
        float[] b = RandG(N * K, rnd);
        var da = DenseTensor<float>.OfValues(a.AsSpan(), new int[] { M, N });
        var db = DenseTensor<float>.OfValues(b.AsSpan(), new int[] { N, K });
        float[] packed = PackBK(b, N, K, out int nKo, out int nNo);
        Tensor<float>.MatMul(da, db);
        MmBlocked(a, packed, M, N, K, nKo, nNo);
        var tR = new double[reps];
        var tB = new double[reps];
        var sw = new System.Diagnostics.Stopwatch();
        for (int r = 0; r < reps; r++)
        {
            sw.Restart(); Tensor<float>.MatMul(da, db); sw.Stop(); tR[r] = sw.Elapsed.TotalMilliseconds;
            sw.Restart(); MmBlocked(a, packed, M, N, K, nKo, nNo); sw.Stop(); tB[r] = sw.Elapsed.TotalMilliseconds;
        }
        Array.Sort(tR);
        Array.Sort(tB);
        Console.WriteLine(name + " ref best=" + tR[0].ToString("F2") + "ms median=" + tR[reps / 2].ToString("F2") + "ms");
        Console.WriteLine(name + " blk best=" + tB[0].ToString("F2") + "ms median=" + tB[reps / 2].ToString("F2") + "ms");
    }
    static int RunBench(int reps)
    {
        Console.WriteLine("hardware: VectorCount=" + Vector<float>.Count + " accelerated=" + Vector.IsHardwareAccelerated);
        BenchShape("e5-8tok-mlp", 8, 384, 1536, reps);
        BenchShape("e5-128tok-mlp", 128, 384, 1536, reps);
        BenchShape("gpt2-32tok-mlp", 32, 768, 3072, reps);
        BenchShape("e5-512tok-mlp", 512, 384, 1536, reps);
        return 0;
    }
    static int StrideTwin()
    {
        const int M = 1, K = 768, N = 50257;
        var rnd = new Random(4242);
        float[] a = RandG(M * K, rnd);
        float[] wRowMajor = RandG(N * K, rnd);
        var da = DenseTensor<float>.OfValues(a.AsSpan(), new int[] { M, K });
        var view = new DenseTensor<float>(new Memory<float>(wRowMajor), new int[] { K, N }, true);
        Console.WriteLine("stride: view reversed=" + view.IsReversedStride + " buffer=" + view.Buffer.Length + " len=" + view.Length);
        var twin = new float[K * N];
        for (int k = 0; k < K; k++)
        for (int j = 0; j < N; j++) twin[k * N + j] = wRowMajor[j * K + k];
        var dt = DenseTensor<float>.OfValues(twin.AsSpan(), new int[] { K, N });
        float[] rv = Tensor<float>.MatMul(da, view).ToDenseTensor().Buffer.ToArray();
        float[] rt = Tensor<float>.MatMul(da, dt).ToDenseTensor().Buffer.ToArray();
        double worst = 0;
        for (int i = 0; i < rv.Length; i++)
        {
            double ds = Math.Abs(rv[i] - rt[i]) / (1.0 + Math.Abs(rt[i]));
            if (ds > worst) worst = ds;
        }
        Console.WriteLine("stride: agreement maxScaled=" + worst.ToString("E2"));
        if (worst > 1e-4) { Console.WriteLine("stride: twin MISMATCH, no timing."); return 1; }
        int reps = 9;
        var tV = new double[reps];
        var tT = new double[reps];
        var sw = new System.Diagnostics.Stopwatch();
        for (int r = 0; r < reps; r++)
        {
            sw.Restart(); Tensor<float>.MatMul(da, view); sw.Stop(); tV[r] = sw.Elapsed.TotalMilliseconds;
            sw.Restart(); Tensor<float>.MatMul(da, dt); sw.Stop(); tT[r] = sw.Elapsed.TotalMilliseconds;
        }
        Array.Sort(tV);
        Array.Sort(tT);
        Console.WriteLine("stride view best=" + tV[0].ToString("F2") + "ms median=" + tV[reps / 2].ToString("F2") + "ms");
        Console.WriteLine("stride twin best=" + tT[0].ToString("F2") + "ms median=" + tT[reps / 2].ToString("F2") + "ms");
        return 0;
    }
}
