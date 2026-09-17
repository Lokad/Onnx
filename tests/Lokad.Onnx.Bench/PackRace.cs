namespace Lokad.Onnx.Bench;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Lokad.Onnx;

/// <summary>
/// E61 region race: 6-row packed twin against the proven 2-row (and 3-row)
/// packed kernels over prepared-format panels at E5 projection shapes.
/// Measurement-only Bench verb; prints parseable median lines plus agreement.
/// </summary>
internal static class PackRace
{
    internal static int RunPackRace(string[] args)
    {
        int cpu = 0, reps = 11;
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--cpu" && i + 1 < args.Length && int.TryParse(args[i + 1], out var c)) { cpu = c; i++; }
            else if (args[i] == "--reps" && i + 1 < args.Length && int.TryParse(args[i + 1], out var k) && k >= 1) { reps = k; i++; }
            else { Console.WriteLine("usage: Bench packrace [--cpu N] [--reps K]"); return 2; }
        }
        if (!System.Runtime.Intrinsics.X86.Fma.IsSupported) { Console.WriteLine("packrace: x86 FMA not available; refusing."); return 1; }
        global::Bench.EnforceSingleCpuAffinity(cpu);
        var shapes = new (int M, int N, int K)[]
        {
            (30, 384, 384), (30, 384, 1536), (30, 1536, 384),
            (126, 384, 384), (126, 384, 1536), (126, 1536, 384),
            (128, 384, 384), (128, 384, 1536), (128, 1536, 384),
            (8, 384, 384), (64, 384, 384), (30, 32, 384),
        };
        var rnd = new Random(611);
        int rc = 0;
        foreach (var (M, N, K) in shapes)
            rc |= RaceOne(M, N, K, reps, rnd);
        Console.WriteLine(rc == 0 ? "packrace: all cases agree." : "packrace: FAILED.");
        return rc;
    }

    static unsafe int RaceOne(int M, int N, int K, int reps, Random rnd)
    {
        // Kernels accumulate into C, so every timed call starts from a cleared
        // destination (equal clear cost inside every lambda). Non-multiples of 6
        // compose the twin with the proven packed tail, exactly like dispatch.
        var a = new float[M * N];
        var b = new float[N * K];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rnd.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rnd.NextDouble() * 2 - 1);
        var p = new float[N * K];
        var cRef = new float[M * K];
        var cNew = new float[M * K];
        fixed (float* bp = b, pp = p)
            MathOps.PackPanelsB(N, K, bp, pp);
        var ha = System.Runtime.InteropServices.GCHandle.Alloc(a, System.Runtime.InteropServices.GCHandleType.Pinned);
        var hp = System.Runtime.InteropServices.GCHandle.Alloc(p, System.Runtime.InteropServices.GCHandleType.Pinned);
        var hcR = System.Runtime.InteropServices.GCHandle.Alloc(cRef, System.Runtime.InteropServices.GCHandleType.Pinned);
        var hcN = System.Runtime.InteropServices.GCHandle.Alloc(cNew, System.Runtime.InteropServices.GCHandleType.Pinned);
        try
        {
            IntPtr ap = ha.AddrOfPinnedObject(), pp = hp.AddrOfPinnedObject();
            IntPtr cr = hcR.AddrOfPinnedObject(), cn = hcN.AddrOfPinnedObject();
            double Time(System.Action run)
            {
                for (int w = 0; w < 5; w++) run();
                var sw = new System.Diagnostics.Stopwatch();
                var ts = new System.Collections.Generic.List<double>(reps);
                for (int r = 0; r < reps; r++) { sw.Restart(); run(); sw.Stop(); ts.Add(sw.Elapsed.TotalMilliseconds); }
                ts.Sort();
                return ts[ts.Count / 2];
            }
            int six = M - (M % 6);
            IntPtr apT = ap + six * N * 4, cnT = cn + six * K * 4, crT = cr + six * K * 4;
            int rem = M - six;
            Action ref2 = () => { Array.Clear(cRef, 0, cRef.Length); MathOps.mm_unsafe_vectorized_intrinsics_2x4packed(M, N, K, (float*)ap, (float*)pp, (float*)cr); };
            Action twin = () => {
                Array.Clear(cNew, 0, cNew.Length);
                MathOps.mm_unsafe_vectorized_intrinsics_6x4packed(six, N, K, (float*)ap, (float*)pp, (float*)cn);
                if (rem == 3) MathOps.mm_unsafe_vectorized_intrinsics_3x4packed(rem, N, K, (float*)apT, (float*)pp, (float*)cnT);
                else if (rem > 0) MathOps.mm_unsafe_vectorized_intrinsics_2x4packed(rem, N, K, (float*)apT, (float*)pp, (float*)cnT);
            };
            double t2 = Time(ref2);
            double t6 = Time(twin);
            double worst = 0;
            for (int i = 0; i < cRef.Length; i++)
            {
                double d = Math.Abs(cRef[i] - cNew[i]);
                if (double.IsNaN(d)) { worst = double.NaN; break; }
                if (d > worst) worst = d;
            }
            string agree = double.IsNaN(worst) ? "NaN" : worst.ToString("E2");
            bool ok = worst == 0.0;
            string line = $"packrace M={M} N={N} K={K} ref2={t2:F3}ms twin6={t6:F3}ms speedup={t2 / t6:F2}x agree={agree} " + (ok ? "OK" : "FAIL");
            if (M % 3 == 0)
            {
                var c3 = new float[M * K];
                var hc3 = System.Runtime.InteropServices.GCHandle.Alloc(c3, System.Runtime.InteropServices.GCHandleType.Pinned);
                double t3;
                try
                {
                    IntPtr c3p = hc3.AddrOfPinnedObject();
                    t3 = Time(() => { Array.Clear(c3, 0, c3.Length); MathOps.mm_unsafe_vectorized_intrinsics_3x4packed(M, N, K, (float*)ap, (float*)pp, (float*)c3p); });
                }
                finally { hc3.Free(); }
                double w3 = 0;
                for (int i = 0; i < cRef.Length; i++) { double d = Math.Abs(cRef[i] - c3[i]); if (d > w3) w3 = d; }
                line += $" ref3={t3:F3}ms agree3={w3:E2}" + (w3 == 0.0 ? "" : " FAIL3");
                ok = ok && w3 == 0.0;
            }
            Console.WriteLine(line);
            return ok ? 0 : 1;
        }
        finally { ha.Free(); hp.Free(); hcR.Free(); hcN.Free(); }
    }
}


