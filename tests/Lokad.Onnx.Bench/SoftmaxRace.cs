namespace Lokad.Onnx.Bench;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection;

/// <summary>
/// E73 region race: row-pair softmax twins against the dispatched single-row
/// span kernels at E5 attention shapes. Measurement-only Bench verb; prints
/// parseable median lines plus agreement. The span kernels are internal to
/// the core, so this verb binds them once via reflection (loud refusal if a
/// name is missing) and times direct delegates.
/// </summary>
internal static class SoftmaxRace
{
    delegate void MaskedKernel(Span<float> xs, Span<float> mask, Span<float> ys, int outer, int block, bool useSimd);
    delegate void PlainKernel(Span<float> xs, Span<float> ys, int outer, int block, bool useSimd);

    static T Bind<T>(string name) where T : Delegate
    {
        var tensorFloat = typeof(DenseTensor<float>).Assembly.GetType("Lokad.Onnx.Tensor`1")!.MakeGenericType(typeof(float));
        var m = tensorFloat.GetMethod(name, BindingFlags.Static | BindingFlags.NonPublic);
        if (m is null) { Console.WriteLine("softmaxrace: kernel " + name + " not found; refusing."); Environment.Exit(1); throw new InvalidOperationException(name); }
        return (T)m.CreateDelegate(typeof(T));
    }

    internal static int RunSoftmaxRace(string[] args)
    {
        int cpu = 0, reps = 11;
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--cpu" && i + 1 < args.Length && int.TryParse(args[i + 1], out var c)) { cpu = c; i++; }
            else if (args[i] == "--reps" && i + 1 < args.Length && int.TryParse(args[i + 1], out var k) && k >= 1) { reps = k; i++; }
            else { Console.WriteLine("usage: Bench softmaxrace [--cpu N] [--reps K]"); return 2; }
        }
        if (!System.Runtime.Intrinsics.X86.Fma.IsSupported) { Console.WriteLine("softmaxrace: x86 FMA not available; refusing."); return 1; }
        global::Bench.EnforceSingleCpuAffinity(cpu);
        var mRef = Bind<MaskedKernel>("SoftmaxMaskedFloatSpan");
        var m4 = Bind<MaskedKernel>("SoftmaxMaskedFloatSpan4x");
        var mP = Bind<MaskedKernel>("SoftmaxMaskedFloatSpanPtr");
        var m2 = Bind<MaskedKernel>("SoftmaxMaskedFloatSpan2x");
        var pRef = Bind<PlainKernel>("SoftmaxContiguousFloatSpan");
        var p4 = Bind<PlainKernel>("SoftmaxContiguousFloatSpan4x");
        var pP = Bind<PlainKernel>("SoftmaxContiguousFloatSpanPtr");
        var p2 = Bind<PlainKernel>("SoftmaxContiguousFloatSpan2x");
        var shapes = new (int Outer, int Block)[]
        {
            (96, 8), (360, 30), (1536, 128),
            (12, 512), (3, 128), (5, 30), (4, 7), (7, 9),
        };
        var rnd = new Random(733);
        int rc = 0;
        foreach (var (outer, block) in shapes)
            rc |= RaceOne(outer, block, reps, rnd, mRef, m2, m4, mP, pRef, p2, p4, pP);
        Console.WriteLine(rc == 0 ? "softmaxrace: all cases agree." : "softmaxrace: FAILED.");
        return rc;
    }

    static int RaceOne(int outer, int block, int reps, Random rnd, MaskedKernel mRef, MaskedKernel m2, MaskedKernel m4, MaskedKernel mP, PlainKernel pRef, PlainKernel p2, PlainKernel p4, PlainKernel pP)
    {
        var x = new float[outer * block];
        var mask = new float[block];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rnd.NextDouble() * 2 - 1) * 6;
        for (int i = 0; i < mask.Length; i++) mask[i] = (float)(rnd.NextDouble() * 2 - 1);
        var yRef = new float[x.Length];
        var y2 = new float[x.Length];
        var y4 = new float[x.Length];
        double TimeM(MaskedKernel k, float[] y)
        {
            for (int w = 0; w < 5; w++) k(x, mask, y, outer, block, true);
            var sw = new Stopwatch();
            var ts = new List<double>(reps);
            for (int r = 0; r < reps; r++) { sw.Restart(); k(x, mask, y, outer, block, true); sw.Stop(); ts.Add(sw.Elapsed.TotalMilliseconds); }
            ts.Sort();
            return ts[ts.Count / 2];
        }
        double TimeP(PlainKernel k, float[] y)
        {
            for (int w = 0; w < 5; w++) k(x, y, outer, block, true);
            var sw = new Stopwatch();
            var ts = new List<double>(reps);
            for (int r = 0; r < reps; r++) { sw.Restart(); k(x, y, outer, block, true); sw.Stop(); ts.Add(sw.Elapsed.TotalMilliseconds); }
            ts.Sort();
            return ts[ts.Count / 2];
        }
        long BadBits(float[] a, float[] b)
        {
            long n = 0;
            for (int i = 0; i < a.Length; i++)
                if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i])) n++;
            return n;
        }
        double tmRef = TimeM(mRef, yRef);
        double tm2 = TimeM(m2, y2);
        long badM = BadBits(y2, yRef);
        double tm4 = TimeM(m4, y4);
        long badM4 = BadBits(y4, yRef);
        double tmP = TimeM(mP, y4);
        long badMP = BadBits(y4, yRef);
        double tpRef = TimeP(pRef, yRef);
        double tp2 = TimeP(p2, y2);
        long badP = BadBits(y2, yRef);
        double tp4 = TimeP(p4, y4);
        long badP4 = BadBits(y4, yRef);
        double tpP = TimeP(pP, y4);
        long badPP = BadBits(y4, yRef);
        Console.WriteLine("softmaxrace " + outer + "x" + block + ": masked-ref=" + tmRef.ToString("F3") + "ms masked-2x=" + tm2.ToString("F3") + "ms badM=" + badM + " masked-4x=" + tm4.ToString("F3") + "ms badM4=" + badM4 + " masked-ptr=" + tmP.ToString("F3") + "ms badMP=" + badMP + " plain-ref=" + tpRef.ToString("F3") + "ms plain-2x=" + tp2.ToString("F3") + "ms badP=" + badP + " plain-4x=" + tp4.ToString("F3") + "ms badP4=" + badP4 + " plain-ptr=" + tpP.ToString("F3") + "ms badPP=" + badPP);
        return (badM == 0 && badP == 0 && badM4 == 0 && badP4 == 0 && badMP == 0 && badPP == 0) ? 0 : 1;
    }
}