namespace Lokad.Onnx.Bench;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection;

/// <summary>
/// E72 region race: two-way and four-way unrolled BiasGelu pointer twins
/// against the dispatched single-vector pointer kernel at E5 BiasGelu shapes.
/// Measurement-only Bench verb; prints parseable median lines plus agreement.
/// The span kernels are internal to the core, so this verb binds them once via
/// reflection (loud refusal if a name is missing) and times direct delegates.
/// </summary>
internal static class GeluRace
{
    delegate void SpanKernel(ReadOnlySpan<float> xs, ReadOnlySpan<float> bias, Span<float> ys);

    static SpanKernel Bind(string name)
    {
        var t = typeof(DenseTensor<float>);
        var asm = t.Assembly;
        var tensorFloat = asm.GetType("Lokad.Onnx.Tensor`1")!.MakeGenericType(typeof(float));
        var m = tensorFloat.GetMethod(name, BindingFlags.Static | BindingFlags.NonPublic);
        if (m is null) { Console.WriteLine("gelurace: kernel " + name + " not found; refusing."); Environment.Exit(1); throw new InvalidOperationException(name); }
        return (SpanKernel)m.CreateDelegate(typeof(SpanKernel));
    }

    internal static int RunGeluRace(string[] args)
    {
        int cpu = 0, reps = 11;
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--cpu" && i + 1 < args.Length && int.TryParse(args[i + 1], out var c)) { cpu = c; i++; }
            else if (args[i] == "--reps" && i + 1 < args.Length && int.TryParse(args[i + 1], out var k) && k >= 1) { reps = k; i++; }
            else { Console.WriteLine("usage: Bench gelurace [--cpu N] [--reps K]"); return 2; }
        }
        if (!System.Runtime.Intrinsics.X86.Fma.IsSupported) { Console.WriteLine("gelurace: x86 FMA not available; refusing."); return 1; }
        global::Bench.EnforceSingleCpuAffinity(cpu);
        var refK = Bind("BiasGeluSpanFloatPtr");
        var k2 = Bind("BiasGeluSpanFloatPtr2x");
        var k4 = Bind("BiasGeluSpanFloatPtr4x");
        var shapes = new (int Rows, int M)[]
        {
            (8, 384), (30, 384), (128, 384),
            (8, 1536), (30, 1536), (128, 1536),
            (512, 1536), (7, 384), (5, 64),
        };
        var rnd = new Random(722);
        int rc = 0;
        foreach (var (rows, m) in shapes)
            rc |= RaceOne(rows, m, reps, rnd, refK, k2, k4);
        Console.WriteLine(rc == 0 ? "gelurace: all cases agree." : "gelurace: FAILED.");
        return rc;
    }

    static int RaceOne(int rows, int m, int reps, Random rnd, SpanKernel refK, SpanKernel k2, SpanKernel k4)
    {
        var x = new float[rows * m];
        var bias = new float[m];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rnd.NextDouble() * 2 - 1) * 6;
        for (int i = 0; i < bias.Length; i++) bias[i] = (float)(rnd.NextDouble() * 2 - 1) * 2;
        var yRef = new float[x.Length];
        var y2 = new float[x.Length];
        var y4 = new float[x.Length];
        double Time(SpanKernel k, float[] y)
        {
            for (int w = 0; w < 5; w++) k(x, bias, y);
            var sw = new Stopwatch();
            var ts = new List<double>(reps);
            for (int r = 0; r < reps; r++) { sw.Restart(); k(x, bias, y); sw.Stop(); ts.Add(sw.Elapsed.TotalMilliseconds); }
            ts.Sort();
            return ts[ts.Count / 2];
        }
        double tRef = Time(refK, yRef);
        double t2 = Time(k2, y2);
        double t4 = Time(k4, y4);
        long BadBits(float[] a, float[] b)
        {
            long n = 0;
            for (int i = 0; i < a.Length; i++)
                if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i])) n++;
            return n;
        }
        long bad2 = BadBits(y2, yRef);
        long bad4 = BadBits(y4, yRef);
        Console.WriteLine("gelurace " + rows + "x" + m + ": ref=" + tRef.ToString("F3") + "ms 2x=" + t2.ToString("F3") + "ms 4x=" + t4.ToString("F3") + "ms bad2=" + bad2 + " bad4=" + bad4);
        return (bad2 == 0 && bad4 == 0) ? 0 : 1;
    }
}