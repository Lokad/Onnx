namespace Lokad.Onnx.Bench;

using System;
using System.Collections.Generic;

// W0 paired-runner primitive (I1): every timed repetition executes public,
// reusable-context, and ORT work as interleaved pairs sharing one ORT
// control, instead of running all reusable-context iterations in a block
// before the alternating section. The schedule is pure data: TimedRun
// executes it in order, unit tests pin its shape, and the manifest records
// which run id produced every sample.
enum PairedRole
{
    PublicWarm,
    OrtWarm,
    PublicTimed,
    OrtTimed,
    CtxTimed,
}

readonly struct PairedStep
{
    public PairedStep(int runId, int pairId, PairedRole role, bool ortFirst)
    {
        RunId = runId;
        PairId = pairId;
        Role = role;
        OrtFirst = ortFirst;
    }

    public int RunId { get; }

    // Warm steps carry -1. Experiment C (context) owns 0..iters-1,
    // experiment P (public) owns iters..2*iters-1.
    public int PairId { get; }

    public bool IsWarm => PairId < 0;

    public PairedRole Role { get; }

    // For timed pairs: which side of this pair runs first. Warms ignore it.
    public bool OrtFirst { get; }
}

static class PairedModelRunner
{
    // Optional manifest path prefix (set once from --manifest-out). Each
    // TimedRun appends its case name and writes one versioned JSON file.
    // Empty (the default) writes nothing and changes no console output.
    public static string ManifestPrefix { get; set; } = "";

    // Warmups alternate public/ORT starting with public. Experiment C then
    // alternates context/ORT pairs and experiment P public/ORT pairs, with
    // the starting side alternating per pair in both experiments.
    public static List<PairedStep> BuildSchedule(int warmups, int iters)
    {
        if (warmups < 1) throw new ArgumentOutOfRangeException(nameof(warmups));
        if (iters < 1) throw new ArgumentOutOfRangeException(nameof(iters));
        var steps = new List<PairedStep>(2 * warmups + 4 * iters);
        int run = 0;
        for (int w = 0; w < warmups; w++)
        {
            steps.Add(new PairedStep(run++, -1, PairedRole.PublicWarm, false));
            steps.Add(new PairedStep(run++, -1, PairedRole.OrtWarm, true));
        }
        for (int e = 0; e < 2; e++)
        {
            bool isContext = e == 0;
            for (int i = 0; i < iters; i++)
            {
                bool ortFirst = (i % 2 == 0);
                int pair = (isContext ? 0 : iters) + i;
                PairedRole first = ortFirst ? PairedRole.OrtTimed :
                    (isContext ? PairedRole.CtxTimed : PairedRole.PublicTimed);
                PairedRole second = ortFirst ?
                    (isContext ? PairedRole.CtxTimed : PairedRole.PublicTimed) : PairedRole.OrtTimed;
                steps.Add(new PairedStep(run++, pair, first, ortFirst));
                steps.Add(new PairedStep(run++, pair, second, ortFirst));
            }
        }
        return steps;
    }

    public static PairedSample Sample(PairedStep step, string experiment, double ms, long ticks) =>
        new PairedSample { RunId = step.RunId, PairId = step.PairId, Role = step.Role.ToString(), Experiment = experiment, Ms = ms, Ticks = ticks };

    public static string SanitizeFileName(string name)
    {
        if (string.IsNullOrEmpty(name)) return "case";
        var bad = System.IO.Path.GetInvalidFileNameChars();
        var chars = name.ToCharArray();
        for (int i = 0; i < chars.Length; i++)
            if (System.Array.IndexOf(bad, chars[i]) >= 0) chars[i] = '_';
        return new string(chars);
    }
}
