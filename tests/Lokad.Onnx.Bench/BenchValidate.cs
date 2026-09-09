namespace Lokad.Onnx.Bench;

using System;

// Shared benchmark validation: every timed pairing is proven element by
// element before timing. Each breach throws a descriptive error instead of
// timing incorrect results. In every comparison the first tensor is the
// reference and the second is the candidate under test; the scaled error
// divides by the reference magnitude, so argument order matters.
internal static class BenchValidate
{
    public static void RequireFinite(string what, float[] reference, float[] candidate)
    {
        for (int i = 0; i < reference.Length; i++)
        {
            if (!float.IsFinite(reference[i]))
                throw new InvalidOperationException(what + " reference: non-finite value " + reference[i] + " at index " + i + ".");
        }
        for (int i = 0; i < candidate.Length; i++)
        {
            if (!float.IsFinite(candidate[i]))
                throw new InvalidOperationException(what + " candidate: non-finite value " + candidate[i] + " at index " + i + ".");
        }
    }

    public static void RequireSameDims(string what, int[] referenceDims, int[] candidateDims)
    {
        if (referenceDims.Length != candidateDims.Length)
            throw new InvalidOperationException(what + ": rank mismatch (reference "
                + string.Join("x", referenceDims) + ", candidate " + string.Join("x", candidateDims) + ").");
        for (int i = 0; i < referenceDims.Length; i++)
        {
            if (referenceDims[i] != candidateDims[i])
                throw new InvalidOperationException(what + ": shape mismatch (reference "
                    + string.Join("x", referenceDims) + ", candidate " + string.Join("x", candidateDims) + ").");
        }
    }

    public static (double scaled, int scaledIndex, double abs, int absIndex) ScaledAndAbsDiff(float[] reference, float[] candidate)
    {
        if (reference.Length != candidate.Length)
            throw new InvalidOperationException("length mismatch (reference "
                + reference.Length + ", candidate " + candidate.Length + ").");
        double worstScaled = 0;
        int scaledIndex = -1;
        double worstAbs = 0;
        int absIndex = -1;
        for (int i = 0; i < reference.Length; i++)
        {
            double abs = Math.Abs((double)reference[i] - candidate[i]);
            if (abs > worstAbs) { worstAbs = abs; absIndex = i; }
            double scaled = abs / (1.0 + Math.Abs((double)reference[i]));
            if (scaled > worstScaled) { worstScaled = scaled; scaledIndex = i; }
        }
        return (worstScaled, scaledIndex, worstAbs, absIndex);
    }

    public static (double scaled, double abs) RequireAgreement(string what, int[] referenceDims, float[] reference, int[] candidateDims, float[] candidate, double tolerance)
    {
        RequireSameDims(what, referenceDims, candidateDims);
        RequireFinite(what, reference, candidate);
        var diff = ScaledAndAbsDiff(reference, candidate);
        if (diff.scaled > tolerance)
            throw new InvalidOperationException(what + ": outputs diverge (max reference-scaled diff "
                + diff.scaled.ToString("E2") + " at index " + diff.scaledIndex
                + " [ref=" + reference[diff.scaledIndex] + " cand=" + candidate[diff.scaledIndex] + "]"
                + " vs tolerance " + tolerance.ToString("E2")
                + "; max abs diff " + diff.abs.ToString("E2") + " at index " + diff.absIndex + ").");
        return (diff.scaled, diff.abs);
    }
}
