namespace Lokad.Onnx.Bench;

using System;

// Tracked known divergences (PLAN.md C01). An entry records a case output that
// breaches the unchanged bench agreement gate from an attributed mechanism with
// no localizable defect, so the runner excludes the case explicitly instead of
// throwing a generic error. An entry never moves the gate and never permits
// timing: the matching breach skips every timed row. The tripwire fails anything
// at or above its ceiling as a fresh regression.
internal sealed record KnownDivergence(string CaseLabel, string OutputName, double TripwireScaled, string Tracking, string Reason)
{
    internal bool Matches(string caseLabel, string outputName, double measuredScaled, double gate) =>
        string.Equals(caseLabel, CaseLabel, StringComparison.Ordinal)
        && string.Equals(outputName, OutputName, StringComparison.Ordinal)
        && measuredScaled > gate
        && measuredScaled < TripwireScaled;
}

internal static class KnownDivergences
{
    internal static readonly KnownDivergence[] All = new KnownDivergence[]
    {
        new KnownDivergence(
            "dinov2-224",
            "last_hidden_state",
            1e-3,
            "C01",
            "DINOv2-small depth-amplified fp32 summation-order drift after bit-identical GELU fusion and tail order parity; uniform layer-0 seed with no localizable kernel defect."),
    };

    internal static bool TryMatch(string caseLabel, string outputName, double measuredScaled, double gate, out KnownDivergence match)
    {
        foreach (var known in All)
        {
            if (known.Matches(caseLabel, outputName, measuredScaled, gate))
            {
                match = known;
                return true;
            }
        }
        match = null!;
        return false;
    }
}

internal sealed class KnownDivergenceException : InvalidOperationException
{
    internal KnownDivergenceException(string caseLabel, string outputName, double measuredScaled, double gate, KnownDivergence known, Exception inner)
        : base(caseLabel + ":" + outputName + ": known divergence " + known.Tracking
            + " (measured reference-scaled diff " + measuredScaled.ToString("E2")
            + " vs gate " + gate.ToString("E2")
            + ", tripwire " + known.TripwireScaled.ToString("E2") + "): " + known.Reason
            + " Timings withheld; case excluded from publication.", inner)
    {
    }
}
