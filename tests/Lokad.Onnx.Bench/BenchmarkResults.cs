namespace Lokad.Onnx.Bench;

using System;
using System.Collections.Generic;
using System.Linq;

// One timed sample: which schedule step produced it and what it cost.
// Experiment is "C" (reusable context) or "P" (public API); Role names the
// PairedRole. Ms duplicates the console raw-sample value for that run id.
sealed class PairedSample
{
    public int RunId { get; set; }

    public int PairId { get; set; }

    public string Role { get; set; } = "";

    public string Experiment { get; set; } = "";

    public double Ms { get; set; }

    public long Ticks { get; set; }
}

static class BenchmarkResults
{
    // Upper median matching the console Dist/MedLong convention.
    public static double Median(IEnumerable<double> values)
    {
        if (values is null) throw new ArgumentNullException(nameof(values));
        var ts = values.OrderBy(x => x).ToArray();
        if (ts.Length == 0) throw new ArgumentOutOfRangeException(nameof(values), "empty sample.");
        return ts[ts.Length / 2];
    }
}
