namespace Lokad.Onnx.Backend.Tests;

using Lokad.Onnx.Bench;

/// <summary>
/// Pins the W0 paired-runner schedule: warmups alternate starting public,
/// experiments C (context) then P (public) alternate pair starting sides,
/// run ids are dense, and the manifest round-trips. TimedRun executes this
/// exact schedule; any shape change here must match a TimedRun change.
/// </summary>
public class PairedRunnerTests
{
    [Fact]
    public void Schedule_HasExpectedCountsAndOrder()
    {
        var steps = PairedModelRunner.BuildSchedule(3, 7);
        Assert.Equal(2 * 3 + 4 * 7, steps.Count);
        int at = 0;
        for (int w = 0; w < 3; w++)
        {
            Assert.True(steps[at].IsWarm);
            Assert.Equal(PairedRole.PublicWarm, steps[at++].Role);
            Assert.True(steps[at].IsWarm);
            Assert.Equal(PairedRole.OrtWarm, steps[at++].Role);
        }
        for (int i = 0; i < 7; i++)
        {
            var rc = steps[at].Role;
            Assert.True(rc == PairedRole.CtxTimed || rc == PairedRole.OrtTimed, "C experiment holds ctx/ORT only.");
            at += 2;
        }
        for (int i = 0; i < 7; i++)
        {
            var r = steps[at].Role;
            Assert.True(r == PairedRole.PublicTimed || r == PairedRole.OrtTimed, "P experiment holds public/ORT only.");
            at += 2;
        }
        Assert.Equal(steps.Count, at);
    }

    [Fact]
    public void Schedule_PairsShareIdsAndAlternateStartingSides()
    {
        var steps = PairedModelRunner.BuildSchedule(2, 5);
        var seen = new System.Collections.Generic.HashSet<int>();
        int at = 2 * 2;
        for (int e = 0; e < 2; e++)
        {
            for (int i = 0; i < 5; i++)
            {
                var first = steps[at++];
                var second = steps[at++];
                Assert.Equal(first.PairId, second.PairId);
                Assert.True(seen.Add(first.PairId), "pair ids repeat across pairs.");
                Assert.NotEqual(first.Role, second.Role);
                bool ortFirst = (i % 2 == 0);
                Assert.Equal(ortFirst, first.OrtFirst);
                Assert.Equal(ortFirst, second.OrtFirst);
                Assert.Equal(ortFirst ? PairedRole.OrtTimed : (e == 0 ? PairedRole.CtxTimed : PairedRole.PublicTimed), first.Role);
            }
        }
        int run = 0;
        foreach (var s in steps) Assert.Equal(run++, s.RunId);
    }

    [Fact]
    public void Schedule_RejectsInvalidCounts()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => PairedModelRunner.BuildSchedule(0, 5));
        Assert.Throws<ArgumentOutOfRangeException>(() => PairedModelRunner.BuildSchedule(2, 0));
    }

    [Fact]
    public void Sample_CarriesStepIdentity()
    {
        var steps = PairedModelRunner.BuildSchedule(1, 2);
        var step = steps[2];
        var s = PairedModelRunner.Sample(step, "C", 12.5, 987654321L);
        Assert.Equal(step.RunId, s.RunId);
        Assert.Equal(step.PairId, s.PairId);
        Assert.Equal(step.Role.ToString(), s.Role);
        Assert.Equal("C", s.Experiment);
        Assert.Equal(12.5, s.Ms);
        Assert.Equal(987654321L, s.Ticks);
    }

    [Fact]
    public void Manifest_RoundTripsWithSamples()
    {
        var manifest = new BenchmarkManifest
        {
            Case = "parakeet-decoder",
            LokDesc = "single-cpu-auto-1",
            OrtDesc = "intraop=1",
            Iters = 2,
            Warmup = 1,
            StartedUtc = "2026-09-18T00:00:00.0000000Z",
            Samples = new System.Collections.Generic.List<PairedSample>
            {
                new PairedSample { RunId = 0, PairId = -1, Role = "PublicWarm", Experiment = "W", Ms = 1.25, Ticks = 100 },
                new PairedSample { RunId = 3, PairId = 0, Role = "CtxTimed", Experiment = "C", Ms = 12.5, Ticks = 200 },
            },
        };
        var parsed = BenchmarkManifest.ReadJson(BenchmarkManifest.ToJson(manifest));
        Assert.NotNull(parsed);
        Assert.Equal(1, parsed.SchemaVersion);
        Assert.Equal("parakeet-decoder", parsed.Case);
        Assert.Equal(2, parsed.Samples.Count);
        Assert.Equal(12.5, parsed.Samples[1].Ms);
        Assert.Equal("CtxTimed", parsed.Samples[1].Role);
    }

    [Fact]
    public void Results_MedianMatchesConsoleConvention()
    {
        Assert.Equal(2.0, BenchmarkResults.Median(new double[] { 1.0, 3.0, 2.0 }));
        Assert.Equal(2.0, BenchmarkResults.Median(new double[] { 2.0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => BenchmarkResults.Median(new double[0]));
    }

    [Fact]
    public void SanitizeFileName_RemovesSeparators()
    {
        Assert.Equal("a_b_c", PairedModelRunner.SanitizeFileName("a/b:c"));
        Assert.Equal("parakeet-decoder", PairedModelRunner.SanitizeFileName("parakeet-decoder"));
        Assert.Equal("case", PairedModelRunner.SanitizeFileName(""));
    }
}
