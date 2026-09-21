"""Summarize closed historical GC attribution with its limits and refusals."""
from common import *


def main():
    proof, analysis = read(BASE / 'closed.json'), read(BASE / 'analysis.json')
    assert proof['passed'] and analysis['passed'] and proof['analysis'] == pin(BASE / 'analysis.json')
    verify(proof['files'])
    for identity in proof['identities']:
        terminal(identity)
    report, observations = TOOLS / 'results-20260922.md', TOOLS / 'observations-20260922.json'
    assert not report.exists() and not observations.exists()
    save(observations, dict(closure=pin(BASE / 'closed.json'), analysis=proof['analysis'],
        captures=[{key: value for key, value in capture.items() if key not in ['pauses', 'collections']}
            for capture in analysis['captures']], resources=analysis['resources'], resource_samples=analysis['resource_samples'],
        peak_rss=analysis['peak_rss'], identities=analysis['identities'], tests=analysis['tests'], scope=analysis['scope']))
    rows = '\n'.join(f"| {capture['name']} | {row['name']} | {row['wall_seconds']:.6f} | {row['gc_envelope_ms']:.3f} | {row['gc_envelope_to_wall']:.3%} |"
        for capture in analysis['captures'] for row in capture['totals'])
    report.write_text(f'''# GC suspension in the retained complete pyannote traces

GC suspension covers an upper envelope of **1.11% and 1.26%** of the complete
30-second request's captured wall time. This is a small direct suspension
component in these two older Windows captures. It supports keeping matrix
computation ahead of further work aimed solely at GC pauses. It does not bound
background GC CPU usage, explain unprofiled timing variability, or establish
anything about the newer integrated build or the AMD target.

Every row below contains all three measured requests for that fixture. The GC
column is the union of suspension-to-resumption intervals whose recorded reason
is `SuspendForGC` or `SuspendForGCPrep`, clipped to the public request markers.

| Capture | Fixture | Total request seconds | GC suspension envelope ms | Envelope / wall |
|---|---|---:|---:|---:|
{rows}

The [Microsoft GC event explanation](https://devblogs.microsoft.com/dotnet/gc-etw-events-3/)
describes why suspension request through completed resumption is an upper
envelope: some threads can run during the suspension and restart transitions.
The [event reference](https://learn.microsoft.com/en-us/dotnet/framework/performance/garbage-collection-etw-events)
defines their recorded reasons and fields. These semantics do not turn elapsed
event intervals into exact calling-thread stalls or GC CPU measurements.

## Coverage and two preserved analyzer refusals

The original closed captures contain **59 and 60 GC collections**, with zero
reported event loss. All **24 measured requests** have complete begin/end
markers on the original target thread. Marker durations differ from their
Stopwatch intervals by less than the predeclared 2 ms bound. Every saved
generation counter delta matches the number and depth of collection-start
events exactly, counting higher generations toward the lower ones.

The first analyzer incorrectly assumed one serial suspension sequence per
runtime. A resuming thread can still be emitting its final event when another
thread begins suspending the runtime. The corrected parser pairs each sequence
by process, runtime instance and emitting thread, then takes interval unions.
The original runtime-only refusal and the first overlapping event records are
retained in `pairing-refusal.json`.

The second analyzer incorrectly required each GC to finish in the same request
where it started. Several background collections cross request boundaries.
The [background GC documentation](https://learn.microsoft.com/en-us/dotnet/standard/garbage-collection/background-gc)
explains its concurrent operation. The successor keeps every collection's start
and end, reports crossings and completion counts, and reconciles the original
counters at collection start. Collection duration itself is never used as a
pause interval. `counter-refusal.json` preserves the original rejection and its
first crossing collection. All missing-event, pairing, count and resource
checks remain enforced. The final parser passes **17 invariant tests**.

There are also many `SuspendOther` events: 1,448 in capture A, with the exact
count for capture B retained in the observations. They are excluded from GC
totals. Their suspension-to-resumption envelopes cover roughly half of captured
full-request wall time, demonstrating why they must not be relabeled as GC or
as exact stopped-thread time. These event reasons alone do not establish their
cause. Per-reason components and overlaps are retained; overlapping totals
must not be added together.

## Exact scope and reproduction

Inputs are the unchanged `.nettrace` files from the
[sampled-thread diagnostic](../sampled-thread-time/results-20260921.md), captured
on Windows i7-14700KF CPU2, normal .NET 10.0.12, with four warmups and three
measured passes per fixture. Captured Core is
`0d098ba5fd3fd8799bb1dd018148123f296802c769db5f6148a1a5fa9d80118e`;
Data is `1d34666456a5da749dc3b40ee25621af0bed806c9167f3ad12e96d0736dca662`.
The old complete public output/native/ownership proof remains unchanged.
No model executes, no collector attaches, and no product binary changes here.

The exporter uses the exact retained TraceEvent libraries. Session 21438 exited
zero after its restore, build and two offline reads. All **{analysis['resource_samples']} resource
samples** pass; **{len(analysis['identities'])} process identities** are terminal. Peak sampled owned
RSS is **{analysis['peak_rss']:,} bytes**. Bounds are 8 GiB before build, 2 GiB before
each small offline read, 2 GiB owned RSS, 1 GiB available, 20 GiB free disk,
1 GiB output and 300 seconds per child.

Use `C:/Python313/python.exe -X utf8 -B` from repository root. The recorded
sequence is `export.py`, the preserved original tests/refusals, then
`test_analysis_v3.py`, `analyze_v3.py` and `report.py`. Existing output paths
are refused. Original analyzers, captures and failed assumptions are retained.
Full pause and collection records remain in the closed artifact analysis;
[observations](observations-20260922.json) retain every request and summary.

Closure: `{pin(BASE / 'closed.json')['sha256']}` ({pin(BASE / 'closed.json')['bytes']:,} bytes).
Accepted benchmark tables and production defaults remain unchanged. Pyannote
stays first, Parakeet second and Whisper deferred; the e5 and queued AMD owners
are untouched.
''', encoding='utf8')
    print(json.dumps(dict(report=rel(report), closure=pin(BASE / 'closed.json'), resources=analysis['resource_samples'])))


if __name__ == '__main__':
    main()
