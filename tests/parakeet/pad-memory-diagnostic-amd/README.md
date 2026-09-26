# Current-root Pad memory diagnosis

This is one diagnostic on unchanged Core f3992f40 and a74acb17. It cannot admit
a product or replace the failed screen. The exact-model retained ORT profile
shows no per-node live arena allocation increase in all 2,880 measured Pad calls.
Lokad still allocates a fresh owned array. The unresolved question is whether
faults/touching new storage explain the sustained fast/slow candidate blocks.

Reuse the current screen's twelve fixtures, independent oracle, immutable-input
and held-output checks, four fresh processes in selected/candidate/candidate/
selected order, and ten-seconds-after-first-complete-census prefix. Preserve all
600 warmups and 180 subsequent calls per case, plus every prefix call. No
duration search, forced GC, implementation flag, profiler or product rebuild.
Only the common consumer is built, with the existing offline SDK/feed.

Before each public-call timer, take Linux x86-64 getrusage(RUSAGE_THREAD), GC
collection counts and thread allocation counters. Read corresponding counters
after the timer stops. Keep absolute outer/inner bounds. The counter interval
includes its own sampling overhead, while the Pad interval does not. Record
all before/after values, never subtract timing overhead. Two sets of 128 empty
brackets, before and after the complete workload, expose baseline counter cost
and resolution; keep all calibration calls, including startup/JIT effects.
Per-call record allocation follows the snapshots and may affect later GC.

Falsifiable predictions: output-page touching predicts slow calls accompanied
by minor faults and increased thread system CPU, proportional to output pages;
collection pauses predict collection-count changes (counts alone are not a
pause trace); scheduling predicts context switches or wall time without matching
thread CPU. Changed compilation/cache behavior can remain unresolved if these
signals do not account for the regimes. Lack of faults or collection changes
must not be interpreted as proof of a particular compiler or copy bottleneck.
No instruction samples are taken, so this run cannot attribute time within Pad
to allocation, filling or copying. A clear fault association selects an exact
allocation-path/source review before any memory implementation; absent or mixed
signals select diagnosis of the unresolved mechanism, not a kernel sweep.

The source generator verifies exact reversal to the original fixed-prefix
consumer. An additional managed helper defines and validates the 144-byte Linux
x86-64 rusage layout. The original arithmetic, public API and binaries stay
identical. Original score calculations may be retained as descriptive numbers,
but `admitted` is always false. Independently audit all counter monotonicity,
clock brackets, original numerical checks, prefix rules and resource observations.
Publish every fixed block and the fault/GC/scheduling cohorts without replacing
the original screen or trimming clocks. Compare instrumentation-wide times and
empty brackets with the retained uninstrumented distribution; a changed regime
can make the diagnostic inconclusive.

Bounds before preparation: CPU2 runs, CPU0 monitors; 4 GiB available RAM and
1 GiB tmpfs preflight, 2 GiB owned RSS, at least 1 GiB RAM/tmpfs remaining,
256 MiB output/job and 512 MiB total, 900 seconds/job and four hours overall.
No new models, native trace or product build. Freeze tools at preparation.
Namespace artifacts/parakeet-pad-memory-diagnostic-amd-20260926 locally and
/dev/shm/lokad-parakeet-pad-memory-diagnostic-20260926 remotely.

From this directory, use C:/Python313/python.exe -X utf8 -B with test_counters.py
and test_score.py, then run.py prepare, stage, launch, observe, collect separately.
Audit once after every owner is terminal. Never retry a closed or failed worker.
