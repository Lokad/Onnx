# Wide-projection runtime diagnostic

Trace the rejected M52 candidate and unchanged selected product using existing
compiled instruments. Six serial jobs: SDK/tracer identity, two captures and two
complete exports. No build, disassembly, score or application comparison.

The unchanged consumer executes 21 fixtures with 120 calls each, paired events,
whole-process allocation/GC counters and full output/input/guard/destination
checks. Reuse M51's complete reconciliation and rejection tests byte-for-byte.
Require 5,040 calls, 10,080 markers, every raw/decoded event and zero loss.
Inspect fixture 5's late interval while retaining all cases and fixed blocks.

Reuse hash-verified hardlinks for fixtures, products, consumer and complete
exporter/tracer. Workers run CPU2; collector/exporter CPU0. Ordinary runtime flags
and identical provider string. No sampling profiler. Method loads show code
availability, not executed versions. Instrumentation changes timing/history.

Use C:/Python313/python.exe -X utf8 -B with the reused tests, then run.py prepare,
stage, launch, observe, collect separately and audit.py after termination.
Commit the frozen tools before launch. Refuse existing namespaces and preserve
failures. Original component verdicts and BENCHMARK.md remain unchanged.

Bounds: 12GiB available/3GiB tmpfs preflight, 8GiB RSS, 1GiB minimum available/tmpfs,
256MiB output/job, 512MiB campaign, 900seconds/job, 4hours overall; every OS thread
affinity and monitoring gap under10seconds checked. No model copies.
