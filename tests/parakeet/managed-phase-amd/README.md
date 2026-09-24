# Matched Parakeet phase and operator diagnosis

The capture is complete; all 240 requests pass. Read the
[matched results](../managed-phase-results/results-20260924.md).
`compare.py` reconciles complete constant projections, including fused scale;
`compare_slices.py` checks all 24 attention Slice/Reshape pairs, original and
optimized constants, graph edges and every observed native shape. Both analyze
retained records without inference. Existing output is refused. Do not observe
or repeat the closed build/capture namespaces.

Measure the selected Lokad application on the same twenty clips as the completed
ORT diagnosis. Keep Core `672e5f30` unchanged. An isolated Data assembly adds one
`using var observation = ParakeetPhaseProbe.Enter(context);` at the start of the
private Parakeet graph-call helper. The original helper body, public transcription,
decoder, graph outputs, resets and tensor ownership remain intact.

The observer records complete graph-call intervals. In wall mode it enables the
existing Core wall profiler and copies only its scalar node records before the
next reset. It retains no tensors or execution contexts. Node/constant metadata
is captured once during warmup. Actual kernel dispatch is not claimed from this
metadata: this stage first identifies which complete groups explain excess time.

The original sampled application consumer keeps every public/native reference,
input hash, repeat and held-output check. Only Main gains observer initialization,
post-clock record saving and an explicit diagnostic Data hash check. Every other
original compiled consumer method and Data method must match, except the one
instrumented graph-call helper. Public surface and implementation flags remain
unchanged; new observer types are internal. Core is not rebuilt.

Three fresh processes run in fixed order: original Data control, observed Data
phase-only, observed Data wall-profile. Each performs one warmup and three measured
passes: 240 requests total. Save all clocks and quantify both overhead steps.
Diagnostic times cannot replace BENCHMARK.md or admit a candidate.

Use the existing AMD VM, CPU2 for builds/workers and CPU0 for monitoring. Builds
require 2 GiB available/1 GiB tmpfs, with 3 GiB owned RSS, 180 seconds per command
and 512 MiB total output. Inference requires 11 GiB available/2 GiB tmpfs before
each process; cap owned RSS at 12 GiB, each process at 900 seconds and stage output
at 512 MiB. All stages retain at least 1 GiB available memory/tmpfs. Preserve
exact PID/birth ownership, resource samples and terminal status. Reuse canonical
models, clips, runtime and offline feed. No system settings or product defaults change.

Local artifacts: `artifacts/parakeet-managed-phase-amd-20260924`.
VM artifacts: `/dev/shm/lokad-parakeet-managed-phase-20260924`.
Tools refuse existing output and retain failed stages. Never rerun inference to
repair an analyzer. The living plan is `.agent/m65-parakeet-ort-diagnosis-20260924.md`.
