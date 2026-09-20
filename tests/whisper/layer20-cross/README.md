# Controlled natural-input Whisper layer-20 diagnostic

The [selected full encoder trace](../trace-selected/results-20260920.md) first
crossed the fixed1e-4 limit at layer20 for all six unique case/feature pairs.
This diagnostic reuses the exact48-node/27-initializer cut, verified against the
original graph and all external-weight bytes. No encoder inference is repeated.

For each of three selected cases plus a first-case repeat, and both feature
sources, run both saved layer-19 arrays through each engine. Sixteen fresh
sequential workers each make two calls, holding the first outputs. Each call
retains all twelve cut outputs:384 arrays and4,423,680,000 bytes overall.

Cell letters now name cut engine and incoming-array producer:MM/MN/NM/NN.
Feature source is separate. Diagonal cells bridge to their same-engine traced
layer20 outputs; off-diagonal cells distinguish incoming effects from same-input
engine differences. Failed extraction bridges stay explicit. All pairwise
comparisons and the common-NN-denominator decomposition retain every value.
This does not replace full-corpus numerical acceptance or identify either FP32
engine as mathematical truth.

Run `prepare.py generate --artifact artifacts/whisper-layer20-cross-20260920`
to create host/tools from receipt-bound trace sources and save the full diff.
Build generated `host/Probe.csproj` in Release with `--tl:off --nologo -v minimal`,
`FrozenProductDirectory=<absolute selected-trace artifact>/bin`, output`bin`;
retain `build.log`. Run generated `tools/test_audit.py`, retaining `unit-tests.log`.
Commit, then `prepare.py freeze --artifact <artifact>` binds every source,
model, incoming array, traced baseline, build and native module identity.

Launch generated `tools/run.py --artifact <artifact>` in a hidden background
process with retained stdout/stderr and PID/creation time. Workers inherit CPU2;
supervisorCPU0. Managed uses original core c6bf781/.NET10.0.12, fresh Memory
contexts and256-MiB packing. NativeORT1.29.0 uses the original pinned environment,
one thread, sequential execution/all optimizations/spinning off. Preserve global
environment and unrelated work; scrub settings only in inference children.

Each worker has fixed180-second/8-GiB sampled group-RSS/1-GiB available-memory
limits, with10GiB before launch,25GiB disk and at most900seconds of memory waiting.
Preserve all failure prefixes; never replay a successful writer or closed run.

After all original identities terminate, run generated `tools/audit.py --artifact
<artifact> --output <artifact>/audit.json`, then `tools/close.py --artifact <artifact>`.
The closer separately recomputes every same-input report comparison and all
sixteen extraction bridges from raw arrays. Independently verify the final
receipt, report pins and terminal identities. No product or performance claim.
