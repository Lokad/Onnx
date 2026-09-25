# Separate Parakeet weight preparation from multiplication

This diagnostic addresses one unresolved question from
[the retained feed-forward review](../slice-dense-conversion-results/feed-forward-review-20260925.md).
It selects no optimization, changes no product source and starts no VM workload.
The current slice-conversion release qualification keeps the VM until completion.

`source.py` generates isolated copies of four Core files, a diagnostic-only stage
vocabulary, a complete patch and a reversible source review. Run from the root:

    C:/Python313/python.exe -X utf8 -B tests/parakeet/feed-forward-cost-diagnostic/source.py

The source generation has completed at review `be4e3a92`; do not rerun it.
The output directory is `artifacts/parakeet-feed-forward-cost-source-20260925`.
Existing output is refused. Read `review.json` and `diagnostic.patch`; do not
interpret source review as compiled equivalence or a successful measurement.

The completed `verify_source.py` review `61b75a57`, under
`artifacts/parakeet-feed-forward-cost-observer-source-20260925`, restores all
original bytes including line endings. It also prepares the one-line Data hook
and the scalar-only observer, verifies all 2,856 encoder descriptors and 96
feed-forward targets, and confirms reuse of qualified consumer `38ab5c7e`.
Both reviews and all generated sources are retained. No compiler or inference
has run for this diagnostic.

The added boundaries separate scratch rental, packing, packed multiplication,
scratch return, an odd final row, output allocation/clearing and prepared-weight
multiplication. Existing arithmetic leaves, flags, weight budgets, ownership,
validation and fallback decisions are retained. Integer stage IDs 1000–1006 are
private diagnostic vocabulary. The original enum and public API are unchanged;
these sources are for the diagnostic consumer, not the CLI or NuGet package.

The existing stage profiler excludes its transition overhead. The diagnostic
also records enclosing node intervals. The analyzer must retain and report the
difference between stage sums and enclosing clocks, and the difference between
node sums and complete graph calls. It must not subtract estimated overhead or
discard unclassified time. All 48 scalar multiplications corresponding to ORT's
fused scale remain part of their complete projection groups.

Before capture, compile on the AMD VM after release qualification, verify the
exact allowed method changes and unchanged arithmetic leaves, and reuse the
original complete twenty-clip public/native checks. Compare three fresh
processes: original Core with graph clocks, original Core with existing stage
profiling, and diagnostic Core with stage profiling plus node intervals. This
measures profiler and marker effects separately; it does not admit a speedup.
`PhaseProbe.cs.txt` preserves the consumer's reflection entry point. Keep
`PARAKEET_PHASE_MODE=phase`; `PARAKEET_COST_MODE=clock|stages|markers` selects the
three observer roles. Assigned Core and Data hashes remain mandatory. The
consumer itself needs no modification or rebuild.

Each process retains one warmup and three measured passes, with all clocks and
results. The single-thread setting and CPU2 affinity are required.

Join every feed-forward call by graph node, clip, pass and actual length to the
retained exact ORT groups. Require 96 weights, 80 calls per weight, the unchanged
prepared-weight census, no operand copies and the original numerical/public
contracts. Do not infer per-node native buffers from a source path alone. The
current native proof identifies the actual installed AVX-512 routine; its global
sample share includes convolution and decoder work as well as feed-forward work.

Only a trustworthy cost split can select one implementation. If preparation
cannot explain useful whole-application savings, stop pursuing packing changes
and inspect the actual arithmetic route. No new budget or kernel sweep is part
of this diagnostic. The bounded Core/Data build, capture and analysis adapters
remain to be implemented and qualified; this directory presently reviews source
and consumer reuse only.
