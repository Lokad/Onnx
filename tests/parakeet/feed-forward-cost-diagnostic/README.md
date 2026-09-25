# Separate Parakeet weight preparation from multiplication

This diagnostic addresses one unresolved question from
[the retained feed-forward review](../slice-dense-conversion-results/feed-forward-review-20260925.md).
It selects no optimization and changes no product source. The slice-conversion
graph campaign is terminal: numerical and regression checks pass, but two e5
repeatability controls fail. That release remains unadmitted. This isolated
diagnostic uses its exact compiled and Parakeet-qualified candidate; it neither
waives those controls nor requires promoting the candidate first.

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

Before capture, compile on the now-idle AMD VM, verify the
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
of this diagnostic.

The bounded transport (`run.py`), VM worker (`vm.py`), compiled build review
(`review_build.py`, `build_checks.py`) and capture audit (`audit.py`, `analyze.py`)
are drafted. `isolated_baseline.py` requires the exact candidate source, compiled
review, both tensor suites, full Parakeet/native/public checks, Parakeet
application admission, shared/Pyannote correctness and the retained failed graph
verdict. It verifies all 428 source inputs and reconstructs the inspected
candidate method inventory from existing bodies plus the sole reviewed override.
It does not rebuild or invent a baseline body. The selected release's unchanged
697 Data methods remain the Data baseline. No diagnostic campaign is prepared yet.

The build review requires all 3,254 original Core methods: 3,245 unchanged, seven
wrappers equivalent after removing only profiler markers, and two explicitly
reviewed enclosing-clock methods. Branches, cleanup regions, locals, stack and
implementation flags remain checked. All 697 original Data methods remain exact
except Execute, whose observation scope must match the previously certified
observer. Reuse both the existing assembly inspector and application consumer.
The warning check preserves the two existing CS8604 warnings and rejects added
warnings; this is not a warning-free baseline.

Eleven checker tests pass, including deliberately changed arithmetic, branch
targets, exception regions, locals and stage labels:

    C:/Python313/python.exe -X utf8 -B -m unittest discover -s tests/parakeet/feed-forward-cost-diagnostic -p test_il_check.py -v

The normalizer was also exercised on all seven actual retained wrapper bodies.
These validate the checker, not a diagnostic binary.

`reference.py` checks seven retained evidence files, all 2,856 encoder nodes,
96 complete projection groups and 7,680 matched calls. Their digests and the
compact reference are bound into the prepared workload. `analyze.py` joins
each actual shape and route, keeps both output-preparation intervals and every
separate scale operation, and reconciles stage, node, graph and request clocks.
Graph counters are fresh; per-node scratch/copy counts and ORT attribution are
explicitly identified as earlier observations.

Fourteen accounting tests pass, including wrong clock units, omitted scale
work, wrong shapes, inconsistent packing routes, missing passes and excessive
observer effects:

    C:/Python313/python.exe -X utf8 -B -m unittest discover -s tests/parakeet/feed-forward-cost-diagnostic -p test_analyze.py -v

The prospective controls require each role's corpus max/min <= 1.10 and each
clip's max/min <= 1.20 across its three measured passes. Both stages/clock and
markers/stages corpus ratios must fall in [0.95, 1.05]. The auditor preserves
all 240 requests, all failed controls and the uncorrected timings. A failed
control sets `usable_for_candidate_selection` to false; successful collection
and complete accounting alone do not make the split suitable for a decision.

Run `isolated_baseline.py` to verify the prerequisites, then
`run.py prepare`, `stage`, `launch build`, `observe build`, `collect build`,
then `review_build.py`. Only a passing compiled review permits `launch capture`.
Observe until terminal, collect once and run `audit.py` once, with stdout outside
the artifact directory. Keep the first verdict and all failed controls.
Release qualification, root integration and BENCHMARK promotion remain separate
and require resolving the e5 controls, then the remaining original release gates.
