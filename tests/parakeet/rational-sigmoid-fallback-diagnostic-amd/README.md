# Explain the rational candidate's fallback failures

The closed ordinary-runtime screen fc8a6d97 rejects rational Core 946ddfb6:
74.992572% weighted saving misses 75%; 13 repeatability controls and four fallback
gates fail. The root reference is Core 8bb22038. The rational arithmetic is already
numerically qualified and emitted as predicted. Do not tune it or rerun admission.

The contract process emitted only Tier0 for the public wrapper, so it did not
establish the optimized fallback used after the performance screen's warmup.
The retained ordinary clocks also show varying empty/scalar costs in both products.
Two explanations remain: changed wrapper/loop code and runtime allocation/GC
effects. This observation determines which needs work before any further change.

Use the exact two retained products and original complete-call consumer with the
same six reversible diagnostic edits used in sigmoid-execution-diagnostic-amd.
Keep all 46 fixtures, values, calls, public-result checks, 600 warmup rounds,
180 subsequent rounds and four-process order. Collect thread allocation bytes and
GC collection counts outside the timer. Preserve all 143,520 clocks and counters.
No product source or binary changes, model inference or performance admission.

Allow only DOTNET_JitDisasm for CPUExecutionProvider.Sigmoid and its private
SigmoidRationalVector helper, plus DOTNET_JitDisasmWithCodeBytes=1. Record every
emitted version and Vector<float>.Count. Compare optimized scalar/double/empty
paths with the current product, including branches, calls, pointer/index handling
and stack operands. Listings do not associate a tier with a specific clock.

If extra output/materialization bytes appear, investigate that allocation path.
If allocation bytes agree but candidate fallback loops retain more bounds, reloads
or calls, that supplies a specific code-generation hypothesis. If loops agree and
large batch costs coincide with recorded collections, investigate the screen's
runtime behavior before attributing those costs to arithmetic. Keep all samples;
do not subtract GC, trim clocks or reclassify old failures. Similar clocks without
collections would leave allocator latency/tiering or other runtime effects open.
New diagnostic observations cannot retrospectively identify old GC/tier events.

Reuse the original worker, monitor and whole-record validator. Its final evaluator
reports observations without timing admission. The original rational screen stays
failed regardless of these diagnostic clocks. Use code and counters to select one
next decision; no ISA, polynomial, unrolling or warmup sweep is authorized.

Bounds are unchanged from the earlier small diagnostic: 2 GiB available and 1 GiB
tmpfs before jobs; RSS <3 GiB; >=1 GiB free; output <512 MiB; 180 seconds/build job,
900 seconds/capture process. CPU2 computes, CPU0 monitors. Check idle owners and
immutable products before launch; use the offline feed and common consumer.

Prefix commands with C:/Python313/python.exe -X utf8 -B from repository root:

    -m unittest discover -s tests/parakeet/rational-sigmoid-fallback-diagnostic-amd -v
    tests/parakeet/rational-sigmoid-fallback-diagnostic-amd/run.py prepare
    tests/parakeet/rational-sigmoid-fallback-diagnostic-amd/run.py stage
    tests/parakeet/rational-sigmoid-fallback-diagnostic-amd/run.py launch build
    tests/parakeet/rational-sigmoid-fallback-diagnostic-amd/run.py observe build
    tests/parakeet/rational-sigmoid-fallback-diagnostic-amd/run.py collect build
    tests/parakeet/rational-sigmoid-fallback-diagnostic-amd/audit.py build
    tests/parakeet/rational-sigmoid-fallback-diagnostic-amd/run.py launch capture
    tests/parakeet/rational-sigmoid-fallback-diagnostic-amd/run.py observe capture
    tests/parakeet/rational-sigmoid-fallback-diagnostic-amd/run.py collect capture
    tests/parakeet/rational-sigmoid-fallback-diagnostic-amd/audit.py capture

Freeze at preparation. Only observe repeats while original owners are live.
Collect/audit once per terminal phase; audit stdout belongs outside the campaign.
No completed experiment or failed screen is repeated.

Local: artifacts/parakeet-rational-sigmoid-fallback-diagnostic-amd-20260927.
VM: /dev/shm/lokad-parakeet-rational-sigmoid-fallback-diagnostic-20260927.
