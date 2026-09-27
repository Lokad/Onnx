# The pointwise candidate passes complete Parakeet correctness

All eight jobs pass: **3,136 arrays / 12,361,976 values and eighty public
transcriptions**, covering current and candidate in normal and AVX512-disabled
modes. Maximum scaled error against the retained Microsoft ORT reference is
0.00005155801773 for every product/mode, below the unchanged 0.0001 bound.

All 784 candidate tensors per mode are byte-identical to current. Integer
outputs, decoder decisions, tokens and complete public results match exactly.
Input immutability and independently held outputs pass. No model gate changed.

Current Core `47984318` and candidate `7cac6788` share Data `dd56902f`. The
compiled-scope check reconciles all 3,983 original Core/Data methods through
the qualified root: one existing Core body changes and two private helpers
are added; the other 3,285 Core methods, all 697 Data methods and all original
flags/public bindings remain intact. Both compatibility tests reject unrelated
changes, altered flags/bindings and stale product identities.

The existing TranscribeReplay and AudioBenchmark binaries, original native
reference and twenty-clip fixtures are reused. No product, consumer or model
was built or downloaded. All eight jobs exit zero and all 1,141 resource samples
pass, with peak owned RSS 8,068,247,552 bytes. Job durations are correctness
execution costs, not benchmark scores.

The [component screen](pointwise-tail-timing-20260927.md) remains not admitted,
including its failed prediction. The [runtime diagnosis](pointwise-tail-runtime-20260927.md)
finds kernel recompilation inside measurement while retaining unexplained
variation. This correctness result revises neither verdict and establishes no
isolated speedup.

Next is the original independent complete-transcription comparison for this
unchanged candidate: current/candidate/ORT/ORT/candidate/current, every clock
retained, >=1% corpus gain, no clip >5% slower and all original repeatability
and public-result gates. Broader release checks remain required before promotion.
Root source and BENCHMARK.md retain the qualified **1.188× ORT** result.

[Checks, product identities and resources](pointwise-tail-models-observations-20260927.json),
[correctness protocol](../pointwise-tail-models-amd/README.md),
[application protocol](../pointwise-tail-app-amd/README.md).

Closure: `f773277daa848121c199c58e67bce3777677c2ee2654fbe26250d02719155960`.
Raw evidence: `artifacts/parakeet-pointwise-tail-models-amd-20260927`.
Supervisor 1225489 / birth1790533437.08 and all children are terminal;
4,188 files were collected once and independently audited.
