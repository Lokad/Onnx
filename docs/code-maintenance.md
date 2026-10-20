# Code maintenance

The core library is packaged; Data, CLI and development tools are not. Preserve
public APIs, supported numerical behavior and tensor ownership when simplifying
code. A source search cannot establish whether an external application uses a
public member or whether a diagnostic invokes an internal method by reflection.

## Maintained entrypoints

Use `eng/test.ps1` for normal tests (`-Coverage` opts into local coverage),
`eng/test-opfuzz.ps1` for native differential contracts, and `eng/smoke-pack.ps1`
for a real package consumer. The e5 and Gemma
lanes are `eng/test-e5.ps1` and `eng/test-embeddinggemma2.ps1`. Audio replay lanes
remain `eng/test-whisper.ps1`, `eng/test-parakeet.ps1`, and the frontend,
transcription, embedding, diarization and accuracy examples under `tests`.

For CPU optimization, use the [routine comparison command](../eng/bench.md).
It reuses the current Campaign/Gemma workers and their supervision/correctness
checks. Strict scoring protocols remain available when a release claim needs
stronger qualification. Run one measurement lane at a time and clean up verified
owned leftovers before timing.

## Review and numerical contracts

Prefer explicit arguments and small changes. TODO comments should explain the
missing work. Review those conventions with the code rather than maintaining a
custom C# parser or a CI allowlist of comments. The previous Roslyn source audit
and optional-parameter/TODO scanners are [archived](../archives/README.md).
The null-suppression and documentation syntax scanners have also been retired;
compiled assembly metadata still checks the approved friend boundary. Normal
maintenance does not depend on SDK-internal analysis assemblies or a member-retention registry.

Keep independent numerical expectations in `tests/Shared/ReferenceKernels`.
Calling the optimized implementation from its reference would invalidate the
comparison. The prepared LSTM projection contract compares every finite value,
signed zero and infinity bit-for-bit; exceptional arithmetic must preserve NaN
classification. NaN payload/sign selection across independently compiled
arithmetic is unspecified and is not part of that contract. Input immutability,
output guards and zero-allocation assertions remain enforced.

Some production methods intentionally have equal source arithmetic but different
JIT compilation attributes or instruction widths. Examples are `MathOps.ErfVector`
and `ErfVectorInline`, and `ExpVectorEstrinPruned` and its inline twin. Packed
matrix and convolution kernels likewise require numerical and affected-model
checks before consolidation. Keep their separate compiled entrypoints when
qualification shows a benefit.

## Preparation ownership and model policy

Graphs and their execution contexts share one internal `GraphPreparationCache`
for prepared weights and synchronization. Each context retains its own bindings,
preparation fingerprint, diagnostics and released-buffer cache. The public graph
remains mutable: keep structural-change detection and invalidation semantics.

`PrepareOwnedMatMulWeights()` is the existing opt-in for the qualified Parakeet
export. `ParakeetOwnedWeightPolicy` holds its exact name/shape selection;
`GraphOwnedPacking` owns the generic consumer, alias and storage checks. A node
rename can still disable eligibility. Additional exports/shapes need explicit
qualification; do not broaden selection solely because dimensions match.

## Runtime experiments and archives

The nine qualified comparison controls are listed in
[runtime options](runtime-options.md). Default-off internal experiments and
mutable legacy-routing hooks have been retired. Public options and entrypoints
remain compatible. New experiments should have a concrete workload, acceptance
criterion and promotion/retirement decision before becoming permanent runtime
branches.

Closed producers and obsolete recovery/retention tools are indexed against their
original Git commit in [archives](../archives/README.md). Reports and result data
stay browsable at their historical paths. Original source paths/bytes, including
the forty previously hash-locked C# inputs, can be restored into an isolated
directory. Active examples, replay tools, reference generators and their shared
dependencies remain maintained. Keep future experiments on the shared runner
instead of cloning launch/observe scripts.

## CI scope

Every push/PR gets a lightweight change-classification job. Markdown-only edits
skip the numerical matrix; README and CHANGELOG also run package smoke because
they ship in the package. Code, workflow, fixtures, build files and unknown paths
conservatively run the full Windows/Linux normal/scalar, differential and package
jobs. Deletions and renames count. Check a published commit's CI status minutes
apart, and investigate failures rather than adding automatic retries.
