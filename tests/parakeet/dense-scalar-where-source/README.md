# Dense scalar Where source proposal

This isolated M59candidate replaces the uniform-only helper with a selector
that also handles mixed dense masks in contiguous broadcast runs. It preserves
the4096element provider guard, all existing null/option/dtype validation, and
all generic Tensor.Where code. Unsupported classes, reversed layouts, ranks,
non-scalar true inputs and incompatible shapes retain the selected fallback.

Try keeps exact prior layout/shape guards and zero/nonzero Boolean truth.
Uniform masks copy or fill fresh output. Mixed masks build at most8effective
condition strides, then process trailing contiguous or scalar condition runs.
An explicitly cleared stack position counter advances only at run boundaries.
The inner selection performs no float arithmetic, indexing division, per-element
coordinate allocation or input mutation. Outputs always own their storage.

The proposal follows rejected M57screen a6fed3ec and completed M58diagnostic
 ae8b002e/review7382ced4. Both providers emit full Tier1 there, but mixed masks
 still scan then execute generic indexed broadcasting. Neither past rejection
 nor historical timed-tier uncertainty is changed by this proposal.

Freeze tools then use C:/Python313/python.exe -X utf8 -B prepare.py. The generator
refuses an existing artifacts/parakeet-dense-scalar-where-source-20260924,
verifies every selected root input and prerequisite, and records423inputs with
only the provider file modified and Zzz.DenseScalarWhere.cs added. It verifies
the prior helper's exact layout/shape-validation prefix and the retained ORT
source hash. Root product files stay unchanged.

Source preparation is not compilation, correctness or performance admission.
Follow .agent/m59-parakeet-dense-scalar-where-20260924.md for exact compiled-scope
checks, both-boundary/both-mode numerical/ownership/contracts,32additional run
layouts, native code review and the full performance/release gates. No Windows
build or inference. Any redesigned performance protocol must first demonstrate
selected/selected stability. BENCHMARK.md remains the qualified release.
