# Evaluate the observed dense-mask route on the qualified release

The [complete-corpus observation](../managed-phase-results/masking-layouts-20260924.md)
finds all 5,760 Where masks are all-false and their operands are exact dense
row-major tensors. Every case meets the retained selector's existing guard.
The [matched ORT work](../managed-phase-results/masking-padding-20260924.md)
identifies repeated broadcast index calculations as a specific source of excess
work. This preparation reuses the existing implementation; it designs no variant.

Compose the exact [numerically qualified helper](../dense-scalar-where-results/numerics-20260924.md)
with source `a2ab32e7`. The only changes are the retained provider float arm and
`Zzz.DenseScalarWhere.cs`. Keep the 4,096-element guard, mixed-mask broadcast
runs, zero/nonzero Boolean truth, owned outputs and every fallback unchanged.
Generic `Tensor<T>.Where` remains exact. Root source and BENCHMARK stay qualified.

The failed isolated timing controls remain rejected. The new work uses the
existing complete-application protocol, whose repeats already pass for this
exact starting product, and still requires every fresh control and admission
gate. Retain the full 235-case correctness census, matched attribution and
complete model/public checks before any application comparison. Source
preparation alone admits neither performance nor integration.

From the repository root:

    C:/Python313/python.exe -X utf8 -B tests/parakeet/observed-dense-where-source/prepare.py

The exclusive output is `artifacts/parakeet-observed-dense-where-source-20260924`.
Preparation verifies all 425 current inputs and produces exactly 426 isolated
files, with the two named changes only. It binds the current release, previous
numerical qualification, actual layout capture and existing application controls.
No build or model inference occurs. The living plan is
`.agent/m70-parakeet-observed-dense-where-20260924.md`.
