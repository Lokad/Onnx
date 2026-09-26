# Parakeet sigmoid: arithmetic is the next bounded intervention

The completed diagnostic supports replacing general exponential evaluation with
the rational logistic arithmetic observed in ORT. It does **not** establish a
cycle-by-cycle explanation of the remaining difference. No product changed and
no benchmark score was admitted. Parakeet's release comparison remains
50.160754 seconds versus ORT's 39.242580 seconds, ratio 1.278223.

The common consumer ran the qualified scalar Core `8bb22038` and retained,
rejected vector-exp Core `9bae182b` in four fresh processes. It preserved all
46 fixtures, 600 warmup rounds, 180 subsequent rounds, 143,520 clocks, numerical
checks and ownership checks. Allocation and GC counters were outside the timer.
All four processes and the supervisor are terminal; capture, resource and
correctness checks pass. The original vector screen remains rejected.

| Evidence | Conclusion and limit |
|---|---|
| Actual ORT samples and binary | The optimized graph contains 72 fused SiLU nodes; samples identify `MlasSiluKernelAvx512F`: rational logistic arithmetic, 512-bit vectors, then input multiplication. Samples cover request intervals rather than identifying each individual node. |
| Managed emitted code | `Vector<float>.Count` is 8. The rejected candidate's optimized vector loop uses 256-bit operands, including on this AVX-512 machine. |
| General exponential loop | 70 emitted instructions per eight-element loop, including range reduction, float/integer conversions, exponent reconstruction and special-value handling. `ExpVector` is inlined: no helper call, no vector spill, and one output-pointer reload from the stack. Instruction count is not a cycle estimate. |
| Allocation census | All 46 cases have identical minimum bytes per call across all four processes. The 106-frame dense fixtures have exactly equal bytes in every measured batch. Sparse extra allocations in other fixtures are retained and unexplained. This excludes systematic additional payload in these fixtures, not differences in allocation latency. |
| Scalar-option path | Diagnostic means are 344/347 microseconds for current, 457/428 for candidate. Current-0 and candidate-1 have no observed GC collection in these measured batches. Collection events cannot account for that pair's difference. |
| Scalar generated loops | Final Tier1 current uses a pointer-increment loop; candidate uses indexed iteration, retaining length and output-pointer reloads. The first scalar Tier1-OSR loops have identical instructions apart from relative call bytes. There is no tier-to-clock join, so the observed Tier1 difference is a plausible mechanism, not a proved cause of the slowdown. |
| Empty path | All four allocate exactly 280 bytes per call. Means are 91/79 ns for current, 95/79 ns for candidate. These observations do not isolate the cause of the original empty regression or overturn it. |

Tier1 is the runtime's optimized compiled version. Tier1-OSR is optimized code
entered from an already running loop. Every emitted version is retained; a
listing alone does not say which version ran during a particular clock.

For the feed-forward fixture `[1,106,4096]`, every measured call allocates
1,737,008 bytes: the 1,736,704-byte output payload plus 304 bytes. The convolution
fixture `[1,1024,106]` allocates 434,480 bytes: payload plus the same 304 bytes.
The public implementation creates an owned output. `DenseTensor.ToDenseTensor`
returns the existing tensor for ordinary dense storage. These facts concern the
observed fixtures and reviewed source; runtime tensor identities were not captured
inside all model nodes.

The current complete profile assigns 2.458878 seconds to Sigmoid and just
0.115897 seconds to separate activation multiplies. Total activation excess
against the dated ORT profile is 2.375274 seconds, about 4.7% of the managed
profile. Fusion alone has a small measured opportunity. Removing the activation
gap entirely would still leave substantial decoder, convolution and projection
work. These diagnostic clocks are not interchangeable with the release score.

The next experiment is one fixed rational-sigmoid implementation using ORT's
observed coefficients and operation order. Keep the observed 256-bit width,
public output allocation, graph structure and default compiler settings fixed.
Use a separate vector helper that also handles its remainder, so the original
scalar fallback can retain its zero-based loop rather than sharing a mutable
index with vector processing. Keep double, disabled-SIMD and empty behavior.
Do not introduce graph fusion, a 512-bit variant, polynomial tuning or a sweep.

The working prediction is at least 75% lower corpus-weighted public Sigmoid cost
under the existing screen, which projects to about 1.84 seconds, or 3.6% of the
current profiled application if the operator saving transfers. That projection
is a hypothesis. Preserve the original numerical, ownership, repeatability and
per-case regression gates; test the full model and application before promotion.
Inspect generated code to confirm that exponential calls/conversions disappear
from the vector body and the scalar loop remains independent. A numerical failure,
fallback regression, missed timing prediction or absent application gain rejects
the corresponding claim. Preserve a failure and diagnose it before another variant.

Remaining uncertainty is explicit: no instruction sampling or tier-to-clock
association; no allocation-latency breakdown; sparse extra allocations lack a
source; synthetic values are not captured model activations; ORT's standalone
gate-Sigmoid leaf was not identified by the SiLU samples. The exact ORT SiLU
algorithm still supplies a concrete arithmetic hypothesis for sigmoid.

[All 46 cases and selected generated loops](observations-20260927.json),
[read-only summary](summarize.py),
[actual ORT instructions and samples](../ort-activation-review/diagnosis-20260926.md),
[complete current attribution](../pad-current-profile-results/diagnosis-20260927.md),
[original rejected screen](../vector-sigmoid-results/screen-20260925.md).

Reproduce the summary locally, without contacting the VM or changing evidence:

    C:/Python313/python.exe -X utf8 -B tests/parakeet/sigmoid-execution-results/summarize.py

The one-time `--publish` writes the observations file only if absent. Subsequent
default runs check that the derived observations still match the published file.
Do not rerun the completed diagnostic campaign or its publication.

Build review: `a98b3ba289daea3e91e8d96c70e7d9c890b658597888b24d52d0c06b3dcd65cc`.
Capture closure: `e2ec74acf9b8fe9c36a48226aa478b5efded38071f6b699a04f64410265dd868`.
Artifact: `artifacts/parakeet-sigmoid-execution-diagnostic-amd-20260927`.
