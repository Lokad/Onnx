# ORT-derived rational sigmoid: complete Parakeet comparison

**The candidate passes the unchanged application admission.**
Complete transcription latency is 3.249% lower than current root.
Repeatability: 63/63 checks. Performance: 21/21 gates.
Candidate/ORT is **1.244628**; the independent
1.05 parity target is not met.

| Twenty-clip corpus (213.265 s audio) | Seconds | Relative to ORT |
| --- | ---: | ---: |
| current root baseline | 50.448331 | 1.286423 |
| Rational-sigmoid candidate | 48.809282 | 1.244628 |
| Microsoft ORT 1.29.0 | 39.215963 | 1.000000 |

Six fresh processes run current root, candidate, ORT, ORT, candidate, current root on AMD EPYC
9V74 CPU 2. Each uses one warmup and three measured passes over every clip:
480 requests, 120 warmups and 360 measurements. The common consumers,
per-request validation and exact-clock scorer are unchanged. There is no
profiling, compilation or runtime override in this comparison. No clock is trimmed.

The prospective limits require at least 3% corpus gain, no clip more than 5%
slower, corpus process max/min <= 1.10 and per-clip max/min <= 1.20 for all engines.
All workers are terminal/code0; all 2,252
resource samples pass. Peak owned RSS is 9,002,823,680 bytes.

The single product change uses ORT-derived rational sigmoid arithmetic for eligible
contiguous float tensors, with the existing vector width and independently owned
outputs. [Complete model correctness](models-20260927.md) passes the original ORT
bound, exact decisions/transcripts and ownership checks in normal and AVX512-disabled
execution. Its floating-point differences are recorded; no bit-identity claim is made.
The [executed ORT SiLU review](../ort-activation-review/diagnosis-20260926.md) selected
this arithmetic, and the [fallback diagnosis](fallback-diagnosis-20260927.md) records
its remaining limitations.

The [operator screen](screen-20260927.md) remains rejected: 13 repeatability checks,
four fallback regressions and its 75% weighted-gain threshold failed. Its separate
diagnostic leaves double latency unresolved. This application result does not
change that verdict or establish universal fallback performance equivalence.

Next: resolve or explicitly justify the retained fallback limitation, then require shared/e5, Pyannote, graph and actual-root/package qualification before any source or BENCHMARK.md promotion.
The current release and BENCHMARK.md stay unchanged. Parity remains a separate goal.

[Every case](application-20260927.csv), [all 480 clocks](application-20260927-clocks.csv),
and [all controls, identities and resources](application-20260927.json).
Closure: `629183e44963638c0285ea4daf70f8f9aab3b94d5e4c6164225d205aaaa443b4`.
Raw evidence: `artifacts/parakeet-rational-sigmoid-app-amd-20260927`.
