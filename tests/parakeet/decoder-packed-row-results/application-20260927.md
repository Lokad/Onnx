# Prepared single-row weights: complete Parakeet comparison

**The candidate passes the prospective application admission.**
Complete transcription latency is 3.123% lower than current root.
Repeatability: 63/63 checks. Performance: 21/21 gates.
Candidate/ORT is **1.207329**; the independent
1.05 parity target is not met.

| Twenty-clip corpus (213.265 s audio) | Seconds | Relative to ORT |
| --- | ---: | ---: |
| current root baseline | 48.944688 | 1.246248 |
| Prepared-row candidate | 47.416186 | 1.207329 |
| Microsoft ORT 1.29.0 | 39.273619 | 1.000000 |

Six fresh processes run current root, candidate, ORT, ORT, candidate, current root on AMD EPYC
9V74 CPU 2. Each uses one warmup and three measured passes over every clip:
480 requests, 120 warmups and 360 measurements. The common consumers,
per-request validation and exact-clock aggregation are unchanged. There is no
profiling, compilation or runtime override in this comparison. No clock is trimmed.

The prospective limits require at least 1% corpus gain, no clip more than 5%
slower, corpus process max/min <= 1.10 and per-clip max/min <= 1.20 for all engines.
All workers are terminal/code0; all 2,205
resource samples pass. Peak owned RSS is 9,943,367,680 bytes.

The single product change reads the existing packed final-projection weights while
preserving arithmetic, reduction order, vector width and independently owned
outputs. [Complete model correctness](models-20260927.md) passes the original ORT
bound, byte-exact current/candidate tensors, decoder decisions/transcripts and
ownership checks in normal and AVX512-disabled execution. The
[original decoder observation](../decoder-projection-observation-results/README.md)
identified an available packed constant with an executed row-major kernel.

The [operator screen](screen-20260927.md) remains rejected: two repeatability
controls and two fallback regressions failed. The [per-call diagnosis](unmapped-calls-20260927.md)
rejects the first-call-only explanation and shows recovery over several calls.
Cache competition is an inference, not a measured hardware cause. This independent
application result does not change either verdict or establish universal fallback
performance equivalence.

Next: resolve or explicitly justify the retained fallback limitation, then require shared/e5, Pyannote, graph and actual-root/package qualification before any source or BENCHMARK.md promotion.
The current release and BENCHMARK.md stay unchanged. Parity remains a separate goal.

[Every case](application-20260927.csv), [all 480 clocks](application-20260927-clocks.csv),
and [all controls, identities and resources](application-20260927.json).
Closure: `26ac173f867a964b73761d3c716aa5db0c384909bb1849e2cc810ed90e0a1781`.
Raw evidence: `artifacts/parakeet-decoder-packed-row-app-amd-20260927`.
