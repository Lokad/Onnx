# Contiguous padding: complete Parakeet comparison

**The candidate passes the unchanged application admission.**
Complete transcription latency is 6.023% lower than current root.
Repeatability: 63/63 checks. Performance: 21/21 gates.
Candidate/ORT is **1.278223**; the independent
1.05 parity target is not met.

| Twenty-clip corpus (213.265 s audio) | Seconds | Relative to ORT |
| --- | ---: | ---: |
| current root baseline | 53.375791 | 1.360150 |
| Copy-padding candidate | 50.160754 | 1.278223 |
| Microsoft ORT 1.29.0 | 39.242580 | 1.000000 |

Six fresh processes run current root, candidate, ORT, ORT, candidate, current root on AMD EPYC
9V74 CPU 2. Each uses one warmup and three measured passes over every clip:
480 requests, 120 warmups and 360 measurements. The common consumers,
per-request validation and exact-clock scorer are unchanged. There is no
profiling, compilation or runtime override in this comparison. No clock is trimmed.

The prospective limits require at least 3% corpus gain, no clip more than 5%
slower, corpus process max/min <= 1.10 and per-clip max/min <= 1.20 for all engines.
All workers are terminal/code0; all 2,321
resource samples pass. Peak owned RSS is 9,919,242,240 bytes.

The single product change copies contiguous rows for eligible last-axis padding,
keeping the generic implementation for cropping, outer-axis and reflection cases.
[Full model correctness](models-20260926.md) matches current root exactly and
passes ORT bounds. The [exact ORT allocation evidence](../pad-memory-results/ort-allocation-20260926.md)
and [managed memory diagnosis](../pad-memory-results/managed-diagnosis-20260926.md)
selected this one application comparison; their instrumented clocks are not scores.

All six failed isolated repeatability controls remain recorded. Passing complete
transcription does not qualify isolated-call stability. Fresh shared/e5/Pyannote
regression and actual root/package qualification are required before promoting
source or BENCHMARK.md. Application admission alone grants no release admission;
a failed verdict permits no unchanged retry. The parity target remains separate.

[Every case](application-20260926.csv), [all 480 clocks](application-20260926-clocks.csv),
and [all controls, identities and resources](application-20260926.json).
Closure: `2b40b54d8e94de7326ceec5228ad1529c7513964211e8b5dfbb2c921cd798151`.
Raw evidence: `artifacts/parakeet-pad-current-app-amd-20260926`.
