# Observed-mask optimization: complete Parakeet comparison

**The candidate passes the unchanged application admission.** Complete
transcription latency is 4.436% lower than the current release.
Repeatability controls: 63/63. Performance gates: 21/21.
Candidate latency is 1.632486 times Microsoft ORT.

| Twenty-clip corpus (213.265 s audio) | Seconds | Relative to ORT |
|---|---:|---:|
| Current release | 67.709545 | 1.708271 |
| Observed-mask candidate | 64.705703 | 1.632486 |
| Microsoft ORT 1.29.0 | 39.636305 | 1.000000 |

Six fresh processes execute current, candidate, ORT, ORT, candidate, current.
Each runs one warmup and three measured passes over all 20 clips: 480 requests,
120 warmups and 360 measurements. Every raw clock contributes with equal
process weights. All original numerical, public-result, input-immutability
and held-output checks pass. This run uses the ordinary application consumers,
without a profiler, runtime override or pooled historical clock.

The fixed thresholds remain at least 3% corpus gain, no clip more than 5%
slower, corpus process repeatability at most 1.10 and per-clip at most 1.20
for each engine. The separate parity target is candidate/ORT at most 1.05.
All 2,790 resource observations pass;
peak owned RSS is 9,491,566,592 bytes.
Every process owner is terminal with code 0.

The [matched masking profile](profile-20260924.md) motivated this one candidate;
the [full model qualification](models-20260924.md) preserves both instruction
modes and native bounds. [Every case](application-20260924.csv) and
[all controls, gates, identities, process clocks and resources](application-20260924.json)
remain available. An application admission still requires shared/e5, Pyannote,
graph, meeting, normal-root/full-suite and package qualification before product
integration or BENCHMARK.md changes. A failed admission retains its verdict.

Closure: `4ed28b9161ecdf4d31b2bffc0e7e25b1e539bf68a78b8f80e0edfb4255b86752`.
Raw evidence: `artifacts/parakeet-observed-dense-where-app-amd-20260924`.
