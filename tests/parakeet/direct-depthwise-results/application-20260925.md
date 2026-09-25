# Direct depthwise: complete Parakeet comparison

**The candidate passes the unchanged application admission.**
Complete transcription latency is 4.991% lower than M78.
Repeatability: 63/63 checks. Performance: 21/21 gates.
Candidate/ORT is **1.360769**; the independent
1.05 parity target is not met.

| Twenty-clip corpus (213.265 s audio) | Seconds | Relative to ORT |
| --- | ---: | ---: |
| M78 baseline | 56.290885 | 1.432259 |
| Direct-depthwise candidate | 53.481156 | 1.360769 |
| Microsoft ORT 1.29.0 | 39.302157 | 1.000000 |

Six fresh processes run M78, candidate, ORT, ORT, candidate, M78 on AMD EPYC
9V74 CPU 2. Each uses one warmup and three measured passes over every clip:
480 requests, 120 warmups and 360 measurements. The common consumers,
per-request validation and exact-clock scorer are unchanged. There is no
profiling, compilation or runtime override in this comparison. No clock is trimmed.

The prospective limits require at least 3% corpus gain, no clip more than 5%
slower, corpus process max/min <= 1.10 and per-clip max/min <= 1.20 for all engines.
All workers are terminal/code0; all 2,424
resource samples pass. Peak owned RSS is 9,954,451,456 bytes.

The single product change directly accumulates nine-tap depthwise convolutions.
[Full model correctness](models-20260925.md) matches M78 exactly and passes ORT
bounds. [Mechanism counters](mechanism-20260925.md) confirm removal of the
targeted matrix calls, temporary views and patch writes at every observed shape.
The present application comparison measures the benefit; diagnostic timings
are not substituted for these clocks or composed with previous improvements.

This comparison uses isolated M78 as its baseline. M78's separately retained
[short-e5 regression](../packed-final-row-results/graphs-20260925.md) remains
unresolved. Fresh shared/e5/Pyannote regression and actual root/package qualification
are required before promoting source or BENCHMARK.md. Application admission
alone grants no release admission; a failed verdict permits no unchanged retry.

[Every case](application-20260925.csv), [all 480 clocks](application-20260925-clocks.csv),
and [all controls, identities and resources](application-20260925.json).
Closure: `e931b8b165a3f2a7cb5ff3f057de8378b42c7133e6a0637aad3207cf5da68e0a`.
Raw evidence: `artifacts/parakeet-direct-depthwise-app-amd-20260925`.
