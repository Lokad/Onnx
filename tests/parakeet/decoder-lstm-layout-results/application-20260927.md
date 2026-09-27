# Grouped LSTM weights: complete Parakeet comparison

**The candidate passes the prospective application admission.**
Complete transcription latency is 2.444% lower than current root.
Repeatability: 63/63 checks. Performance: 21/21 gates.
Candidate/ORT is **1.187749**; the independent
1.05 parity target is not met.

| Twenty-clip corpus (213.265 s audio) | Seconds | Relative to ORT |
| --- | ---: | ---: |
| current root baseline | 48.107529 | 1.217506 |
| LSTM layout candidate | 46.931723 | 1.187749 |
| Microsoft ORT 1.29.0 | 39.513180 | 1.000000 |

Six fresh processes run current root, candidate, ORT, ORT, candidate, current root on AMD EPYC
9V74 CPU 2. Each uses one warmup and three measured passes over every clip:
480 requests, 120 warmups and 360 measurements. The common consumers,
per-request validation and exact-clock aggregation are unchanged. No profiler or
runtime override is enabled. Products were built before this comparison; ordinary
JIT policy remains unchanged. No clock is trimmed.

The prospective limits require at least 1% corpus gain, no clip more than 5%
slower, corpus process max/min <= 1.10 and per-clip max/min <= 1.20 for all engines.
All workers are terminal/code0; all 2,194
resource samples pass. Peak owned RSS is 9,982,353,408 bytes.

The single product change groups prepared LSTM weights for the four vectors
already consumed together, preserving arithmetic, reduction order, vector width,
retained bytes and independently owned outputs. [Complete model correctness](models-20260927.md)
passes the original ORT bound, byte-exact current/candidate tensors, decoder
decisions/transcripts and ownership checks in normal and AVX512-disabled modes.
The [exact export and pinned ORT review](../decoder-lstm-current-review/next-decision-20260927.md)
identified compact prepared-weight groups versus Lokad's 10,240-byte reduction stride.

The [component screen](timing-20260927.md) remains rejected: 100/168 repeatability
controls fail, despite all raw performance thresholds passing. A separate
[runtime diagnosis](../decoder-lstm-runtime-results/README.md) establishes JIT
compilation inside its measured region, including an LSTM compilation inside a
call and background compilation on the same CPU. Tracing moves the peak pass;
prepared-path variation remains unresolved. Those elapsed overlaps are not
additive savings, and no old clock is corrected. This independent application
result changes no failed verdict and establishes no isolated LSTM speedup or
universal fallback equivalence.

Next: retain the diagnosed component limitation explicitly and require shared/e5, Pyannote, graph and actual-root/package qualification before any source or BENCHMARK.md promotion.
The current release and BENCHMARK.md stay unchanged. Parity remains a separate goal.

[Every case](application-20260927.csv), [all 480 clocks](application-20260927-clocks.csv),
and [all controls, identities and resources](application-20260927.json).
Closure: `e3ba182a323209926e26c885d190875b04087d430f5e97bed41f460031b4883b`.
Raw evidence: `artifacts/parakeet-decoder-lstm-layout-app-amd-20260927`.
