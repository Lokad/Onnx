# Dispatch relocation: complete Parakeet comparison against the direct-depthwise parent

**The candidate passes the prospective application gates.**
Complete transcription latency is 0.105% higher than the direct-depthwise parent.
Repeatability: 63/63 checks. Performance: 21/21 gates.
Candidate/ORT is **1.361401**; the separate
1.05 parity target is not met.

| Twenty-clip corpus (213.265 s audio) | Seconds | Relative to ORT |
| --- | ---: | ---: |
| Direct-depthwise parent | 53.304778 | 1.359967 |
| Dispatch-relocation candidate | 53.361009 | 1.361401 |
| Microsoft ORT 1.29.0 | 39.195646 | 1.000000 |

Six fresh processes run baseline, candidate, ORT, ORT, candidate, baseline on
AMD EPYC 9V74 CPU 2. Each uses one warmup and three measured passes per clip:
480 complete requests, 120 warmups and 360 measurements. The existing public
consumer, numerical validators and exact-clock scorer are unchanged. No clock
is trimmed, and no profiling or runtime override is enabled.

The prospective limits require at most 5% corpus regression against the direct-depthwise parent,
no clip more than 5% slower, corpus process max/min <=1.10 and per-clip max/min
<=1.20 for all three engines. All workers are terminal/code0. All
2,367 resource samples pass;
peak owned RSS is 10,211,139,584 bytes.

This is a retention check after moving the packed-weight dispatch decision. It does not require another optimization gain against the successful depthwise parent. The separate direct-release comparison retains its original 3% improvement requirement.

[Full Parakeet correctness](models-20260925.md) and
[combined graph qualification](../../benchmarks/e5-steady-short-results/qualified-graphs-20260925.md)
are separate completed proofs. Shared/Pyannote and actual root/package
qualification remain required before promotion. Root source and BENCHMARK.md
are unchanged. A failed verdict does not permit an unchanged retry.

[Every case](parent-application-20260925.csv), [all 480 clocks](parent-application-20260925-clocks.csv), and
[all controls, product identities and resources](parent-application-20260925.json).
Closure: `e63185755882fe1f3c7c19c5601ff0e1c9975afce91aa29bd271631c53c0bf13`.
Raw evidence: `artifacts/parakeet-owned-batch-isolation-parent-app-amd-20260925`.
