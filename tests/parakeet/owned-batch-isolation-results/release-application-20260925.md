# Dispatch relocation: complete Parakeet comparison against the qualified release

**The candidate passes the prospective application gates.**
Complete transcription latency is 14.862% lower than the qualified release.
Repeatability: 63/63 checks. Performance: 21/21 gates.
Candidate/ORT is **1.354514**; the separate
1.05 parity target is not met.

| Twenty-clip corpus (213.265 s audio) | Seconds | Relative to ORT |
| --- | ---: | ---: |
| Qualified release | 62.377845 | 1.590959 |
| Dispatch-relocation candidate | 53.107381 | 1.354514 |
| Microsoft ORT 1.29.0 | 39.207695 | 1.000000 |

Six fresh processes run baseline, candidate, ORT, ORT, candidate, baseline on
AMD EPYC 9V74 CPU 2. Each uses one warmup and three measured passes per clip:
480 complete requests, 120 warmups and 360 measurements. The existing public
consumer, numerical validators and exact-clock scorer are unchanged. No clock
is trimmed, and no profiling or runtime override is enabled.

The prospective limits require at least 3% corpus improvement over the qualified release,
no clip more than 5% slower, corpus process max/min <=1.10 and per-clip max/min
<=1.20 for all three engines. All workers are terminal/code0. All
2,511 resource samples pass;
peak owned RSS is 9,554,628,608 bytes.

This compares the actual release and candidate directly, with their respective Core and Data binaries. Retained outputs for these exact products also match byte for byte across 1,568 tensor pairs and 40 complete public result pairs in both instruction modes. No historical times are pooled.

[Full Parakeet correctness](models-20260925.md) and
[combined graph qualification](../../benchmarks/e5-steady-short-results/qualified-graphs-20260925.md)
are separate completed proofs. Shared/Pyannote and actual root/package
qualification remain required before promotion. Root source and BENCHMARK.md
are unchanged. A failed verdict does not permit an unchanged retry.

[Every case](release-application-20260925.csv), [all 480 clocks](release-application-20260925-clocks.csv), and
[all controls, product identities and resources](release-application-20260925.json).
Closure: `e2ddd739372df25f93323a717f97b8cc08999faa143f67ede16130e32b572f03`.
Raw evidence: `artifacts/parakeet-owned-batch-isolation-release-app-amd-20260925`.
