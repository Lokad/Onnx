# Prepared single-row weights: complete Pyannote comparison

**All application regression gates pass.**

| Fixture | Current seconds | Candidate seconds | Microsoft ORT seconds | Candidate / ORT | Candidate / current |
|---|---:|---:|---:|---:|---:|---:|
| dialogue-30s | 10.043519 | 9.988436 | 8.955742 | 1.115311 | 0.994516 |
| dialogue-0-10s | 0.466606 | 0.461913 | 0.428685 | 1.077511 | 0.989944 |
| dialogue-10-20s | 0.466101 | 0.461759 | 0.430814 | 1.071829 | 0.990685 |
| dialogue-20-30s | 0.460721 | 0.462341 | 0.430863 | 1.073058 | 1.003518 |

Repeatability: 12/12 controls pass; full-dialogue process max/min <= 1.10, crops <= 1.20.
Regression: 4/4 gates pass; candidate/current <= 1.05.

AMD EPYC 9V74 CPU 2, .NET 10.0.8 and ORT 1.29.0 CPUExecutionProvider.
The clock includes frontend, segmentation, embeddings, clustering and owned
public results. Setup and validation are separate. Six fresh processes run
current, candidate, ORT, ORT, candidate, current. Every process uses one
warmup and three measured passes over each of the four fixtures.
All 96 requests, including 24 warmups and 72 measurements, remain in the
published clocks. The unchanged scorer retains every clock with equal
process weights and exact clock fractions. No profiling or timing correction.

Four fresh native public requests pass the original conformance bounds.
Candidate timed public results equal the current root exactly, including
centroids. Both 600-second meetings and the 30-second recovery preserve
the original complete reference results and the native/portable bounds.
Input immutability and held-output ownership checks remain intact.
These checks establish compatibility; they do not measure human-label accuracy.

All nine workers and their supervisor are terminal. 1,731
resource observations pass; peak owned RSS is 1,282,584,576 bytes.

This comparison evaluates the same prepared-row product measured on
Parakeet. Its isolated operator screen remains rejected: two repeatability
controls and two fallback cases failed. The first-call-only explanation
was also rejected; cache competition remains an inference. This result
preserves both verdicts and establishes no universal fallback performance
equivalence. See the [screen](screen-20260927.md) and
[per-call diagnosis](unmapped-calls-20260927.md).
Actual root/package qualification still precedes release promotion and the
BENCHMARK.md update. A failed admission remains failed; no unchanged retry.

[Every clock](pyannote-application-clocks-20260927.csv),
[all timing setup intervals](pyannote-application-setups-20260927.csv),
[complete results, controls and identities](pyannote-application-20260927.json).

Closure: `17861b3815ecb72d636fc86562e2c8605d01af9cc561841f069d5d1c23e14188`.
Raw evidence: `artifacts/parakeet-decoder-packed-row-pyannote-app-amd-20260927`.
