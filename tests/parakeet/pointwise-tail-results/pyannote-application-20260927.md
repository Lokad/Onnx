# Pointwise remainder sharing: complete Pyannote comparison

**All application regression gates pass.**

| Fixture | Current seconds | Candidate seconds | Microsoft ORT seconds | Candidate / ORT | Candidate / current |
|---|---:|---:|---:|---:|---:|---:|
| dialogue-30s | 9.964000 | 9.884276 | 8.942847 | 1.105272 | 0.991999 |
| dialogue-0-10s | 0.458568 | 0.459032 | 0.428002 | 1.072499 | 1.001012 |
| dialogue-10-20s | 0.457021 | 0.459335 | 0.429867 | 1.068552 | 1.005064 |
| dialogue-20-30s | 0.460485 | 0.460740 | 0.429894 | 1.071753 | 1.000555 |

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

All nine workers and their supervisor are terminal. 1,720
resource observations pass; peak owned RSS is 1,491,300,352 bytes.

This comparison evaluates the unchanged pointwise remainder change selected on Parakeet.
Its [component screen](../decoder-lstm-layout-profile-results/pointwise-tail-timing-20260927.md) remains rejected because
38/246 repeatability controls failed. The
[runtime diagnosis](../decoder-lstm-layout-profile-results/pointwise-tail-runtime-20260927.md) finds
JIT activity within measured calls, but does not correct or rescore
the original clocks or resolve all component variation.
Actual root/package qualification still precedes release promotion and the
BENCHMARK.md update. A failed admission remains failed; no unchanged retry.

[Every clock](pyannote-application-clocks-20260927.csv),
[all timing setup intervals](pyannote-application-setups-20260927.csv),
[complete results, controls and identities](pyannote-application-20260927.json).

Closure: `7e0b08fc21b487d9e945e00133ff8394e9b3ab825a8cac3f6213e61b9406d4c4`.
Raw evidence: `artifacts/parakeet-pointwise-tail-pyannote-app-amd-20260927`.
