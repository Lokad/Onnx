# Prepared LSTM layout: complete Pyannote comparison

**All application regression gates pass.**

| Fixture | Current seconds | Candidate seconds | Microsoft ORT seconds | Candidate / ORT | Candidate / current |
|---|---:|---:|---:|---:|---:|---:|
| dialogue-30s | 10.313033 | 10.244691 | 9.022760 | 1.135428 | 0.993373 |
| dialogue-0-10s | 0.478297 | 0.473070 | 0.431733 | 1.095747 | 0.989072 |
| dialogue-10-20s | 0.474258 | 0.471228 | 0.435507 | 1.082023 | 0.993611 |
| dialogue-20-30s | 0.476782 | 0.478772 | 0.438794 | 1.091109 | 1.004174 |

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

All nine workers and their supervisor are terminal. 1,764
resource observations pass; peak owned RSS is 1,201,655,808 bytes.

This comparison evaluates the unchanged LSTM layout selected on Parakeet.
Its [component screen](timing-20260927.md) remains rejected because
100/168 repeatability controls failed. The
[runtime diagnosis](../decoder-lstm-runtime-results/README.md) finds
JIT activity within measured calls, but does not correct or rescore
the original clocks or resolve all prepared-path variation.
Actual root/package qualification still precedes release promotion and the
BENCHMARK.md update. A failed admission remains failed; no unchanged retry.

[Every clock](pyannote-application-clocks-20260927.csv),
[all timing setup intervals](pyannote-application-setups-20260927.csv),
[complete results, controls and identities](pyannote-application-20260927.json).

Closure: `5942b121dadb101c52e4221b7e1143d0c7ac37719d0de8db1e64f54f990e1404`.
Raw evidence: `artifacts/parakeet-decoder-lstm-layout-pyannote-app-amd-20260927`.
