# Contiguous padding: complete Pyannote comparison

**All application regression gates pass.**

| Fixture | Current seconds | Candidate seconds | Microsoft ORT seconds | Candidate / ORT | Candidate / current |
|---|---:|---:|---:|---:|---:|---:|
| dialogue-30s | 9.983378 | 9.955860 | 8.944933 | 1.113017 | 0.997244 |
| dialogue-0-10s | 0.459385 | 0.459647 | 0.428023 | 1.073885 | 1.000571 |
| dialogue-10-20s | 0.458451 | 0.459328 | 0.429872 | 1.068522 | 1.001912 |
| dialogue-20-30s | 0.457880 | 0.458146 | 0.429927 | 1.065635 | 1.000580 |

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

All nine workers and their supervisor are terminal. 1,710
resource observations pass; peak owned RSS is 1,326,698,496 bytes.

This comparison qualifies the same padding candidate already measured on
Parakeet. Its six failed isolated Pad repeatability controls remain recorded.
Actual root/package qualification still precedes release promotion and the
BENCHMARK.md update. A failed admission remains failed; no unchanged retry.

[Every clock](pyannote-application-clocks-20260926.csv),
[all timing setup intervals](pyannote-application-setups-20260926.csv),
[complete results, controls and identities](pyannote-application-20260926.json).

Closure: `60ea42f647c8c55b5ea97c289a69ccefd04e8b6cdb645c12daf996611e3fb7e8`.
Raw evidence: `artifacts/parakeet-pad-current-pyannote-app-amd-20260926`.
