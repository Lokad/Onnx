# Tiled axis movement: complete Pyannote comparison

**All application regression gates pass.**

| Fixture | Current seconds | Candidate seconds | Microsoft ORT seconds | Candidate / ORT | Candidate / current |
|---|---:|---:|---:|---:|---:|---:|
| dialogue-30s | 9.915714 | 9.920118 | 8.949371 | 1.108471 | 1.000444 |
| dialogue-0-10s | 0.460890 | 0.459767 | 0.428479 | 1.073020 | 0.997564 |
| dialogue-10-20s | 0.461696 | 0.462340 | 0.429740 | 1.075858 | 1.001395 |
| dialogue-20-30s | 0.461351 | 0.457636 | 0.430014 | 1.064234 | 0.991948 |

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

All nine workers and their supervisor are terminal. 1,714
resource observations pass; peak owned RSS is 1,431,957,504 bytes.

This comparison evaluates the unchanged transpose dispatch change selected on Parakeet.
The [complete Parakeet comparison](application-20260928.md) admits a
2.420895% matched gain. Arithmetic leaves and cache limits remain unchanged.
Actual root/package qualification still precedes release promotion and the
BENCHMARK.md update. A failed admission remains failed; no unchanged retry.

[Every clock](pyannote-application-clocks-20260928.csv),
[all timing setup intervals](pyannote-application-setups-20260928.csv),
[complete results, controls and identities](pyannote-application-20260928.json).

Closure: `a25671c2480b7b9db3ec03b9e24a97253cc75d9464715515237c26b785c36899`.
Raw evidence: `artifacts/parakeet-transpose-axis-pyannote-app-amd-20260928`.
