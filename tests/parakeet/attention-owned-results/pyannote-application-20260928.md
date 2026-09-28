# Owned attention weights: complete Pyannote comparison

**All application regression gates pass.**

| Fixture | Current seconds | Candidate seconds | Microsoft ORT seconds | Candidate / ORT | Candidate / current |
|---|---:|---:|---:|---:|---:|---:|
| dialogue-30s | 9.985585 | 9.986467 | 8.944572 | 1.116483 | 1.000088 |
| dialogue-0-10s | 0.461282 | 0.458245 | 0.428581 | 1.069215 | 0.993417 |
| dialogue-10-20s | 0.462190 | 0.458232 | 0.429734 | 1.066315 | 0.991438 |
| dialogue-20-30s | 0.461192 | 0.459999 | 0.429880 | 1.070065 | 0.997414 |

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

All nine workers and their supervisor are terminal. 1,726
resource observations pass; peak owned RSS is 1,199,890,432 bytes.

This comparison evaluates the unchanged attention preparation change selected on Parakeet.
The [complete Parakeet comparison](application-20260928.md) admits a
1.128490% matched gain. Arithmetic leaves and cache limits remain unchanged.
Actual root/package qualification still precedes release promotion and the
BENCHMARK.md update. A failed admission remains failed; no unchanged retry.

[Every clock](pyannote-application-clocks-20260928.csv),
[all timing setup intervals](pyannote-application-setups-20260928.csv),
[complete results, controls and identities](pyannote-application-20260928.json).

Closure: `d74cdd3d0aa6e15e9d24dba323eec0f30b38fd5ea7e1b011d2ec7cbad94922bf`.
Raw evidence: `artifacts/parakeet-attention-owned-pyannote-app-amd-20260928`.
