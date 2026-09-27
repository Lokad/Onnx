# Rational sigmoid: complete Pyannote comparison

**All application regression gates pass.**

| Fixture | Current seconds | Candidate seconds | Microsoft ORT seconds | Candidate / ORT | Candidate / current |
|---|---:|---:|---:|---:|---:|---:|
| dialogue-30s | 9.943906 | 9.965163 | 8.945839 | 1.113944 | 1.002138 |
| dialogue-0-10s | 0.455777 | 0.460087 | 0.428209 | 1.074445 | 1.009455 |
| dialogue-10-20s | 0.459689 | 0.458726 | 0.429805 | 1.067287 | 0.997905 |
| dialogue-20-30s | 0.458091 | 0.459115 | 0.430278 | 1.067020 | 1.002236 |

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

All nine workers and their supervisor are terminal. 1,700
resource observations pass; peak owned RSS is 1,203,200,000 bytes.

This comparison evaluates the same rational sigmoid product measured on
Parakeet. Its isolated operator screen remains rejected, including thirteen
repeatability controls and four fallback cases. Double latency remains
unresolved; this result does not erase that documented limitation.
Actual root/package qualification still precedes release promotion and the
BENCHMARK.md update. A failed admission remains failed; no unchanged retry.

[Every clock](pyannote-application-clocks-20260927.csv),
[all timing setup intervals](pyannote-application-setups-20260927.csv),
[complete results, controls and identities](pyannote-application-20260927.json).

Closure: `f1c4e873f19f5562b2da8a31a59ec5b54f47959b8f4c1c996ec6c996f5e2218d`.
Raw evidence: `artifacts/parakeet-rational-sigmoid-pyannote-app-amd-20260927`.
