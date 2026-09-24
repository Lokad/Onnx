# Combined recurrence and slice changes: complete Parakeet comparison

The combined candidate reduces complete transcription latency by **9.477%**.
All 63 repeatability controls and 21 performance gates pass. Every clip improves;
the smallest improvement is 4.716%. The candidate remains
**1.700 times Microsoft ORT latency**, above the 1.05 parity target.

| Twenty-clip corpus (213.265s audio) | Seconds | Relative to ORT |
|---|---:|---:|
| Selected release | 74.476402 | 1.877963 |
| Combined candidate | 67.418614 | 1.699997 |
| Microsoft ORT 1.29.0 | 39.658078 | 1.000000 |

These are fresh measurements of the composition, not added standalone gains.
Six fresh processes run selected/candidate/ORT/ORT/candidate/selected, each with
one warmup and three measured corpus passes: 480 requests, 120 warmups and
360 measured requests. Equal process weights retain every original clock.
All original numerical, transcript, input and held-output checks pass.
The existing corpus gain threshold is 3%; no clip may regress more than 5%.
Corpus repeatability is bounded at 1.10 and every clip at 1.20 for each engine.

The candidate is Core37c24375/Data cc37b19e, already qualified by the
[compiled/tensor review](qualification-20260924.md) and
[complete both-mode model checks](models-20260924.md). No third optimization
is included. All 2,941 resource observations pass;
peak owned RSS is 9,271,640,064 bytes.
Supervisor944229/birth1790254959.39 and every worker are terminal, code0.

Closure: `f04c09fbc6c0455c4420d6680d60bb4f8fc5cac9dda2f2507768ba94b86335e4`.
Raw evidence: `artifacts/parakeet-validated-composition-app-amd-20260924`.
[Every clip](application-clips-20260924.csv) and
[all controls, identities, clocks and resources](application-20260924.json).

The auditor console output was inadvertently opened inside its recursive
inventory. Its complete output is retained separately; the empty pre-print
placeholder declared in the unchanged closure was restored and every closed
file reverified. The linked JSON identifies that reconciliation receipt.
No input, request clock, analysis, scoring rule or closure bytes changed.

Shared models/e5, Pyannote, warmed release graphs, meetings, normal root/full
suites and package consumption remain required before release integration.
BENCHMARK.md continues to describe the qualified release.
