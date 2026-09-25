# Packed final row: complete Parakeet comparison

**The candidate passes the unchanged application admission.** Complete
transcription latency is 10.479% lower than the qualified release baseline.
Repeatability controls: 63/63. Performance gates: 21/21.
Candidate latency is 1.424132 times Microsoft ORT.
The separate 1.05 parity target is not met.

| Twenty-clip corpus (213.265 s audio) | Seconds | Relative to ORT |
|---|---:|---:|
| Qualified release baseline | 62.371370 | 1.590842 |
| M78 packed final-row candidate | 55.835237 | 1.424132 |
| Microsoft ORT 1.29.0 | 39.206516 | 1.000000 |

Six fresh processes execute selected, candidate, ORT, ORT, candidate, selected.
Each runs one warmup and three measured passes over all 20 clips: 480 requests,
120 warmups and 360 measurements. Every clock contributes with equal process
weights. Ordinary application consumers run without profiling or runtime
overrides. Numerical, public-result, immutable-input and held-output checks pass.

The fixed thresholds require at least 3% corpus gain, no clip more than 5%
slower, corpus process repeatability at most 1.10 and per-clip at most 1.20
for each engine. Every process is terminal with code 0; all
2,560 resource observations pass.
Peak owned RSS is 10,092,752,896 bytes.

This evaluates the complete owned-weight candidate with the targeted
remaining-row correction identified from the installed ORT path. It retains
87 constant weights in packed form and computes the remaining row directly
from that storage. Existing two-/three-row arithmetic, the nine already prepared
feed-forward weights and the clone budget are unchanged.
[Complete correctness](models-20260925.md) passes in both instruction modes;
[actual counters](counters-20260925.md) prove zero reconstruction events and
1,740 avoided transient packs per corpus. These counts are not independent
application savings. This comparison measures M78 directly against qualified
release Core f95a13c5/Data a893952f. Exact model/public lineage binds release,
intermediate M73 and M78; historical performance ratios are not composed.

The separately retained M73 graph failures keep their original verdicts.
The completed [M78 graph comparison](graphs-20260925.md) fails: e5-8tok candidate/release 1.077654.
This application result alone does not promote product source or BENCHMARK.md.
Every graph gate, Pyannote application admission and actual root/package
qualification are also required. A failed admission permits no unchanged retry.

[Every case](release-application-20260925.csv) and
[all controls, gates, identities, process clocks and resources](release-application-20260925.json)
are retained, including the two prior failed release controls.

Closure: `db3dd71fe7196b5b9e2b12c9e3ff172e2397ad526c290b79f59cf63743892af6`.
Raw evidence: `artifacts/parakeet-packed-final-row-release-app-amd-20260925`.
