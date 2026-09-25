# Packed final row: complete Parakeet comparison

**The candidate passes the unchanged application admission.** Complete
transcription latency is 5.299% lower than the isolated M73 baseline.
Repeatability controls: 63/63. Performance gates: 21/21.
Candidate latency is 1.438313 times Microsoft ORT.
The separate 1.05 parity target is not met.

| Twenty-clip corpus (213.265 s audio) | Seconds | Relative to ORT |
|---|---:|---:|
| Isolated M73 baseline | 60.177045 | 1.518788 |
| M78 packed final-row candidate | 56.988487 | 1.438313 |
| Microsoft ORT 1.29.0 | 39.621750 | 1.000000 |

Six fresh processes execute selected, candidate, ORT, ORT, candidate, selected.
Each runs one warmup and three measured passes over all 20 clips: 480 requests,
120 warmups and 360 measurements. Every clock contributes with equal process
weights. Ordinary application consumers run without profiling or runtime
overrides. Numerical, public-result, immutable-input and held-output checks pass.

The fixed thresholds require at least 3% corpus gain, no clip more than 5%
slower, corpus process repeatability at most 1.10 and per-clip at most 1.20
for each engine. Every process is terminal with code 0; all
2,550 resource observations pass.
Peak owned RSS is 9,462,562,816 bytes.

This evaluates the complete owned-weight candidate with the targeted
remaining-row correction identified from the installed ORT path. It retains
87 constant weights in packed form and computes the remaining row directly
from that storage. Existing two-/three-row arithmetic, the nine already prepared
feed-forward weights and the clone budget are unchanged.
[Complete correctness](models-20260925.md) passes in both instruction modes;
[actual counters](counters-20260925.md) prove zero reconstruction events and
1,740 avoided transient packs per corpus. These counts are not independent
application savings. This fresh comparison measures the combined candidate
against M73; it does not isolate an incremental gain against historical M76.

The selected baseline is isolated M73, not the qualified repository release.
The prior e5-8tok and e5-512tok repeatability failures remain release blockers.
This result changes neither product source nor BENCHMARK.md. Shared models,
Pyannote, root/package and full-suite admission remain required before release.
A failed application admission retains its verdict and permits no unchanged retry.

[Every case](application-20260925.csv) and
[all controls, gates, identities, process clocks and resources](application-20260925.json)
are retained, including the two prior failed release controls.

Closure: `b90aa8fc3949c756cc7c55e636d2a8741c31fdb8d336174502639a323ebc0c06`.
Raw evidence: `artifacts/parakeet-packed-final-row-app-amd-20260925`.
