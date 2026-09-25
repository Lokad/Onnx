# Owned packed weights: complete Parakeet comparison

**The candidate fails the unchanged application admission.** Complete
transcription latency is 2.590% lower than the isolated M73 baseline.
Repeatability controls: 63/63. Performance gates: 20/21.
Candidate latency is 1.473371 times Microsoft ORT.
The separate 1.05 parity target is not met.

Failed controls or gates:

- Performance: corpus-at-least-three-percent-gain, candidate/selected 0.974102 > 0.97.

| Twenty-clip corpus (213.265 s audio) | Seconds | Relative to ORT |
|---|---:|---:|
| Isolated M73 baseline | 59.360823 | 1.512543 |
| Corrected M76 candidate | 57.823506 | 1.473371 |
| Microsoft ORT 1.29.0 | 39.245713 | 1.000000 |

Six fresh processes execute selected, candidate, ORT, ORT, candidate, selected.
Each runs one warmup and three measured passes over all 20 clips: 480 requests,
120 warmups and 360 measurements. Every clock contributes with equal process
weights. Ordinary application consumers run without profiling or runtime
overrides. Numerical, public-result, immutable-input and held-output checks pass.

The fixed thresholds require at least 3% corpus gain, no clip more than 5%
slower, corpus process repeatability at most 1.10 and per-clip at most 1.20
for each engine. Every process is terminal with code 0; all
2,547 resource observations pass.
Peak owned RSS is 9,527,910,400 bytes.

This tests one change motivated by the exact installed ORT and the measured
[feed-forward costs](../feed-forward-cost-results/diagnosis-20260925.md): retain
87 constant weights in the existing packed layout. Multiplication kernels,
nine already prepared feed-forward weights and the clone budget are unchanged.
[Complete correctness](models-20260925.md) passes in both instruction modes.
[Actual packing and reconstruction counts](counters-20260925.md) prove the
intended path, including the odd-row fallback copies; those counters are not
independent application savings.

The selected baseline is isolated M73, not the qualified repository release.
The prior e5-8tok and e5-512tok repeatability failures remain release blockers.
This result changes neither product source nor BENCHMARK.md. Shared models,
Pyannote, root/package and full-suite admission remain required before release.
A failed application admission retains its verdict and permits no unchanged retry.

[Every case](application-20260925.csv) and
[all controls, gates, identities, process clocks and resources](application-20260925.json)
are retained, including the two prior failed release controls.

Closure: `959d8d180db3431e1aeb5d7ab555fd9b77035ae84d4a8a678d2204cccc71eb48`.
Raw evidence: `artifacts/parakeet-owned-packed-weight-app-amd-20260925`.
