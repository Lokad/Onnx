# 0.5.0 sustained audio and CPU regression evidence

Measured source `7c0f1f2ce3b97fc14e72a475741462baab33cc61`; actual package SHA256 `25e477cb8dd4ca003c7ac769a1d3f16c75ee25fdbfdd30c2f470e0e519e9c544`.
The package contains independently qualified .NET 10 and .NET 11 assets. Performance uses .NET 11 RC3 and Microsoft ORT 1.29.0 CPUExecutionProvider on the existing AMD EPYC 9V45 Standard_D4as_v7 VM. Workers run serially on CPU 2, controllers on CPU 0.

[Raw process/call records, fixtures and binary identities](release-0.5.0-sustained-20261030.json).

## Complete audio applications

| Application | Workload | Managed mean seconds | ORT mean seconds | Paired geometric ratio | 95% process-pair interval | <1.03 target |
|---|---|---:|---:|---:|---|---|
| Parakeet TDT 0.6B V3 | 20 clips / 213.265 seconds of audio | 17.870971 | 17.557958 | 1.017825 | [1.014959, 1.020700] | Pass |
| Pyannote Community-1 | complete 30-second dialogue | 3.882969 | 3.896787 | 0.996427 | [0.988092, 1.004831] | Pass |

Parakeet uses twenty-four independent managed/ORT process pairs, twelve in each order; Community uses twelve pairs, six in each order. Every process runs one first-use pass, six conditioning passes and six measured complete passes. Its score is the arithmetic mean of the six full-workload measured sums. The point estimate is the geometric mean of its independently paired process ratios; the interval is two-sided Student-t on their logarithms, df=23 for Parakeet and df=11 for Community. The ratio therefore need not equal the ratio of the displayed arithmetic mean times. Every process and call is retained; no optional stopping, trimming or favorable reruns.

Two additional original-baseline/ORT pairs per application bridge the historical six-pair sustained baseline to current conditions. They are descriptive controls, excluded from primary acceptance. The original measured Core/Data closure is unchanged. All six original/candidate/native conformance workers pass before timings.

| Pair | Parakeet managed / ORT | Community-1 managed / ORT |
|---|---:|---:|
| 1 | 1.021432 | 1.025150 |
| 2 | 1.022951 | 0.992572 |
| 3 | 1.019198 | 0.995890 |
| 4 | 1.019945 | 1.000721 |
| 5 | 1.024655 | 0.982793 |
| 6 | 1.020123 | 0.990009 |
| 7 | 1.004039 | 1.002738 |
| 8 | 1.014969 | 0.985990 |
| 9 | 1.022078 | 1.005362 |
| 10 | 1.010892 | 1.003224 |
| 11 | 1.006604 | 0.973193 |
| 12 | 0.998893 | 1.000438 |
| 13 | 1.025823 | — |
| 14 | 1.019534 | — |
| 15 | 1.018604 | — |
| 16 | 1.022671 | — |
| 17 | 1.021448 | — |
| 18 | 1.012292 | — |
| 19 | 1.022361 | — |
| 20 | 1.023199 | — |
| 21 | 1.020625 | — |
| 22 | 1.017332 | — |
| 23 | 1.021589 | — |
| 24 | 1.017074 | — |

| Original baseline bridge | Parakeet managed / ORT | Community-1 managed / ORT |
|---|---:|---:|
| 1 | 1.142419 | 1.013095 |
| 2 | 1.142926 | 1.024592 |

## Preparation, first use and memory

| Application / engine | Median setup seconds | Median first-use workload seconds | Median allocated MiB per measured workload | Median GC pause seconds per workload | Peak RSS MiB range |
|---|---:|---:|---:|---:|---|
| Parakeet TDT 0.6B V3 / Managed | 2.590 | 20.865 | 6517.3 | 0.014526 | 6161.2–6741.1 |
| Parakeet TDT 0.6B V3 / ORT | 1.925 | 17.566 | — | — | 2766.7–2771.6 |
| Pyannote Community-1 / Managed | 0.284 | 9.047 | 612.0 | 0.004049 | 922.6–1146.8 |
| Pyannote Community-1 / ORT | 0.188 | 4.494 | — | — | 647.9–656.9 |

Setup/loading and first use are separate from the sustained score. Allocation is cumulative managed allocation inside public calls, not retained memory or GC pause time. All per-call GC counters, pauses, heap observations and process RSS are preserved in the raw report. Measured-half trends above 2% are reported without discarding observations.

| Process with >2% measured-half change | Last three / first three workload means |
|---|---:|
| None | — |

## Non-audio regression controls

| Model | Current-package seconds | Candidate seconds | ORT seconds | Candidate/current | Candidate/ORT |
|---|---:|---:|---:|---:|---:|
| multilingual-e5-small / 30 tokens | 0.010101 | 0.010183 | 0.009336 | 1.008198 | 1.091 |
| EmbeddingGemma 2 FP32 / 128 tokens / 768 values | 0.173149 | 0.174886 | 0.171609 | 1.010034 | 1.019 |
| DINOv3 ViT-S/16 / 224x224 | 0.055462 | 0.056851 | 0.049462 | 1.025051 | 1.149 |
| ResNet50 / 224x224 | 0.047762 | 0.047870 | 0.032078 | 1.002263 | 1.492 |
| GPT-2 / four-token prefill | 0.025625 | 0.025593 | 0.017221 | 0.998742 | 1.486 |

These controls retain their previous per-model protocols, independently of the sustained-audio protocol. Graph processes use 1,200 warmups for e5 or 600 otherwise, then 180 calls and a process mean. Gemma retains its existing at-least-32-call/10-second steady warmup, 33 calls and process median. Each control uses two fresh processes for the qualified current package, candidate and native, in current/candidate/native/native/candidate/current order, with the median of each pair of process results. Direct source controls use current package b42f5095 / source 2074c006; those source controls are distinct from the descriptive original 8c5ba9 audio bridges. Each candidate/current score must be <=1.02. All calls and process values remain in the raw report. Small differences within those ranges are inconclusive; unrelated models are not required to meet the audio target.

DINOv3 initially MISSED the 1.02 direct source-control limit at 1.025051; that result remains a failure. A separate node-wall diagnostic measured only about 0.005 ms change in Concat against about 1.389 ms whole-call movement. A single prospectively declared corroboration then used six independent processes, each with separate old/new AssemblyLoadContexts and owned graphs, 600 alternating warmup pairs and 64 alternating measured call pairs. Each process contributes one geometric call-pair score; the six process scores give 0.999546, two-sided 95% log-Student-t interval [0.981786, 1.017627], df=5. Both point and upper satisfy <=1.02. This interleaved design shares GC/cache conditions and is not equivalent to the original isolated-process design. Neither result is pooled away, and every warmup/pair/process is retained. All complete arrays, input immutability and held-output ownership pass. The other four original source controls pass unchanged.

## Earlier observations retained

The earlier fixed package b42f5095/source2074c006 cohort passed Community at 1.003511 [0.998181,1.008869] and missed Parakeet at 1.030416 [1.024190,1.036679]. Its raw observations/proof remain unchanged and are not pooled into this cohort. All three subsequent copy/selection operator screens retained their unsuccessful synthetic controls; none is reclassified as a pass. The separate four-pair source trial on frozen3a5cc47c passed its declared whole-API retention gates, after which the reviewed source removed a capturing alias-check predicate. Those engineering clocks are not attributed to the final package. The present exact-package audio confidence, four passing direct source controls and separately qualified DinoV3 corroboration provide the acceptance evidence for this source. No timing cohort was extended or favorably repeated after inspecting its confidence endpoints.

## Correctness and provenance

Both framework assets pass offline installed-package consumers with zero core NuGet runtime dependencies, 5,928 ordinary tests, 5,709 hardware-disabled tests and 319 ORT differential cases per framework. The actual packaged cores also pass all 133 focused cases on real AVX-512 hardware per target. Complete managed/native audio conformance passes on both frameworks. Checks retain exact token/frame/speaker decisions, timeline tolerance 1e-12, centroid/graph scaled error <=1e-4, finite values, unchanged inputs, repeatability and held-output ownership. Gemma retains its component, relative-L2, cosine and unit-norm gates. Data and the Protobuf adapter remain nonpackable.

The bounded private Parakeet position transformation retains 24 projected tables totaling 73,826,304 bytes, retiring its original matrices and position table. All supported input lengths are covered by the existing 30-second request bound. The core convolution region retains live padding/fill, original-composition fallbacks and charged/returned scratch.

DINOv2 remains excluded under its existing failed full-output numerical gate; Whisper has no matched timing. Historical one-warmup/three-pass results remain in the earlier report and have not been reinterpreted as sustained measurements.

Audio proof: 15954 files, 54540209 bytes, SHA256 `c2b7aef20cbeecc15e0f5a9c276785e1bc7823fb886871630066de96c65af945`.
Non-audio proof: 498 files, 42295889 bytes, SHA256 `23ebcf05a20aa215d2ba9f2a3392e443e4951d4cbc0e8996c6d15293f990c58c`.
The accompanying raw JSON records the design, package/assembly/source identities, every process, every audio call clock/hash/counter and all control timing samples. These are workload-specific observations on one host.

Measured source CI is [green](https://github.com/Lokad/Onnx/actions/runs/40970290965); terminal observation 2026-10-29T23:24:53.181604+00:00.
The report commit is qualified separately in CI.
