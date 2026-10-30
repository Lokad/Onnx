# 0.5.0 integrated-package audio qualification

Measured source `d5f8ecad10da83f898eb4c21f45c1e0eb6f942f0`; package SHA256 `8b5066108b4306aeef4dd16947fcc2f4e7a3e95099d53a892ed4b4081b0a358b`.

This candidate restores ONNX import, conversions and generated messages to the core package. Google.Protobuf 3.33.5 or newer is its only runtime dependency. Data and CLI remain nonpackable. Offline package-only consumers pass on .NET 10 and .NET 11.

Performance uses .NET 11 RC3 and Microsoft ORT 1.29.0 CPUExecutionProvider on the existing AMD EPYC 9V45 Standard_D4as_v7 VM, one logical CPU. Numerical workers run serially on CPU 2; the controller uses CPU 0.

| Application | Managed mean seconds | ORT mean seconds | Paired geometric ratio | Two-sided 95% interval | <1.03 point and upper |
|---|---:|---:|---:|---|---|
| Parakeet TDT 0.6B V3 | 18.543500 | 18.276449 | 1.014606 | [1.011618, 1.017603] | Pass |
| Pyannote Community-1 | 4.042853 | 4.046777 | 0.998969 | [0.991165, 1.006835] | Pass |

Parakeet transcribes twenty clips totaling 213.265 seconds; Community diarizes a complete 30-second dialogue. Both include frontend/model calls, decoding or clustering, and owned public results. Loading, preparation and validation are outside timing.

The fixed cohort contains four candidate/native conformance workers followed by 24 independent Parakeet process pairs and 12 Community pairs, balanced in engine order. Each process runs one first-use pass, six conditioning passes and six measured complete passes. A process score is the arithmetic mean of its six workload sums. Ratios use geometric means of paired process ratios and two-sided Student-t intervals on their logarithms (df=23 and df=11). No trimming, optional stopping or favorable appended pairs.

The earlier original-source/native bridges remain in the [pre-reintegration report](release-0.5.0-sustained-20261030.md). They are not repeated or pooled into this population. The primary 24/12-pair protocol and confidence criterion are unchanged. Current staged applications contain no separate parser assembly.

| Pair | Parakeet managed / ORT | Community managed / ORT |
|---|---:|---:|
| 1 | 1.010771 | 1.005745 |
| 2 | 1.012689 | 0.984455 |
| 3 | 1.006001 | 0.987158 |
| 4 | 1.021821 | 1.004911 |
| 5 | 1.004864 | 0.991487 |
| 6 | 1.017768 | 0.996041 |
| 7 | 1.016938 | 0.988796 |
| 8 | 1.003401 | 0.989271 |
| 9 | 1.023561 | 1.011171 |
| 10 | 1.010719 | 0.994102 |
| 11 | 1.014246 | 1.025477 |
| 12 | 1.006445 | 1.009852 |
| 13 | 1.010223 | — |
| 14 | 1.020463 | — |
| 15 | 1.019339 | — |
| 16 | 1.003237 | — |
| 17 | 1.020910 | — |
| 18 | 1.026874 | — |
| 19 | 1.023756 | — |
| 20 | 1.019561 | — |
| 21 | 1.019760 | — |
| 22 | 1.019366 | — |
| 23 | 1.008708 | — |
| 24 | 1.009703 | — |

## Setup and memory

| Application / engine | Median setup seconds | Median first-use seconds | Allocated MiB / measured workload | GC pause seconds / measured workload | Peak RSS MiB range |
|---|---:|---:|---:|---:|---|
| Parakeet TDT 0.6B V3 / lok | 2.703 | 21.775 | 6517.3 | 0.014829 | 6159.6–8421.5 |
| Parakeet TDT 0.6B V3 / ort | 2.001 | 18.274 | — | — | 2767.1–2771.6 |
| Pyannote Community-1 / lok | 0.295 | 9.126 | 612.0 | 0.005341 | 1027.0–1121.6 |
| Pyannote Community-1 / ort | 0.197 | 4.682 | — | — | 648.9–656.9 |

All per-call clocks, allocation/GC counters and process records are retained in the [raw JSON](release-0.5.0-integrated-20261030.json). Allocation is cumulative managed allocation inside public calls, not retained memory. Measured-half trends over 2% are disclosed without discarding observations.

| Process with >2% measured-half change | Last three / first three workload means |
|---|---:|
| pyannote / pair 10 / lok | 0.978646 |
| parakeet / pair 18 / lok | 0.976412 |

## Correctness and source correspondence

The actual packaged assemblies match every compiled method in committed-source builds: 3,298 method bodies for .NET 11 and 3,295 for .NET 10, each across 291 types. Ordinary tests pass 5,918 cases per framework; hardware-disabled tests pass 5,699; native differential tests pass 319. Package-only consumers exercise generated messages, conversions, DTOs, file/byte/memory loading and ownership, with only a Lokad.Onnx package reference.

The actual packaged cores pass 133 focused tests per target on AVX-512 hardware. Private-model inspection verifies 24 Conformer regions, 48 pointwise payloads and 24 position tables retaining 73,826,304 bytes. Complete managed/native audio conformance passes on both frameworks, preserving exact token/frame/status/speaker decisions, timeline tolerance 1e-12, centroid/graph scaled error <=1e-4, finite values, input immutability, repeatability and held-output ownership.

Ten fresh workers also verify full outputs for e5, DINOv3, ResNet50, GPT-2 and EmbeddingGemma 2 at 128 tokens/768 values. Graph arrays are independently rechecked against native at scaled error <=1e-4; Gemma retains its existing component, relative-L2, cosine and norm gates. These are correctness observations; their prior timings remain attributed to the earlier source.

Numeric kernels and Data workflows are unchanged from the pre-reintegration source. The core change relocates the maintained importer/generated messages and removes introduced reader injection. Earlier failed synthetic controls, the first missed Parakeet interval and the DinoV3 source-control failure plus its separate corroboration remain in the previous report; none is erased or reclassified. DINOv2 remains excluded under its full-output gate; Whisper has no matched timing.

Proof: 14760 files, 212512618 bytes, SHA256 `56ce007adcaa7ce3a7fdc82f9d3c8a020d7df742534145f714a50b36f2fb1953`.

Measured-source CI is [green](https://github.com/Lokad/Onnx/actions/runs/41031677564); the report commit is checked separately. No NuGet publication or release tag is performed by this qualification.
