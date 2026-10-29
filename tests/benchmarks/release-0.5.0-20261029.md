# 0.5.0 CPU benchmark evidence

Engine source: `8c5ba9c415dade4b4597509f5972ad61b80c9e76`. The measured core is the exact .NET 11 assembly from the qualified 0.5.0 package. SDK `11.0.100-rc.3.26522.148`; runtime `11.0.0-rc.3.26522.148`. The package also contains a separately qualified .NET 10 asset.

Azure `onnx-bench`, Standard_D4as_v7, AMD EPYC 9V45, four vCPUs / 16 GiB. Workers execute serially on CPU 2, monitoring on CPU 0. Native inference is Microsoft ORT 1.29.0 CPUExecutionProvider. Loading and preparation are outside timing.

| Model | Workload | Managed seconds | ORT seconds | Ratio |
|---|---|---:|---:|---:|
| Parakeet TDT 0.6B V3 | Transcribe 20 clips / 213.265 seconds of audio | 20.964610 | 18.299801 | **1.146** |
| Pyannote Community-1 | Complete diarization of a 30-second dialogue | 4.602582 | 4.069858 | **1.131** |
| multilingual-e5-small | One 30-token forward pass | 0.010892 | 0.009975 | **1.092** |
| EmbeddingGemma 2 (FP32) | One 128-token input to an owned 768-value embedding | 0.180462 | 0.180953 | **0.997** |
| DINOv3 ViT-S/16 | One 224x224 image, full weights | 0.056967 | 0.051920 | **1.097** |
| ResNet50 | One 224x224 image, feature export | 0.051771 | 0.033626 | **1.540** |
| GPT-2 | Four-token prefill, empty past state | 0.026663 | 0.019114 | **1.395** |

The [raw report](release-0.5.0-20261029.json) retains every warmup and measured clock, frequency, process interval/range, model/input/output identity, source/assembly/package hash and numerical outcome. All fourteen conformance workers pass before twenty-eight timing workers. Each case uses managed/native/native/managed process order and the median of two process results. No samples are trimmed and no calibration admission is added.

Graph timers include Reset, complete graph execution and copying every float output. They use 1,200 warmups for e5 or 600 for the other graph cases, followed by 180 calls; the process result is their mean. Audio timers include complete public APIs, frontend, decoding or clustering and owned results. Each process has one warmup pass and three measured passes; ASR sums the twenty clip means and diarization reports the 30-second dialogue mean. Gemma includes owned final-vector pooling/normalization/copying, uses its existing at-least-32-call/10-second warmup and reports the median of 33 calls.

## Process ranges

| Model | Managed process seconds | Native process seconds |
|---|---|---|
| Parakeet TDT 0.6B V3 | 20.977808, 20.951412 | 18.355288, 18.244313 |
| Pyannote Community-1 | 4.558534, 4.646630 | 4.065655, 4.074061 |
| multilingual-e5-small | 0.010964, 0.010819 | 0.009940, 0.010009 |
| EmbeddingGemma 2 (FP32) | 0.181494, 0.179429 | 0.180437, 0.181468 |
| DINOv3 ViT-S/16 | 0.055828, 0.058106 | 0.051818, 0.052022 |
| ResNet50 | 0.051980, 0.051562 | 0.033556, 0.033697 |
| GPT-2 | 0.026743, 0.026583 | 0.018817, 0.019411 |

Differences within these ranges are inconclusive. These observations apply to the named workloads on this host; they are not a universal parity claim.

## Correctness and identities

Complete graph arrays are compared directly between engines with identical shapes, finite values and scaled error at most 1e-4. Audio checks preserve exact tokens/frames and speaker decisions, timeline tolerance 1e-12, centroid scaled tolerance 1e-4, unchanged inputs, repeated-call agreement and held-output ownership. Gemma retains final-vector component, relative L2, cosine and unit-norm checks.

DINOv2 remains excluded: its fresh full-output error is 0.000148850994908, above 1e-4 and within the existing 1e-3 tripwire. Its complete managed and native output hashes are identical to the preceding qualification. Whisper has no matched timing in this refresh.

Core SHA256: `9ba5c8d64f747b50ed8352c707b920f3859fcf035ddf7c13ad8b709dd03b24ec`. Qualified package SHA256: `295a7be14f3bc87790e1619c084758079ea7e3a178a40d25dab44ab2e1fee6d1`. The raw report distinguishes the same-source frozen reference build from the measured package assembly.

The [0.4.0 report](release-0.4.0-20261027.md) remains available. It uses .NET 10; changes between those release tables include both engine and runtime changes. Matched engineering controls separately qualified the kernel improvements and .NET 10 compatibility. Both complete audio applications meet the original 1.15 latency-ratio target on this host: Parakeet 1.146 and Community-1 1.131. The process ranges above remain part of these workload-specific observations.
