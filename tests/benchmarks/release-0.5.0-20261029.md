# 0.5.0 CPU benchmark evidence

Engine source: `8e5fb8443a3eeae0fefe598dbaffc7b19175c6f6`. The measured core is the exact .NET 11 assembly from the qualified 0.5.0 package. SDK `11.0.100-rc.3.26522.148`; runtime `11.0.0-rc.3.26522.148`. The package also contains a separately qualified .NET 10 asset.

Azure `onnx-bench`, Standard_D4as_v7, AMD EPYC 9V45, four vCPUs / 16 GiB. Workers execute serially on CPU 2, monitoring on CPU 0. Native inference is Microsoft ORT 1.29.0 CPUExecutionProvider. Loading and preparation are outside timing.

| Model | Workload | Managed seconds | ORT seconds | Ratio |
|---|---|---:|---:|---:|
| Parakeet TDT 0.6B V3 | Transcribe 20 clips / 213.265 seconds of audio | 20.846788 | 18.156939 | **1.148** |
| Pyannote Community-1 | Complete diarization of a 30-second dialogue | 4.818884 | 4.071528 | **1.184** |
| multilingual-e5-small | One 30-token forward pass | 0.010534 | 0.009939 | **1.060** |
| EmbeddingGemma 2 (FP32) | One 128-token input to an owned 768-value embedding | 0.181276 | 0.179100 | **1.012** |
| DINOv3 ViT-S/16 | One 224x224 image, full weights | 0.058466 | 0.051539 | **1.134** |
| ResNet50 | One 224x224 image, feature export | 0.050169 | 0.033084 | **1.516** |
| GPT-2 | Four-token prefill, empty past state | 0.026555 | 0.018835 | **1.410** |

The [raw report](release-0.5.0-20261029.json) retains every warmup and measured clock, frequency, process interval/range, model/input/output identity, source/assembly/package hash and numerical outcome. All fourteen conformance workers pass before twenty-eight timing workers. Each case uses managed/native/native/managed process order and the median of two process results. No samples are trimmed and no calibration admission is added.

Graph timers include Reset, complete graph execution and copying every float output. They use 1,200 warmups for e5 or 600 for the other graph cases, followed by 180 calls; the process result is their mean. Audio timers include complete public APIs, frontend, decoding or clustering and owned results. Each process has one warmup pass and three measured passes; ASR sums the twenty clip means and diarization reports the 30-second dialogue mean. Gemma includes owned final-vector pooling/normalization/copying, uses its existing at-least-32-call/10-second warmup and reports the median of 33 calls.

## Process ranges

| Model | Managed process seconds | Native process seconds |
|---|---|---|
| Parakeet TDT 0.6B V3 | 20.774930, 20.918647 | 18.118606, 18.195271 |
| Pyannote Community-1 | 4.846511, 4.791258 | 4.077660, 4.065395 |
| multilingual-e5-small | 0.010572, 0.010496 | 0.009814, 0.010063 |
| EmbeddingGemma 2 (FP32) | 0.180716, 0.181835 | 0.178552, 0.179648 |
| DINOv3 ViT-S/16 | 0.058928, 0.058004 | 0.051603, 0.051476 |
| ResNet50 | 0.049021, 0.051316 | 0.033118, 0.033051 |
| GPT-2 | 0.026936, 0.026174 | 0.019393, 0.018276 |

Differences within these ranges are inconclusive. These observations apply to the named workloads on this host; they are not a universal parity claim.

## Correctness and identities

Complete graph arrays are compared directly between engines with identical shapes, finite values and scaled error at most 1e-4. Audio checks preserve exact tokens/frames and speaker decisions, timeline tolerance 1e-12, centroid scaled tolerance 1e-4, unchanged inputs, repeated-call agreement and held-output ownership. Gemma retains final-vector component, relative L2, cosine and unit-norm checks.

DINOv2 remains excluded: its fresh full-output error is 0.000148850994908, above 1e-4 and within the existing 1e-3 tripwire. Its complete managed and native output hashes are identical to the preceding qualification. Whisper has no matched timing in this refresh.

Core SHA256: `920c848013a448d407921909cfc33271a4965ac3a9a3c30cea5c24608a648850`. Qualified package SHA256: `5a8885d10045539584963779687867ac0a12ec8c2c981e334d984a443137d380`. The raw report distinguishes the same-source frozen reference build from the measured package assembly.

The [0.4.0 report](release-0.4.0-20261027.md) remains available. It uses .NET 10; changes between those release tables include both engine and runtime changes. Matched engineering controls separately qualified the kernel improvements and .NET 10 compatibility. Parakeet meets the original 1.15 latency-ratio target at 1.148. Community-1 remains above it at 1.184; further performance work is open.
