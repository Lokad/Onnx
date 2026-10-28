# 0.5.0 CPU benchmark evidence

Engine source: `1e8982065f5316c917410378c29cc051b2e03f76`. The measured core is the exact .NET 11 assembly from the qualified 0.5.0 package. SDK `11.0.100-rc.3.26522.148`; runtime `11.0.0-rc.3.26522.148`. The package also contains a separately qualified .NET 10 asset.

Azure `onnx-bench`, Standard_D4as_v7, AMD EPYC 9V45, four vCPUs / 16 GiB. Workers execute serially on CPU 2, monitoring on CPU 0. Native inference is Microsoft ORT 1.29.0 CPUExecutionProvider. Loading and preparation are outside timing.

| Model | Workload | Managed seconds | ORT seconds | Ratio |
|---|---|---:|---:|---:|
| Parakeet TDT 0.6B V3 | Transcribe 20 clips / 213.265 seconds of audio | 20.657613 | 17.778912 | **1.162** |
| Pyannote Community-1 | Complete diarization of a 30-second dialogue | 5.337817 | 3.914974 | **1.363** |
| multilingual-e5-small | One 30-token forward pass | 0.010777 | 0.009266 | **1.163** |
| EmbeddingGemma 2 (FP32) | One 128-token input to an owned 768-value embedding | 0.175807 | 0.170663 | **1.030** |
| DINOv3 ViT-S/16 | One 224x224 image, full weights | 0.057334 | 0.050377 | **1.138** |
| ResNet50 | One 224x224 image, feature export | 0.048641 | 0.032173 | **1.512** |
| GPT-2 | Four-token prefill, empty past state | 0.025921 | 0.017700 | **1.464** |

The [raw report](release-0.5.0-20261028.json) retains every warmup and measured clock, frequency, process interval/range, model/input/output identity, source/assembly/package hash and numerical outcome. All fourteen conformance workers pass before twenty-eight timing workers. Each case uses managed/native/native/managed process order and the median of two process results. No samples are trimmed and no calibration admission is added.

Graph timers include Reset, complete graph execution and copying every float output. They use 1,200 warmups for e5 or 600 for the other graph cases, followed by 180 calls; the process result is their mean. Audio timers include complete public APIs, frontend, decoding or clustering and owned results. Each process has one warmup pass and three measured passes; ASR sums the twenty clip means and diarization reports the 30-second dialogue mean. Gemma includes owned final-vector pooling/normalization/copying, uses its existing at-least-32-call/10-second warmup and reports the median of 33 calls.

## Process ranges

| Model | Managed process seconds | Native process seconds |
|---|---|---|
| Parakeet TDT 0.6B V3 | 20.732399, 20.582828 | 17.775030, 17.782795 |
| Pyannote Community-1 | 5.270086, 5.405547 | 3.903604, 3.926343 |
| multilingual-e5-small | 0.010700, 0.010855 | 0.009222, 0.009310 |
| EmbeddingGemma 2 (FP32) | 0.174949, 0.176665 | 0.171188, 0.170138 |
| DINOv3 ViT-S/16 | 0.057338, 0.057330 | 0.050547, 0.050208 |
| ResNet50 | 0.048532, 0.048750 | 0.032131, 0.032214 |
| GPT-2 | 0.025839, 0.026002 | 0.017888, 0.017511 |

Differences within these ranges are inconclusive. These observations apply to the named workloads on this host; they are not a universal parity claim.

## Correctness and identities

Complete graph arrays are compared directly between engines with identical shapes, finite values and scaled error at most 1e-4. Audio checks preserve exact tokens/frames and speaker decisions, timeline tolerance 1e-12, centroid scaled tolerance 1e-4, unchanged inputs, repeated-call agreement and held-output ownership. Gemma retains final-vector component, relative L2, cosine and unit-norm checks.

DINOv2 remains excluded: its fresh full-output error is 0.000148850994908, above 1e-4 and within the existing 1e-3 tripwire. Its complete managed and native output hashes are identical to the preceding qualification. Whisper has no matched timing in this refresh.

Core SHA256: `6dab42f2175f7f18bf4a7836255b6988ccaed1e61667658c7d24a7f6bc32ace7`. Qualified package SHA256: `b49055af25339688a37a7a3d7e2ea6f2b181b0cf8e154b385b8d2ce47588e827`. The raw report distinguishes the same-source frozen reference build from the measured package assembly.

The [0.4.0 report](release-0.4.0-20261027.md) remains available. It uses .NET 10; changes between those release tables include both engine and runtime changes. Matched engineering controls separately qualified the kernel improvements and .NET 10 compatibility. The initial 1.15 audio ratio remains a goal; the observed ratios above state the current result.
