# 0.5.0 CPU benchmark evidence

Engine source: `a4417547709ec4e36b66ec792b01b47a11a8922a`. The measured core is the exact .NET 11 assembly from the qualified 0.5.0 package. SDK `11.0.100-rc.3.26522.148`; runtime `11.0.0-rc.3.26522.148`. The package also contains a separately qualified .NET 10 asset.

Azure `onnx-bench`, Standard_D4as_v7, AMD EPYC 9V45, four vCPUs / 16 GiB. Workers execute serially on CPU 2, monitoring on CPU 0. Native inference is Microsoft ORT 1.29.0 CPUExecutionProvider. Loading and preparation are outside timing.

| Model | Workload | Managed seconds | ORT seconds | Ratio |
|---|---|---:|---:|---:|
| Parakeet TDT 0.6B V3 | Transcribe 20 clips / 213.265 seconds of audio | 20.791532 | 18.207472 | **1.142** |
| Pyannote Community-1 | Complete diarization of a 30-second dialogue | 4.706645 | 3.942498 | **1.194** |
| multilingual-e5-small | One 30-token forward pass | 0.010219 | 0.009318 | **1.097** |
| EmbeddingGemma 2 (FP32) | One 128-token input to an owned 768-value embedding | 0.174824 | 0.170033 | **1.028** |
| DINOv3 ViT-S/16 | One 224x224 image, full weights | 0.056578 | 0.049859 | **1.135** |
| ResNet50 | One 224x224 image, feature export | 0.047789 | 0.032085 | **1.489** |
| GPT-2 | Four-token prefill, empty past state | 0.025771 | 0.017946 | **1.436** |

The [raw report](release-0.5.0-20261029.json) retains every warmup and measured clock, frequency, process interval/range, model/input/output identity, source/assembly/package hash and numerical outcome. All fourteen conformance workers pass before twenty-eight timing workers. Each case uses managed/native/native/managed process order and the median of two process results. No samples are trimmed and no calibration admission is added.

Graph timers include Reset, complete graph execution and copying every float output. They use 1,200 warmups for e5 or 600 for the other graph cases, followed by 180 calls; the process result is their mean. Audio timers include complete public APIs, frontend, decoding or clustering and owned results. Each process has one warmup pass and three measured passes; ASR sums the twenty clip means and diarization reports the 30-second dialogue mean. Gemma includes owned final-vector pooling/normalization/copying, uses its existing at-least-32-call/10-second warmup and reports the median of 33 calls.

## Process ranges

| Model | Managed process seconds | Native process seconds |
|---|---|---|
| Parakeet TDT 0.6B V3 | 20.963888, 20.619176 | 18.353005, 18.061938 |
| Pyannote Community-1 | 4.712038, 4.701252 | 3.953611, 3.931384 |
| multilingual-e5-small | 0.010216, 0.010222 | 0.009380, 0.009256 |
| EmbeddingGemma 2 (FP32) | 0.174338, 0.175311 | 0.170112, 0.169954 |
| DINOv3 ViT-S/16 | 0.055574, 0.057581 | 0.049712, 0.050005 |
| ResNet50 | 0.047654, 0.047924 | 0.032167, 0.032003 |
| GPT-2 | 0.025759, 0.025783 | 0.017790, 0.018103 |

Differences within these ranges are inconclusive. These observations apply to the named workloads on this host; they are not a universal parity claim.

## Correctness and identities

Complete graph arrays are compared directly between engines with identical shapes, finite values and scaled error at most 1e-4. Audio checks preserve exact tokens/frames and speaker decisions, timeline tolerance 1e-12, centroid scaled tolerance 1e-4, unchanged inputs, repeated-call agreement and held-output ownership. Gemma retains final-vector component, relative L2, cosine and unit-norm checks.

DINOv2 remains excluded: its fresh full-output error is 0.000148850994908, above 1e-4 and within the existing 1e-3 tripwire. Its complete managed and native output hashes are identical to the preceding qualification. Whisper has no matched timing in this refresh.

Core SHA256: `9049b2242efcf081eba999a6ac331e90354e4d492e11f68faa8366c14ce10666`. Qualified package SHA256: `fdbd3b6d5aa2191934b1551ebe9d19d213db21a1cdbfd3783ec815e7a25d5b26`. The raw report distinguishes the same-source frozen reference build from the measured package assembly.

The [0.4.0 report](release-0.4.0-20261027.md) remains available. It uses .NET 10; changes between those release tables include both engine and runtime changes. Matched engineering controls separately qualified the kernel improvements and .NET 10 compatibility. Parakeet meets the original 1.15 target in this package report; Community-1 remains above it at 1.194. Further Community-1 performance work remains open.
