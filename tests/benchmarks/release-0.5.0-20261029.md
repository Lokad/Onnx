# 0.5.0 CPU benchmark evidence

Engine source: `0ade33f23cfb8bd3ef095ba7fd6243cce68d5e8c`. The measured core is the exact .NET 11 assembly from the qualified 0.5.0 package. SDK `11.0.100-rc.3.26522.148`; runtime `11.0.0-rc.3.26522.148`. The package also contains a separately qualified .NET 10 asset.

Azure `onnx-bench`, Standard_D4as_v7, AMD EPYC 9V45, four vCPUs / 16 GiB. Workers execute serially on CPU 2, monitoring on CPU 0. Native inference is Microsoft ORT 1.29.0 CPUExecutionProvider. Loading and preparation are outside timing.

| Model | Workload | Managed seconds | ORT seconds | Ratio |
|---|---|---:|---:|---:|
| Parakeet TDT 0.6B V3 | Transcribe 20 clips / 213.265 seconds of audio | 20.998276 | 18.432081 | **1.139** |
| Pyannote Community-1 | Complete diarization of a 30-second dialogue | 4.801282 | 4.079063 | **1.177** |
| multilingual-e5-small | One 30-token forward pass | 0.010579 | 0.009781 | **1.082** |
| EmbeddingGemma 2 (FP32) | One 128-token input to an owned 768-value embedding | 0.182760 | 0.179390 | **1.019** |
| DINOv3 ViT-S/16 | One 224x224 image, full weights | 0.056910 | 0.052106 | **1.092** |
| ResNet50 | One 224x224 image, feature export | 0.050259 | 0.033537 | **1.499** |
| GPT-2 | Four-token prefill, empty past state | 0.026270 | 0.019544 | **1.344** |

The [raw report](release-0.5.0-20261029.json) retains every warmup and measured clock, frequency, process interval/range, model/input/output identity, source/assembly/package hash and numerical outcome. All fourteen conformance workers pass before twenty-eight timing workers. Each case uses managed/native/native/managed process order and the median of two process results. No samples are trimmed and no calibration admission is added.

Graph timers include Reset, complete graph execution and copying every float output. They use 1,200 warmups for e5 or 600 for the other graph cases, followed by 180 calls; the process result is their mean. Audio timers include complete public APIs, frontend, decoding or clustering and owned results. Each process has one warmup pass and three measured passes; ASR sums the twenty clip means and diarization reports the 30-second dialogue mean. Gemma includes owned final-vector pooling/normalization/copying, uses its existing at-least-32-call/10-second warmup and reports the median of 33 calls.

## Process ranges

| Model | Managed process seconds | Native process seconds |
|---|---|---|
| Parakeet TDT 0.6B V3 | 21.043326, 20.953225 | 18.368808, 18.495355 |
| Pyannote Community-1 | 4.652645, 4.949919 | 4.087148, 4.070978 |
| multilingual-e5-small | 0.010532, 0.010625 | 0.009726, 0.009836 |
| EmbeddingGemma 2 (FP32) | 0.184724, 0.180797 | 0.179948, 0.178833 |
| DINOv3 ViT-S/16 | 0.055957, 0.057864 | 0.052186, 0.052026 |
| ResNet50 | 0.050744, 0.049773 | 0.033596, 0.033478 |
| GPT-2 | 0.026376, 0.026164 | 0.019409, 0.019680 |

Differences within these ranges are inconclusive. These observations apply to the named workloads on this host; they are not a universal parity claim.

## Correctness and identities

Complete graph arrays are compared directly between engines with identical shapes, finite values and scaled error at most 1e-4. Audio checks preserve exact tokens/frames and speaker decisions, timeline tolerance 1e-12, centroid scaled tolerance 1e-4, unchanged inputs, repeated-call agreement and held-output ownership. Gemma retains final-vector component, relative L2, cosine and unit-norm checks.

DINOv2 remains excluded: its fresh full-output error is 0.000148850994908, above 1e-4 and within the existing 1e-3 tripwire. Its complete managed and native output hashes are identical to the preceding qualification. Whisper has no matched timing in this refresh.

Core SHA256: `2c1fd036882d69fb04a5948cb8aa81cbd2da5f08223307428d199b5e8fdaa8b9`. Qualified package SHA256: `8ac00b03750c056c236224fd70858a36bdaa9bb6d82d170925a901bf0086183a`. The raw report distinguishes the same-source frozen reference build from the measured package assembly.

The [0.4.0 report](release-0.4.0-20261027.md) remains available. It uses .NET 10; changes between those release tables include both engine and runtime changes. Matched engineering controls separately qualified the kernel improvements and .NET 10 compatibility. Parakeet meets the original 1.15 latency-ratio target at 1.139. Community-1 remains above it at 1.177; further performance work is open.
