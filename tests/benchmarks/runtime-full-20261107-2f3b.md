# Full seven-model CPU benchmark — 2026-11-07, 2f3b780f

The complete existing prepared CLI invocation finished in **39.33 minutes** (2359.784 seconds), including interpreter startup, checks, staging, loading, warmup, measurements, validation and report writing. All 58 fixed workers and seven complete workloads remain; actual outer and internal clocks are both below 3,600 seconds. Runtime source is `2f3b780f`, with original e42 common caller/compiler identities retained separately.

The existing onnx-bench Standard_D4as_v7 VM has an AMD EPYC 9V45, 4 vCPU and 16 GiB. Numeric workers run serially on CPU 2, supervision on CPU 0. Managed execution uses .NET 11 RC3 and ORT uses 1.29.0 CPUExecutionProvider. .NET 10 compatibility remains separately qualified.

| Model | Workload | Lokad.Onnx seconds | Microsoft ORT seconds | Lokad / ORT |
| --- | --- | ---: | ---: | ---: |
| Parakeet TDT 0.6B V3 | Transcribe 20 clips / 213.265 seconds of audio | 17.855496 | 17.463159 | **1.022** |
| Pyannote Community-1 | Complete diarization of a 30-second dialogue | 3.343137 | 3.883390 | **0.861** |
| multilingual-e5-small | One 30-token forward pass | 0.010202 | 0.009182 | **1.111** |
| EmbeddingGemma 2 (FP32) | One 128-token input to an owned 768-value embedding | 0.172631 | 0.169591 | **1.018** |
| DINOv3 ViT-S/16 | One 224x224 image, full weights | 0.054899 | 0.049388 | **1.112** |
| ResNet50 | One 224x224 image, feature export | 0.044668 | 0.031963 | **1.397** |
| GPT-2 | Four-token prefill, empty past state | 0.025492 | 0.018068 | **1.411** |

Fourteen complete-output verification workers finish before 44 timing workers. Each audio application retains six balanced independent managed/ORT pairs; every timing process retains one first-use, two conditioning and three measured complete passes. All 20 Parakeet clips and all four Community cases remain; Community scores the complete 30-second dialogue. Audio uses process means and geometric paired ratios with two-sided log-Student-t 95% intervals (df 5). The [complete JSON](runtime-full-20261107-2f3b.json) retains every interval and strict target result, including any uncertainty. Graphs retain 180 measured calls per process; Gemma retains 33 calls and owned 768-value vectors. Non-audio rows use the median of two process scores per engine. No profiler or dispatch override is used for scores.

Every worker retains its original numerical, input, finite-output, reset/repeat and held-output checks. Parakeet includes its full frontend, encoder/decoder and decoding; Community includes its frontend, segmentation, embeddings and clustering. Graph timers include Reset, the full forward call and every float output copy; Gemma includes pooling, truncation, normalization and the owned-vector copy. Preparation and CI waiting precede this complete timed invocation; the fixed routine is neither shortened nor substituted by a cost estimate.

Archive: 96220048 bytes, SHA256 `808fd1387b5ef4090858b7fce86fbaeb4d38d0727e75e050e6dc3f815abcdfd7`; release: `2026-11-07T13:15:03.065404+00:00` after local SHA/size verification, all 161 protected assets, globally empty workers, matching-token release and cleanup of only the exported own root. The [Paradee comparison](paradee-runtime-20261107-2f3b.md) and [fresh established controls](paradee-established-controls-20261107.md) remain separate protocols; no clocks are pooled.
