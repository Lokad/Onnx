# 0.5.0 full seven-model CPU benchmark

The complete prepared CLI invocation finished in **41.27 minutes** (2475.962 seconds). All 58 fixed workers and 17,032 calls were retained and independently verified. Internal time through report writing was 2475.901 seconds. Both clocks are below 3,600 seconds.

Product source `d5f8ecad10da83f898eb4c21f45c1e0eb6f942f0`; caller/tooling source `abc51b74a5aee06e3ea478596c388faee739fcd7`; package SHA256 `8b5066108b4306aeef4dd16947fcc2f4e7a3e95099d53a892ed4b4081b0a358b`.

Measured on 2026-10-30 on the existing onnx-bench Standard_D4as_v7 VM, AMD EPYC 9V45, 4 vCPU / 16 GiB. Numerical workers run serially on CPU 2; the supervisor uses CPU 0. Managed execution uses .NET 11 RC3; ORT uses 1.29.0 CPUExecutionProvider. .NET 10 compatibility remains separately qualified.

| Model | Measured workload | Lokad.Onnx seconds | Microsoft ORT seconds | Lokad / ORT |
|---|---|---:|---:|---:|
| Parakeet TDT 0.6B V3 | Transcribe 20 clips / 213.265 seconds of audio | 18.595831 | 18.260361 | **1.018** |
| Pyannote Community-1 | Complete diarization of a 30-second dialogue | 4.124299 | 4.042927 | **1.020** |
| multilingual-e5-small | One 30-token forward pass | 0.010449 | 0.009797 | **1.067** |
| EmbeddingGemma 2 (FP32) | One 128-token input to an owned 768-value embedding | 0.179972 | 0.176568 | **1.019** |
| DINOv3 ViT-S/16 | One 224x224 image, full weights | 0.056845 | 0.051808 | **1.097** |
| ResNet50 | One 224x224 image, feature export | 0.049160 | 0.033390 | **1.472** |
| GPT-2 | Four-token prefill, empty past state | 0.026128 | 0.019495 | **1.340** |

Seconds are per complete workload. Audio uses arithmetic process means and geometric paired latency ratios. Graphs use process means of 180 calls; Gemma uses process medians of 33 calls. The five non-audio rows are medians of two process scores per engine. All timings in this table use the same actual integrated package.

## Fixed short protocol

Fourteen managed/native conformance workers finish before 44 timing workers. Each audio application has six independent managed/native process pairs, balanced in engine order. Each timed audio process runs one first-use pass, two conditioning passes and three measured full passes. All twenty Parakeet clips and all four Community cases remain; the Community score is the 30-second dialogue. No source-control arm or repeated native arm is included.

Each other model has two fresh processes per engine. Graph callers retain 1,200 e5 warmups or 600 other warmups, followed by 180 measured complete forward/reset/copied-output calls. Gemma retains its steady warmup of at least 32 calls / 10 seconds and 33 measured owned embeddings. No timing is trimmed, corrected or extended after inspection.

Audio uncertainty uses a two-sided 95% Student-t interval on log process-pair ratios (df 5, t 2.570581835636305). These six-pair intervals have less precision than the separate larger audio qualification.

| Application | Process pairs | Paired geometric ratio | Two-sided 95% interval | <1.03 point and upper |
|---|---:|---:|---|---|
| Parakeet TDT 0.6B V3 | 6 | 1.018390 | [1.005608, 1.031333] | Not established |
| Pyannote Community-1 | 6 | 1.020031 | [1.007236, 1.032988] | Not established |

The [larger fixed audio cohort](release-0.5.0-integrated-20261030.md) uses the same package with 24 Parakeet / 12 Community pairs and six conditioning / six measured passes. It remains a separate experiment; its clocks and intervals are not pooled into this shorter protocol. Both larger-cohort point estimates and upper bounds pass 1.03.

## Duration and correctness

The external wall clock runs from CLI process creation through exit, including interpreter startup, fixture/asset checks, staging, loading, warmup, measurements, output validation and report writes. Asset downloads, caller/product compilation, fixture preparation, CI waiting and proof compression are prerequisites or postprocessing outside this prepared invocation. A 3,600-second overall deadline bounds owned workers; incomplete runs retain failure evidence and cannot count as success.

Per-request timers include the complete audio public API, or graph Reset/Execute/copied outputs, or the owned Gemma vector including pooling/truncation/normalization. Their loading, conditioning and validation costs remain outside those per-request latency values, while counting toward the suite duration.

Complete numerical outputs, finite values, input immutability, repeats and held-output ownership pass. Graph scaled error is at most 1e-4. Audio preserves exact token/frame/status/speaker decisions, 1e-12 timeline tolerance and 1e-4 centroid tolerance. Gemma retains component, relative-L2, cosine and unit-norm gates. Every raw clock is retained in the [JSON](release-0.5.0-full-20261030.json); complete output arrays/payloads remain in the independently checked hash-bound proof. Guest variation is recorded without the earlier 1% refusal.

The final audit verifies no own numerical workers, both installed .NET runtimes, all 59 protected assets and every additional full-config asset. DINOv2 remains excluded under its numerical gate; Whisper has no matched timing. Earlier failed synthetic/source controls and the separate DinoV3 corroboration remain unchanged in the [historical report](release-0.5.0-sustained-20261030.md).

An earlier attempt stopped during conformance because Gemma default-tiering metadata was absent. It ran no timing-phase pairs and is not a successful duration result. The failed invocation took 81.133 seconds; its proof is retained. The correction declares DOTNET_TieredCompilation=1 only for Gemma, matching the existing routine. Graph/audio settings, workloads, sampling and numerical gates remain unchanged. The retry declaration and failed proof identity are in the JSON.

Proof: 2396 files / 327207498 bytes / SHA256 `06b0193f036752b141596b09b10773205e0f760abffea7c9e6bc1ec8de259d5e`.

Use the [full-suite commands](../../eng/bench.md) with prepared local assets. This qualification creates no NuGet publication or release tag.
