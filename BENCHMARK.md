# CPU benchmarks for Lokad.Onnx

Measured on **2026-10-30** for 0.5.0, product source `d5f8ecad` / benchmark tooling `abc51b74`, on the existing **onnx-bench** Azure Standard_D4as_v7 VM: AMD EPYC 9V45, 4 vCPU / 16 GiB. Workers run serially on **one logical CPU (CPU 2)**. Managed execution uses **.NET 11 RC3**; the package also supports .NET 10. Native execution uses **Microsoft ORT 1.29.0 CPUExecutionProvider**.

The complete prepared seven-model benchmark finished in **41.27 minutes**, including startup, checks, staging, loading, warmup, measurement and report writing. All 58 workers completed. See the [full report](tests/benchmarks/release-0.5.0-full-20261030.md) and [commands](eng/bench.md).

Times below are seconds per complete workload; **lower is better**. Loading, preparation and validation are outside these per-request latency timers.

| Model | Measured workload | Lokad.Onnx seconds | Microsoft ORT seconds | Lokad / ORT |
|---|---|---:|---:|---:|
| Parakeet TDT 0.6B V3 | Transcribe 20 clips / 213.265 seconds of audio | 18.595831 | 18.260361 | **1.018** |
| Pyannote Community-1 | Complete diarization of a 30-second dialogue | 4.124299 | 4.042927 | **1.020** |
| multilingual-e5-small | One 30-token forward pass | 0.010449 | 0.009797 | **1.067** |
| EmbeddingGemma 2 (FP32) | One 128-token input to an owned 768-value embedding | 0.179972 | 0.176568 | **1.019** |
| DINOv3 ViT-S/16 | One 224x224 image, full weights | 0.056845 | 0.051808 | **1.097** |
| ResNet50 | One 224x224 image, feature export | 0.049160 | 0.033390 | **1.472** |
| GPT-2 | Four-token prefill, empty past state | 0.026128 | 0.019495 | **1.340** |

## Measurement boundaries

The routine full suite uses six balanced independent managed/ORT pairs per audio application. Each process retains one first-use, two conditioning and three measured complete passes. Displayed audio times are arithmetic process means; ratios are geometric means of process-pair ratios, so they can differ from the ratio of displayed times. Two-sided 95% log-Student-t intervals use df 5:

| Application | Process pairs | Paired geometric ratio | Two-sided 95% interval | <1.03 point and upper |
|---|---:|---:|---|---|
| Parakeet TDT 0.6B V3 | 6 | 1.018390 | [1.005608, 1.031333] | Not established |
| Pyannote Community-1 | 6 | 1.020031 | [1.007236, 1.032988] | Not established |

The [separate larger audio qualification](tests/benchmarks/release-0.5.0-integrated-20261030.md), on the same actual package, retains 24 Parakeet / 12 Community process pairs with six conditioning / six measured passes. Both point estimates and upper 95% bounds are below 1.03.

Parakeet includes frontend, encoder/decoder, greedy decoding and owned results. Community includes frontend, segmentation, embeddings, clustering and owned speaker timelines. All twenty Parakeet clips and four Community cases remain; the Community score covers the complete 30-second dialogue.

Graph timers include Reset, the complete forward call and copying every float output, with tensor inputs and batch size one. Each process uses 1,200 warmups for e5 or 600 otherwise, then 180 calls and a process mean. Gemma includes masked mean pooling, normalization and the owned FP32 vector; tokenization/context reset are outside its timer. It uses at least 32 calls / 10 seconds of steady warmup, then 33 calls and a process median. Each non-audio row is the median of two process scores per engine.

Complete-output, finite-value, input-immutability, repeat and held-output checks pass. Graph scaled error stays <=1e-4; audio and Gemma keep their existing numerical gates. ORT uses one intra/inter-op thread, sequential execution and all graph optimizations. No profiler or dispatch override is enabled. The core has Google.Protobuf as its sole runtime dependency; Data remains nonpackable.

These are workload-specific observations on one host. DINOv2 remains excluded under its full-output gate; Whisper has no matched timing. See [model support](docs/model-support.md). The [historical report](tests/benchmarks/release-0.5.0-sustained-20261030.md) retains earlier source controls and unfavorable observations, including the DinoV3 source-control miss and separate corroboration. No earlier clocks are relabeled or pooled into this run.

## Paradee runtime — unreleased, 2026-11-04

The original Paradee-8M-v1.2 exports now execute from fixed phoneme-ID tensors
to owned waveforms. A separate **11.45-minute** comparison on the same VM uses
clean source `3cd4f761`, .NET 11 RC3 and ORT 1.29.0. Six balanced independent
process pairs per row warm for at least 32 calls / ten seconds, then measure
33 complete calls per process.

| Original export | Tokens / speed | Lokad.Onnx seconds | Microsoft ORT seconds | Paired Lokad / ORT | 95% interval |
|---|---|---:|---:|---:|---|
| Paradee FP32 | 14 / 1.0 | 0.082032 | 0.021306 | **3.848** | [3.664, 4.040] |
| Paradee FP32 | 62 / 0.8 | 0.222321 | 0.070990 | **3.128** | [2.932, 3.337] |
| Paradee INT8 | 14 / 1.0 | 0.116148 | 0.022440 | **5.165** | [4.782, 5.578] |
| Paradee INT8 | 62 / 0.8 | 0.219429 | 0.070682 | **3.102** | [2.965, 3.246] |

Times are arithmetic process means; ratios are geometric paired means with
log-Student-t intervals (df 5). ORT performance parity remains open. INT8 weights
are dequantized into FP32 during execution. Original graphs contain random noise;
controlled companions separately qualify exact durations, pitch/phase within
`1e-4`, waveform peak error `<=1e-3` and signal-to-error ratio `>=50 dB`.
See the [full report and setup times](tests/benchmarks/paradee-runtime-20261104.md)
and [all clocks and identities](tests/benchmarks/paradee-runtime-20261104.json).
The seven-model 0.5.0 figures above retain their original source and date.

## Running comparisons

Use [eng/bench.py --model full](eng/bench.md) with qualified committed products and prepared local assets. Build/fixture preparation and CI waiting happen before the timed invocation. Reuse `models/multilingual-e5-small/model.onnx`. Individual e5/Gemma/audio source comparisons remain available through the same command; Gemma defaults to 128 tokens and 768 values.

The [full JSON](tests/benchmarks/release-0.5.0-full-20261030.json) records all worker clocks, sources, package/asset/caller hashes, numerical summaries, external duration and aggregation rules. The [larger audio JSON](tests/benchmarks/release-0.5.0-integrated-20261030.json) retains its fixed long protocol. Historical [one-warmup/three-pass results](tests/benchmarks/release-0.5.0-20261029.md) keep their original definition; `bench.ps1` includes CLI startup.
