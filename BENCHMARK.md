# CPU benchmarks for Lokad.Onnx

Measured 2026-11-07 on unreleased runtime source `2f3b780f`, using the existing onnx-bench Standard_D4as_v7 (AMD EPYC 9V45, 4 vCPU / 16 GiB), serial CPU 2, .NET 11 RC3 and Microsoft ORT 1.29.0 CPUExecutionProvider. .NET 10 compatibility remains qualified. The original common callers retain their e42 compiler identities.

The complete seven-model invocation finished in **39.33 minutes**, including startup, checks, staging, loading, warmup, measurement, validation and reports. All 58 workers completed. See the [full report](tests/benchmarks/runtime-full-20261107-2f3b.md) and [commands](eng/bench.md).

Times are seconds per complete workload; lower is better. Loading, preparation and validation remain outside per-request latency timers.

| Model | Workload | Lokad.Onnx seconds | Microsoft ORT seconds | Lokad / ORT |
| --- | --- | ---: | ---: | ---: |
| Parakeet TDT 0.6B V3 | Transcribe 20 clips / 213.265 seconds of audio | 17.855496 | 17.463159 | **1.022** |
| Pyannote Community-1 | Complete diarization of a 30-second dialogue | 3.343137 | 3.883390 | **0.861** |
| multilingual-e5-small | One 30-token forward pass | 0.010202 | 0.009182 | **1.111** |
| EmbeddingGemma 2 (FP32) | One 128-token input to an owned 768-value embedding | 0.172631 | 0.169591 | **1.018** |
| DINOv3 ViT-S/16 | One 224x224 image, full weights | 0.054899 | 0.049388 | **1.112** |
| ResNet50 | One 224x224 image, feature export | 0.044668 | 0.031963 | **1.397** |
| GPT-2 | Four-token prefill, empty past state | 0.025492 | 0.018068 | **1.411** |

## Measurement boundaries

The routine full suite uses six balanced independent managed/ORT pairs per audio application. Each process retains one first-use, two conditioning and three measured complete passes. Displayed audio times are arithmetic process means; ratios are geometric means of process-pair ratios, so they can differ from the ratio of displayed times. Current two-sided 95% log-Student-t intervals use df 5:

| Application | Process pairs | Paired geometric ratio | Two-sided 95% interval | <1.03 point and upper |
| --- | ---: | ---: | --- | --- |
| Parakeet TDT 0.6B V3 | 6 | 1.022466 | [1.019492, 1.025448] | Established |
| Pyannote Community-1 | 6 | 0.860821 | [0.849536, 0.872255] | Established |

The [historical longer audio proof](tests/benchmarks/release-0.5.0-integrated-20261030.md) retains its original 0.5.0 source and date, with 24 Parakeet / 12 Community pairs. Current audio confidence intervals and target decisions are shown above and retained in the new full report. No historical proof or clock is relabeled as the current product.

Parakeet includes frontend, encoder/decoder, greedy decoding and owned results. Community includes frontend, segmentation, embeddings, clustering and owned speaker timelines. All twenty Parakeet clips and four Community cases remain; the Community score covers the complete 30-second dialogue.

Graph timers include Reset, the complete forward call and copying every float output, with supplied tensor inputs and batch size one. Each process uses 1,200 warmups for e5 or 600 otherwise, then 180 calls and a process mean. Gemma includes masked mean pooling, normalization and the owned FP32 vector; tokenization/context reset are outside its timer. It uses at least 32 calls / 10 seconds of steady warmup, then 33 calls and a process median. Each non-audio row is the median of two process scores per engine.

Complete-output, finite-value, input-immutability, repeat and held-output checks retain their original limits. Graph scaled error stays <=1e-4; audio and Gemma keep their existing numerical gates. ORT uses one intra/inter-op thread, sequential execution and all graph optimizations. No profiler or dispatch override is enabled. The core has Google.Protobuf as its sole runtime dependency; Data remains nonpackable.

These are workload-specific observations on one host. DINOv2 remains excluded under its full-output gate; Whisper has no matched timing. See [model support](docs/model-support.md). Historical [0.5.0 source controls and unfavorable observations](tests/benchmarks/release-0.5.0-sustained-20261030.md) retain their original source and date. No earlier clocks are relabeled or pooled into the current run.

## Paradee runtime — unreleased, 2026-11-07

Original supplied phoneme-ID tensors to owned waveforms, source `2f3b780f`, same VM, default .NET 11 RC3 and ORT 1.29.0. All 56 workers and 24 pairs remain; 32 calls AND 40 seconds of conditioning, then 33 complete Reset/Execute/copy calls per process.

| Export | Tokens / speed | Lokad.Onnx seconds | Microsoft ORT seconds | Paired ratio | 95% interval |
| --- | --- | ---: | ---: | ---: | --- |
| Paradee FP32 | 14 / 1.0 | 0.020605 | 0.020456 | **1.0073** | [0.9970, 1.0177] |
| Paradee FP32 | 62 / 0.8 | 0.054747 | 0.069314 | **0.7895** | [0.7617, 0.8182] |
| Paradee INT8 | 14 / 1.0 | 0.021472 | 0.021922 | **0.9795** | [0.9662, 0.9929] |
| Paradee INT8 | 62 / 0.8 | 0.057467 | 0.069535 | **0.8263** | [0.8027, 0.8505] |

All four original workloads pass both point and upper two-sided 95% endpoint <1.03. Arithmetic process means and geometric paired ratios use log-Student-t 95% intervals (df 5); all observations remain. INT8 weights dequantize into FP32. Original numerical, ownership, scalar and native gates and both 18-case, zero-skip AVX-512 checks pass. See the [complete native report](tests/benchmarks/paradee-runtime-20261107-2f3b.md), [all clocks and identities](tests/benchmarks/paradee-runtime-20261107-2f3b.json) and [fresh established controls](tests/benchmarks/paradee-established-controls-20261107.md).

The earlier [prepared-conversion source comparison](tests/benchmarks/paradee-conversion-prefix-20261107.md) retains its own 10% short INT8 gain, 1.7% short FP32 cost and inconclusive long cases against its original managed control. Current native ratios are fresh observations after metadata-allocation cleanup and ordered SIMD Clip; no source ratio or old clock is pooled into them.

## Running comparisons

Use [eng/bench.py --model full](eng/bench.md) with qualified committed products and local prepared assets. Build/fixture preparation and CI waiting precede the complete timed invocation. Reuse `models/multilingual-e5-small/model.onnx`. The [full JSON](tests/benchmarks/runtime-full-20261107-2f3b.json) retains all current worker clocks, source/caller/asset identities, numerical evidence, external duration and aggregation rules. Historical [0.5.0 figures](tests/benchmarks/release-0.5.0-full-20261030.md) retain their original source and date.
