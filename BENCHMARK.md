# CPU benchmarks

Measured 7 November 2026 on the onnx-bench Standard_D4as_v7 VM (AMD EPYC 9V45, 4 vCPU / 16 GiB), using one logical CPU, .NET 11 RC3 and Microsoft ORT 1.29.0 CPUExecutionProvider. The library also supports .NET 10.

Times are seconds per complete workload; lower is better. Loading, preparation and validation are outside the per-request latency timers.

| Model | Workload | Lokad.Onnx seconds | Microsoft ORT seconds | Lokad / ORT | 95% ratio interval |
| --- | --- | ---: | ---: | ---: | --- |
| Parakeet TDT 0.6B V3 | 20 clips / 213.265 s of audio | 17.855496 | 17.463159 | 1.0225 | [1.0195, 1.0254] |
| Pyannote Community-1 | Complete diarization of a 30 s dialogue | 3.343137 | 3.883390 | 0.8608 | [0.8495, 0.8723] |
| multilingual-e5-small | One 30-token forward pass | 0.010202 | 0.009182 | 1.1111 | — |
| EmbeddingGemma 2 FP32 | 128 tokens to an owned 768-value embedding | 0.172631 | 0.169591 | 1.0179 | — |
| DINOv3 ViT-S/16 | One 224x224 image, full weights | 0.054899 | 0.049388 | 1.1116 | — |
| ResNet50 | One 224x224 image, feature export | 0.044668 | 0.031963 | 1.3975 | — |
| GPT-2 | Four-token prefill, empty past state | 0.025492 | 0.018068 | 1.4109 | — |
| Paradee FP32 | 14 phoneme IDs, voice 0, speed 1.0 | 0.020605 | 0.020456 | 1.0073 | [0.9970, 1.0177] |
| Paradee FP32 | 62 phoneme IDs, voice 0, speed 0.8 | 0.054747 | 0.069314 | 0.7895 | [0.7617, 0.8182] |
| Paradee INT8 | 14 phoneme IDs, voice 0, speed 1.0 | 0.021472 | 0.021922 | 0.9795 | [0.9662, 0.9929] |
| Paradee INT8 | 62 phoneme IDs, voice 0, speed 0.8 | 0.057467 | 0.069535 | 0.8263 | [0.8027, 0.8505] |

## Measurement boundaries

Parakeet includes the frontend, encoder/decoder, greedy decoding and owned results. Community includes the frontend, segmentation, embeddings, clustering and owned speaker timelines. Paradee measures the original FP32/INT8 exports from supplied tensors to owned 24 kHz waveforms; INT8 weights dequantize into FP32. Graph timers include Reset, the full forward call and every float output copy. Gemma includes pooling, normalization and the owned embedding; tokenization and context reset are outside its timer.

Audio and Paradee use six balanced independent managed/ORT pairs and geometric ratios of process means. Their two-sided 95% log-Student-t intervals use df 5. Both the point and upper interval endpoint are below 1.03 for Parakeet, Community and all four Paradee workloads. Other rows use the median of two process scores per engine and have no estimated interval in this table.

Each audio process runs one first-use, two conditioning and three measured complete passes, retaining all 20 Parakeet clips and four Community cases. Graph processes use 1,200 warmups for e5 or 600 otherwise, then 180 measured calls. Gemma uses at least 32 calls / 10 seconds of warmup and 33 measured calls. Each Paradee timing process conditions for 32 calls **and** 40 seconds, then retains 33 Reset/Execute/copy calls.

Complete-output, finite-value, input-immutability, repeat and held-output checks retain their numerical limits. Graph scaled error is at most 1e-4; Paradee waveform peak error is at most 1e-3 with signal-to-error at least 50 dB. ORT uses one intra/inter-op thread, sequential execution and all graph optimizations. Numeric workers run serially on CPU 2 with no profiler or dispatch override.

## Reproducing measurements

The complete seven-model routine takes **39.33 minutes** (58 workers), including startup, checks, staging, loading, warmup, measurement, validation and reporting. Paradee uses its separate four-case comparison. See the [comparison commands](eng/bench.md), [complete seven-model results](tests/benchmarks/runtime-full-20261107-2f3b.md), [Paradee results](tests/benchmarks/paradee-runtime-20261107-2f3b.md) and [matched source controls](tests/benchmarks/paradee-established-controls-20261107.md) for exact source/product identities, raw observations and aggregation rules.
