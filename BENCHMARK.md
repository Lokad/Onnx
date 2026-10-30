# CPU benchmarks for Lokad.Onnx

Measured on **2026-10-30** for upcoming 0.5.0, source `7c0f1f2c`, on the
existing `onnx-bench` Azure Standard_D4as_v7 VM: AMD EPYC 9V45, 4 vCPU / 16 GiB.
Workers run serially on **one logical CPU (CPU 2)**. Managed execution uses
**.NET 11 RC3** (`11.0.0-rc.3.26522.148`); the package also supports .NET 10.
Native execution uses Microsoft ONNX Runtime **1.29.0 CPUExecutionProvider**.
Loading, preparation and validation are outside the timers.

Times are seconds; **lower is better**. Lokad / ORT is the latency ratio.
Audio ratios use independent paired processes; other rows use medians of two
process scores per engine. The different aggregation rules are explained below.

| Model | Measured workload | Lokad.Onnx seconds | Microsoft ORT seconds | Lokad / ORT |
|---|---|---:|---:|---:|
| Parakeet TDT 0.6B V3 | Transcribe 20 clips / 213.265 seconds of audio | 17.870971 | 17.557958 | **1.018** |
| Pyannote Community-1 | Complete diarization of a 30-second dialogue | 3.882969 | 3.896787 | **0.996** |
| multilingual-e5-small | One 30-token forward pass | 0.010183 | 0.009336 | **1.091** |
| EmbeddingGemma 2 (FP32) | One 128-token input to an owned 768-value embedding | 0.174886 | 0.171609 | **1.019** |
| DINOv3 ViT-S/16 | One 224x224 image, full weights | 0.056851 | 0.049462 | **1.149** |
| ResNet50 | One 224x224 image, feature export | 0.047870 | 0.032078 | **1.492** |
| GPT-2 | Four-token prefill, empty past state | 0.025593 | 0.017221 | **1.486** |

Both audio applications satisfy the **<1.03 point and upper 95% interval** target:
Parakeet 1.017825 [1.014959, 1.020700]; Community-1 0.996427 [0.988092, 1.004831].
These are workload-specific observations on one host. DINOv2 remains excluded
under its full-output numerical gate; Whisper has no matched timing.
See [model support](docs/model-support.md) for export and numerical limits.

## Measurement boundaries

Audio uses 24 independent managed/ORT process pairs for Parakeet and 12 for
Community, balanced in engine order. Each process runs one first-use pass, six
conditioning passes and six measured complete passes. The displayed times are
arithmetic process means. Ratios are geometric means of paired process ratios,
with two-sided 95% Student-t intervals on their logarithms; consequently the
displayed ratio need not equal the ratio of displayed times. All observations
remain, including measured-half trends and separate original-source bridges.

Parakeet includes frontend, encoder/decoder, greedy decoding and owned results.
Community includes frontend, segmentation, embeddings, clustering and owned
speaker timelines. Loading/preparation and first use are reported separately
with allocation, GC pause and peak memory in the [complete report](tests/benchmarks/release-0.5.0-sustained-20261030.md).

Graph timers include Reset, a complete forward call and copying every float
output; inputs are already tensors, batch size one. Each process uses 1,200
warmups for e5 or 600 otherwise, then 180 measured calls and a process mean.
Gemma includes masked mean pooling, normalization and the owned FP32 vector;
tokenization and context reset are outside timing. Its existing steady warmup
uses at least 32 calls / 10 seconds, followed by 33 calls and a process median.
These five controls use current/candidate/ORT/ORT/candidate/current order;
the table reports the candidate and ORT medians of two process scores.

The DinoV3 direct source control initially missed its 1.02 regression limit.
A separately declared six-process interleaved corroboration passed both its
point and upper interval limit. Both designs and the original failure remain
in the report; their GC/cache conditions differ. Other four source controls pass.

Complete outputs, unchanged inputs, repeatability and held-output ownership
are checked. Graph scaled error is <=1e-4; audio retains exact token/frame/speaker
decisions and its timeline/centroid gates. Gemma retains component, relative-L2,
cosine and unit-norm gates. ORT uses one intra/inter-op thread, sequential
execution and all graph optimizations. No profiler or dispatch override is enabled.
These measurements use the package identified in the report, before the importer
was reintegrated. The current core package depends on Google.Protobuf;
Data remains nonpackable. The changed package is undergoing qualification.

## Running comparisons

Use [eng/bench.py](eng/bench.md) for routine e5/Gemma comparisons with committed
prepared cores; Gemma defaults to 128 tokens and 768 dimensions. Reuse the local
`models/multilingual-e5-small/model.onnx`. The [complete report](tests/benchmarks/release-0.5.0-sustained-20261030.md) records
all raw calls, identities, numerical gates and sampling rules. Historical
[one-warmup/three-pass results](tests/benchmarks/release-0.5.0-20261029.md)
retain their original protocol; `bench.ps1` includes CLI startup.
