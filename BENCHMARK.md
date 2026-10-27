# CPU benchmarks for Lokad.Onnx

Measured on **2026-10-27** for the 0.4.0 candidate, source `83f4e1b4`, on the
existing `onnx-bench` Azure Standard_D4as_v7 VM: AMD EPYC 9V45, 4 vCPU / 16 GiB.
Every worker uses **one logical CPU (CPU 2)**. Managed workers run .NET 10.0.8;
native inference uses Microsoft ONNX Runtime **1.29.0 CPUExecutionProvider**.
Loading, preparation and validation are outside the timers.

Times are seconds; **lower is better**. Lokad / ORT is the latency ratio, so 1.100
means 10% more time. These are repeated, checked workload measurements on one
host; small differences within the process ranges are inconclusive.

| Model | Measured workload | Lokad.Onnx seconds | Microsoft ORT seconds | Lokad / ORT |
|---|---|---:|---:|---:|
| Parakeet TDT 0.6B V3 | Transcribe 20 clips / 213.265 seconds of audio | 33.359048 | 17.891236 | **1.865** |
| Pyannote Community-1 | Complete diarization of a 30-second dialogue | 7.003582 | 3.967896 | **1.765** |
| multilingual-e5-small | One 30-token forward pass | 0.010854 | 0.009490 | **1.144** |
| EmbeddingGemma 2 (FP32) | One 128-token input to an owned 768-value embedding | 0.176912 | 0.178833 | **0.989** |
| DINOv3 ViT-S/16 | One 224x224 image, full weights | 0.060046 | 0.050507 | **1.189** |
| ResNet50 | One 224x224 image, feature export | 0.067117 | 0.032353 | **2.075** |
| GPT-2 | Four-token prefill, empty past state | 0.026500 | 0.018154 | **1.460** |

DINOv2 remains excluded from timing by its full-output numerical gate.
Whisper Large V3 Turbo is supported but has no matched timing in this refresh.
See [model support](docs/model-support.md) for export, numerical and accuracy limits.

## Measurement boundaries

Each workload runs in four fresh processes, ordered managed / ORT / ORT / managed.
The table takes the median of the two process results for each engine. All calls
are retained, with no trimming or strict historical A/A calibration. The
[report and process ranges](tests/benchmarks/release-0.4.0-20261027.md) link every warmup and measured clock,
model/input/binary identities and numerical checks.

Parakeet times the complete public transcription API, including frontend,
encoder/decoder, greedy decoding and owned results. Its figure sums the twenty
clip means. Pyannote times the complete diarization API, including frontend,
segmentation, embeddings, clustering and owned speaker timelines; the dialogue
uses 21 overlapping windows. Each audio process runs one warmup and three
measured passes per fixture. Public results, input immutability, repeatability
and held-output ownership are checked outside timing.

The e5, vision and GPT-2 figures time Reset, a complete forward call and copying
every float output into owned arrays. Inputs are already tensors, batch size is
one. Each graph process uses 1,200 warmups for e5 or 600 for the other cases,
then 180 measured calls; its result is the arithmetic mean. Complete outputs are
compared directly against ORT at scaled error <=1e-4 with exact shapes and finite
values.

Gemma times the owned FP32 final vector, including masked mean pooling,
normalization and copying. Tokenization, loading and context reset are outside
this timer. Each process uses the existing at-least-32-call / 10-second steady
warmup, retains 33 calls and reports their median. Final-vector component,
relative L2, cosine and unit-norm checks run before and after timing.

ORT uses one intra/inter-op thread, sequential execution and all graph
optimizations. Managed execution uses automatic dispatch and one thread. Audio
uses Memory-mode contexts within each request; Gemma reuses one across calls.
No profiler or implementation override is
enabled. The core NuGet package has no runtime dependencies; complete audio
applications use the nonpackable Data project.

## Running comparisons

Use [eng/bench.py](eng/bench.md) for routine e5/Gemma comparisons with committed
prepared cores; Gemma defaults to 128 tokens and 768 dimensions. Reuse the local
models/multilingual-e5-small/model.onnx. The [refresh report](tests/benchmarks/release-0.4.0-20261027.md) describes
the graph/audio consumers and exact sampling behind this table. The local
multi-model harness has its own sampling protocol; bench.ps1 includes CLI startup.
