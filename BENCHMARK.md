# CPU benchmarks for the upcoming release

Current repository product, measured on 2026-09-23 UTC. **Lower is better.** Times
are seconds; Lokad / ORT is the latency ratio, so 1.154 means 15.4% more time.

| Model | Measured workload | Lokad.Onnx seconds | Microsoft ORT seconds | Lokad / ORT | Status |
|---|---|---:|---:|---:|---|
| Parakeet TDT 0.6B V3 | Transcribe 20 clips / 213.265 seconds of audio | 73.816787 | 39.605085 | **1.864** | Qualified |
| Pyannote Community-1 | Complete diarization of a 30-second dialogue | 10.409171 | 9.022426 | **1.154** | Qualified |
| multilingual-e5-small | One 30-token forward pass | 0.016795 | 0.016292 | **1.031** | Qualified |
| DINOv3 ViT-S/16 | One 224x224 image, full weights | 0.118048 | 0.102414 | **1.153** | Qualified |
| ResNet50 | One 224x224 image, feature export | 0.105688 | 0.073183 | **1.444** | Qualified |
| GPT-2 | Four-token prefill, empty past state | 0.036097 | 0.020871 | **1.730** | Qualified |
| DINOv2-small | 224x224 image | — | — | — | Excluded: numerical agreement gate |
| Whisper Large V3 Turbo | Speech transcription | — | — | — | Supported; current-release comparison deferred |

Every numerical timing row uses the same AMD EPYC 9V74 VM, one logical CPU
(CPU2), .NET 10.0.8 and **Microsoft ONNX Runtime 1.29.0 CPUExecutionProvider**.
Each row is a matched comparison for that workload. Audio rows measure complete
applications; the embedding, vision and GPT-2 rows measure prepared graph calls.
The workloads differ, so their absolute times should not be compared to each other.

The selected product is source `81f75c38`, measured as Core `672e5f30` and
Data `065b7a7f`. The [normal root and package qualification](tests/parakeet/wide-entry-first-use-results/root-20260923.md)
verifies all compiled methods, implementation flags and public interfaces, full
backend/tensor suites in both instruction modes and independent NuGet consumption.
Ordinary mode passes 3,449 backend and 343 tensor tests; the hardware-dependent
skip census in each mode is recorded in that report.

## What is timed

Pyannote includes audio frontend, segmentation, speaker embeddings, clustering
and owned diarization results. Its 30-second dialogue uses 21 overlapping windows.
Parakeet includes frontend, encoder/decoder inference, greedy decoding and owned
transcription results. Its total sums the twenty clip means. The ORT comparisons
use matching application policies around native inference. Each audio engine has
two fresh timed processes with one warmup and three measured passes per fixture.

Graph timings include a complete forward call returning all owned float arrays:
`Reset`, `Execute` and output materialization for Lokad.Onnx, and `session.run`
for ORT. Inputs are already tensors, batch size is one, and each timed process
uses 600 fixed warmups and 180 measurements. Separate numerical workers run first.
Each comparison also includes the previous selected product: six fresh
processes run previous, candidate, ORT, ORT, candidate, previous. The table
reports the qualified candidate, which is the current repository product.

Model loading/preparation, file IO, fixture creation, validation and reporting
are outside these timers. ORT uses one intra/inter-op thread, sequential execution
and all graph optimizations. Profilers and managed implementation overrides are
disabled. All clocks are retained; no measurements are trimmed or retried.

## Evidence and coverage

- [Pyannote comparison and complete clocks](tests/parakeet/wide-entry-first-use-results/pyannote-20260923.md):
  six measured calls per engine for the dialogue. All repeatability, native-result,
  ownership and resource checks pass; both ten-minute meetings and recovery pass.
- [Parakeet comparison, all twenty clips and complete clocks](tests/parakeet/wide-entry-first-use-results/application-20260923.md):
  six measured calls per engine per clip. All 63 repeatability controls, native/public
  result checks and resource checks pass.
- [Current graph comparisons and complete clocks](tests/parakeet/wide-entry-first-use-results/graphs-20260923.md):
  includes e5 at 8, 30, 30 padded to 128, 128 and 512 tokens, DINOv3, ResNet50 and GPT-2.
  Every output is checked against fresh ORT at the unchanged scaled-error bound
  `abs(actual-reference) / max(1, abs(reference)) <= 1e-4`, with exact shapes,
  finite values and ownership checks. A qualified row requires repeated process
  means within 10% for each engine.

DINOv2 remains excluded by the [known-divergence registry](tests/Lokad.Onnx.Bench/KnownDivergences.cs):
its registered output exceeds the 1e-4 agreement bound, so the runner withholds
timing. Whisper Large V3 Turbo is supported, but a current-release matched timing
refresh and further optimization are deferred. Neither row has a qualified
current-release ratio.

Native agreement on these fixtures does not establish general transcription or
diarization accuracy. [Model support and qualification](docs/model-support.md)
describes the public APIs, specific exports, accuracy coverage and remaining
numerical limitations. The audio APIs live in `Lokad.Onnx.Data`; the core NuGet
package contains `Lokad.Onnx` only.

## Running comparisons

The [frozen graph protocol](tests/parakeet/wide-entry-first-use-graphs-amd/README.md),
[Pyannote protocol](tests/parakeet/wide-entry-first-use-pyannote-app-amd/README.md) and
[Parakeet protocol](tests/parakeet/wide-entry-first-use-app-amd-v2/README.md) specify the
assets, inputs, process order, boundaries and checks behind these tables.
Use the already downloaded `models/multilingual-e5-small/model.onnx` for e5.

For ordinary local model comparisons, build and run the repository harness:

```powershell
dotnet build tests/Lokad.Onnx.Bench -c Release --tl:off --nologo -v minimal
dotnet tests/Lokad.Onnx.Bench/bin/Release/net10.0/Lokad.Onnx.Bench.dll e5 dinov3 resnet50 gpt2 --mode auto --threads 1 --iters 9
```

The local harness has its own sampling protocol and packaged ORT dependency;
its output is a new measurement. `bench.ps1` additionally includes CLI startup.
Current optimization priority is **Parakeet, then Pyannote**, with a matched
application latency target of Lokad / ORT <= 1.05. Whisper optimization is deferred.
