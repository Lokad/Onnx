# CPU benchmarks for Lokad.Onnx

These are the latest published matched CPU measurements for the listed workloads.
The September rows measure the 0.3.0 implementation (`04584fc2`) on 2026-09-28 UTC;
EmbeddingGemma 2 measures source `98b9234c` on 2026-10-17 UTC. They retain those
measured revisions during 0.4.0 preparation: subsequent correctness and lifetime
fixes have not been given a new matched performance comparison.
**Lower is better.** Times are seconds; Lokad / ORT is the latency ratio, so
1.100 means 10.0% more time.

| Model | Measured workload | Lokad.Onnx seconds | Microsoft ORT seconds | Lokad / ORT | Status |
|---|---|---:|---:|---:|---|
| Parakeet TDT 0.6B V3 | Transcribe 20 clips / 213.265 seconds of audio | 43.459339 | 39.202810 | **1.109** | Qualified |
| Pyannote Community-1 | Complete diarization of a 30-second dialogue | 9.920118 | 8.949371 | **1.108** | Qualified |
| multilingual-e5-small | One 30-token forward pass | 0.015878 | 0.015335 | **1.035** | Qualified |
| EmbeddingGemma 2 (FP32) | One 128-token input to an owned 768-value embedding | 0.165331 | 0.168010 | **0.984** | Repeated comparison |
| DINOv3 ViT-S/16 | One 224x224 image, full weights | 0.109552 | 0.101699 | **1.077** | Qualified |
| ResNet50 | One 224x224 image, feature export | 0.092810 | 0.072859 | **1.274** | Qualified |
| GPT-2 | Four-token prefill, empty past state | 0.033982 | 0.020112 | **1.690** | Qualified |
| DINOv2-small | 224x224 image | — | — | — | Excluded: numerical agreement gate |
| Whisper Large V3 Turbo | Speech transcription | — | — | — | Supported; current-release comparison deferred |

The September timing rows use the same AMD EPYC 9V74 VM. EmbeddingGemma 2 uses
the AMD EPYC 9V45 `onnx-bench` VM. All rows use one logical CPU (CPU 2),
.NET 10.0.8 and **Microsoft ONNX Runtime 1.29.0 CPUExecutionProvider**.
Each row is a matched comparison for that workload. Audio rows measure complete
applications; embedding, vision and GPT-2 rows measure prepared graph calls.
The workloads differ, so their absolute times should not be compared to each other.

Source/binary identities and package checks for the September measurements are
recorded with their [qualification evidence](tests/parakeet/transpose-axis-results/root-20260928.md).
The current 0.4.0 NuGet package contains only the dependency-free core. Audio and
embedding application helpers are nonpackable Data source APIs; their measured
end-to-end workloads include that companion code.

## What is timed

Pyannote includes audio frontend, segmentation, speaker embeddings, clustering
and owned diarization results. Its 30-second dialogue uses 21 overlapping windows.
Parakeet includes frontend, encoder/decoder inference, greedy decoding and owned
transcription results. Its total sums the twenty clip means. ORT uses matching
application policies around native inference. Each audio engine has two fresh
timed processes with one warmup and three measured passes per fixture.

September graph timings include a complete forward call returning all owned float arrays:
`Reset`, `Execute` and output materialization for Lokad.Onnx, and `session.run`
for ORT. Inputs are already tensors and batch size is one. Each process uses
1,200 fixed warmups for 30-token e5, 6,000 for 8-token e5 and 600 for the other
cases, followed by 180 measurements. Separate numerical workers run first.
Each comparison includes the previous release: six fresh processes run release,
candidate, ORT, ORT, candidate, release. The table reports the qualified candidate,
which is the 0.3.0 implementation covered by those September rows.

EmbeddingGemma 2 times an owned FP32 final embedding, including masked mean
pooling, truncation, normalization and copying. Tokenization, loading and context
reset are outside this timer. Its row is the median of two fresh-process medians
per engine, each retaining 33 calls after at least 30 seconds of compute warmup.
Lokad repeats span 0.165033–0.165630 seconds; ORT repeats span
0.167971–0.168049 seconds. A separate 128-token comparison uses the same candidate binary.
See its [comparison and raw samples](tests/embeddinggemma2/benchmarks-20261017-fixedwidth.md)
for the separate protocol, measured source, numerical checks and other token lengths.

Model loading/preparation, file IO, fixture creation, validation and reporting
are outside these timers. ORT uses one intra/inter-op thread, sequential execution
and all graph optimizations. Profilers and managed implementation overrides are
disabled. Every clock is retained and no measurements are trimmed.

## Evidence and coverage

- [EmbeddingGemma 2 FP32 comparison and raw samples](tests/embeddinggemma2/benchmarks-20261017-fixedwidth.md):
  two fresh processes per engine/revision, balanced baseline/candidate/candidate/baseline
  ordering, exact inputs and checked final vectors. Small repeat variations are reported;
  this measurement does not claim acceptance by the historical 3% A/A scorer.
- [Parakeet comparison, all twenty clips and complete clocks](tests/parakeet/transpose-axis-results/application-20260928.md):
  six measured calls per engine per clip. All 63 repeatability controls, numerical,
  complete public-result, ownership and resource checks pass.
- [Pyannote comparison and complete clocks](tests/parakeet/transpose-axis-results/pyannote-application-20260928.md):
  six measured calls per engine for the dialogue. All repeatability, native-result,
  ownership and resource checks pass; both ten-minute meetings and recovery pass.
- [Qualified graph comparisons and complete clocks](tests/parakeet/transpose-axis-results/graphs-20260928.md):
  e5 at 8, 30, 30 padded to 128, 128 and 512 tokens, DINOv3, ResNet50 and GPT-2.
  Every output is checked against ORT at the unchanged scaled-error bound
  `abs(actual-reference) / max(1, abs(reference)) <= 1e-4`, with exact shapes,
  finite values and ownership checks. A qualified row requires repeated process
  means within 10% for each engine.

The performance claims cover the listed complete application and graph workloads.
They do not establish uniform speedups for isolated operators or fallback paths.

DINOv2 is excluded by the [known-divergence registry](tests/Lokad.Onnx.Bench/KnownDivergences.cs):
its registered output exceeds the 1e-4 agreement bound, so the runner withholds
timing. Whisper Large V3 Turbo is supported but has no completed matched timing
refresh. Neither row has a qualified ratio.

Native agreement on these fixtures does not establish general transcription or
diarization accuracy. [Model support and qualification](docs/model-support.md)
describes public APIs, specific exports, accuracy coverage and numerical limits.
The audio APIs live in `Lokad.Onnx.Data`; the core NuGet package contains
`Lokad.Onnx` only.

## Running comparisons

The [graph protocol](tests/parakeet/transpose-axis-graphs-amd/README.md),
[Pyannote protocol](tests/parakeet/transpose-axis-pyannote-app-amd/README.md)
and [Parakeet protocol](tests/parakeet/transpose-axis-app-amd/README.md)
specify the assets, inputs, process order, boundaries and checks behind the table.
Use the already downloaded `models/multilingual-e5-small/model.onnx` for e5.

For new e5/Gemma comparisons, use [eng/bench.py](eng/bench.md) with committed
prepared cores and pinned local inputs. Its default Gemma case is 128 tokens
with 768 output dimensions; tokenization is outside the compute timer.

The multi-model local harness is also available:

```powershell
dotnet build tests/Lokad.Onnx.Bench -c Release --tl:off --nologo -v minimal
dotnet tests/Lokad.Onnx.Bench/bin/Release/net10.0/Lokad.Onnx.Bench.dll e5 dinov3 resnet50 gpt2 --mode auto --threads 1 --iters 9
```

The local harness has its own sampling protocol and packaged ORT dependency;
its output is a new measurement. `bench.ps1` additionally includes CLI startup.
