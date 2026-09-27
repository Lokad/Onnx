# CPU benchmarks for the upcoming release

Current repository product, measured on 2026-09-27 UTC. **Lower is better.** Times
are seconds; Lokad / ORT is the latency ratio, so 1.100 means 10.0% more time.

| Model | Measured workload | Lokad.Onnx seconds | Microsoft ORT seconds | Lokad / ORT | Status |
|---|---|---:|---:|---:|---|
| Parakeet TDT 0.6B V3 | Transcribe 20 clips / 213.265 seconds of audio | 47.416186 | 39.273619 | **1.207** | Qualified |
| Pyannote Community-1 | Complete diarization of a 30-second dialogue | 9.988436 | 8.955742 | **1.115** | Qualified |
| multilingual-e5-small | One 30-token forward pass | 0.015988 | 0.015538 | **1.029** | Qualified |
| DINOv3 ViT-S/16 | One 224x224 image, full weights | 0.113185 | 0.101787 | **1.112** | Qualified |
| ResNet50 | One 224x224 image, feature export | 0.100790 | 0.072763 | **1.385** | Qualified |
| GPT-2 | Four-token prefill, empty past state | 0.033936 | 0.020035 | **1.694** | Qualified |
| DINOv2-small | 224x224 image | — | — | — | Excluded: numerical agreement gate |
| Whisper Large V3 Turbo | Speech transcription | — | — | — | Supported; current-release comparison deferred |

Every numerical timing row uses the same AMD EPYC 9V74 VM, one logical CPU
(CPU 2), .NET 10.0.8 and **Microsoft ONNX Runtime 1.29.0 CPUExecutionProvider**.
Each row is a matched comparison for that workload. Audio rows measure complete
applications; embedding, vision and GPT-2 rows measure prepared graph calls.
The workloads differ, so their absolute times should not be compared to each other.

The selected product is source `a77e6f72`, measured as Core `af19b3b4`
and Data `da72ca54`. Its [root and package qualification](tests/parakeet/decoder-packed-row-results/root-20260927.md)
verifies that the normal build (Core `0d224bcf`, Data `a3745392`) preserves all
3,284 Core and 697 Data method bodies, implementation flags, public declarations
and assembly attributes. Both full test suites pass in normal and AVX512-disabled
modes, and independent NuGet consumption passes. Ordinary mode passes 3,564 backend
and 394 tensor tests; the report records the exact hardware-dependent skips.

## What is timed

Pyannote includes audio frontend, segmentation, speaker embeddings, clustering
and owned diarization results. Its 30-second dialogue uses 21 overlapping windows.
Parakeet includes frontend, encoder/decoder inference, greedy decoding and owned
transcription results. Its total sums the twenty clip means. ORT uses matching
application policies around native inference. Each audio engine has two fresh
timed processes with one warmup and three measured passes per fixture.

Graph timings include a complete forward call returning all owned float arrays:
`Reset`, `Execute` and output materialization for Lokad.Onnx, and `session.run`
for ORT. Inputs are already tensors and batch size is one. Each process uses
1,200 fixed warmups for 30-token e5, 6,000 for 8-token e5 and 600 for the other
cases, followed by 180 measurements. Separate numerical workers run first.
Each comparison includes the previous release: six fresh processes run release,
candidate, ORT, ORT, candidate, release. The table reports the qualified candidate,
which is the current repository product.

Model loading/preparation, file IO, fixture creation, validation and reporting
are outside these timers. ORT uses one intra/inter-op thread, sequential execution
and all graph optimizations. Profilers and managed implementation overrides are
disabled. Every clock is retained and no measurements are trimmed.

## Evidence and coverage

- [Parakeet comparison, all twenty clips and complete clocks](tests/parakeet/decoder-packed-row-results/application-20260927.md):
  six measured calls per engine per clip. All 63 repeatability controls, numerical,
  complete public-result, ownership and resource checks pass.
- [Pyannote comparison and complete clocks](tests/parakeet/decoder-packed-row-results/pyannote-application-20260927.md):
  six measured calls per engine for the dialogue. All repeatability, native-result,
  ownership and resource checks pass; both ten-minute meetings and recovery pass.
- [Qualified graph comparisons and complete clocks](tests/parakeet/decoder-packed-row-results/graphs-20260927.md):
  e5 at 8, 30, 30 padded to 128, 128 and 512 tokens, DINOv3, ResNet50 and GPT-2.
  Every output is checked against ORT at the unchanged scaled-error bound
  `abs(actual-reference) / max(1, abs(reference)) <= 1e-4`, with exact shapes,
  finite values and ownership checks. A qualified row requires repeated process
  means within 10% for each engine.

The complete applications and graphs pass their repeatability gates. Isolated
Pad-call repeatability remains unqualified. The vector sigmoid passes its
numerical checks, but [isolated sigmoid timing](tests/parakeet/rational-sigmoid-results/screen-20260927.md)
has failed repeatability and fallback gates;
[double fallback latency remains unresolved](tests/parakeet/rational-sigmoid-results/fallback-diagnosis-20260927.md).
The [prepared-row operator screen](tests/parakeet/decoder-packed-row-results/screen-20260927.md)
also fails isolated repeatability and fallback gates. Its
[individual-call diagnostic](tests/parakeet/decoder-packed-row-results/unmapped-calls-20260927.md)
shows recovery over several calls when alternating prepared and unprepared weights;
cache competition is an inference, and universal fallback speed equality is unproven.
The qualified performance claims cover the listed application and graph workloads.

DINOv2 is excluded by the [known-divergence registry](tests/Lokad.Onnx.Bench/KnownDivergences.cs):
its registered output exceeds the 1e-4 agreement bound, so the runner withholds
timing. Whisper Large V3 Turbo is supported; a current-release matched timing
refresh and optimization remain deferred. Neither row has a qualified ratio.

Native agreement on these fixtures does not establish general transcription or
diarization accuracy. [Model support and qualification](docs/model-support.md)
describes public APIs, specific exports, accuracy coverage and numerical limits.
The audio APIs live in `Lokad.Onnx.Data`; the core NuGet package contains
`Lokad.Onnx` only.

## Running comparisons

The [graph protocol](tests/parakeet/decoder-packed-row-graphs-amd/README.md),
[Pyannote protocol](tests/parakeet/decoder-packed-row-pyannote-app-amd/README.md)
and [Parakeet protocol](tests/parakeet/decoder-packed-row-app-amd/README.md)
specify the assets, inputs, process order, boundaries and checks behind the table.
Use the already downloaded `models/multilingual-e5-small/model.onnx` for e5.

For ordinary local comparisons, build and run the repository harness:

```powershell
dotnet build tests/Lokad.Onnx.Bench -c Release --tl:off --nologo -v minimal
dotnet tests/Lokad.Onnx.Bench/bin/Release/net10.0/Lokad.Onnx.Bench.dll e5 dinov3 resnet50 gpt2 --mode auto --threads 1 --iters 9
```

The local harness has its own sampling protocol and packaged ORT dependency;
its output is a new measurement. `bench.ps1` additionally includes CLI startup.
Optimization priority is **Parakeet, then Pyannote**, with a matched complete
application target of Lokad / ORT <= 1.05. Whisper optimization is deferred.
