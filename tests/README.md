# Tests

## Lanes

- Fast offline unit tests: the default `dotnet test` run. No network, no ignored large assets, no source-tree writes. Must be deterministic: repeated runs give identical pass/skip counts.
- Integration tests with committed assets: MNIST model and images under `tests/Lokad.Onnx.Backend.Tests/models` and `images`. Always run, offline.
- Local-model conformance: ignored large assets under `models/` (e5, DINOv3, ResNet50, GPT-2, DINOv2). These tests run whenever their asset files exist and skip with a reason when absent; setting `LOKAD_ONNX_RUN_LOCAL_MODEL_TESTS=1` turns a missing requested asset into a failure instead of a skip. DINOv2 is excluded only from the benchmark full-output gate (see BENCHMARK.md); its compact C# oracle (`GraphExecutionDinoV2Tests`) runs like the other model tests.
- Benchmarks: see the [isolated comparison protocol](../eng/bench.md) and the [Gemma benchmark lane](embeddinggemma2/README.md#cpu-benchmarks). Their numerical guards complement the conformance lanes; timing results do not replace correctness evidence.
- Single-op differential conformance: frozen seeded corpus under `tests/opfuzz/corpus` compared in scalar, SIMD, and intrinsics modes against frozen ONNX Runtime references. Selected explicitly via `eng/test-opfuzz.ps1`, never part of the default run. See `tests/opfuzz/README.md`.
- Fallback without hardware intrinsics: the same unit gate with `DOTNET_EnableHWIntrinsic=0`; tests that require x86 FMA skip by design with a reason, everything else must pass. The `fallback-tests` CI job runs exactly this.

## Commands

Ordinary gate (from repo root):

    dotnet restore --tl:off -v minimal eng/Tests.slnx
    dotnet build --tl:off --nologo -v minimal -c Release eng/Tests.slnx
    dotnet test --tl:off --nologo -v minimal -c Release --no-build eng/Tests.slnx

Or use `eng/test.ps1`, which runs the same steps, including the CLI needed by
integration tests. Add `-BuildTools` to also compile benchmarks and standalone
runners through the full root solution; CI does this once in its Linux ordinary
job. Add `-Coverage` to collect Cobertura reports under ignored
`artifacts/coverage`; ordinary CI does not collect them.

Native e5 conformance: `eng/test-e5.ps1` (see `tests/e5/README.md`).

Local ResNet conformance checks complete native-reference vectors and exact
managed repeats; see [reference generation and bounds](resnet/README.md).

EmbeddingGemma 2 conformance: `eng/test-embeddinggemma2.ps1` checks exact tokenization,
frozen tensor execution, owned vectors, Data API, CLI, and repeatability against
Python/native references at 128/256/512/768 dimensions. It requires explicitly
staged local assets; see [the model lane](embeddinggemma2/README.md), including its
separate source-model reference and calibrated CPU benchmark instructions.

## Local setup

- The .NET 10 SDK selected by [global.json](../global.json), PowerShell 7, and
  Python 3.13 for the native lanes. All NuGet references use exact versions
  in the project files; nothing is vendored.
- Python lane environments (test-only oracles, never loaded by .NET):
  `pip install -r tests/e5/python/requirements.txt` for the e5 lane,
  `pip install -r tests/opfuzz/python/requirements.txt` plus the generator
  file for corpus maintenance.
- Large model assets are git-ignored under `models/` and hash-pinned in
  `tests/Lokad.Onnx.Backend.Tests/ModelManifest.json` (verified by
  ModelManifestTests): `models/multilingual-e5-small/model.onnx` plus its
  `sentencepiece.bpe.model` (fetch once from the intfloat repo, see
  `tests/e5/README.md`), `models/dinov3-vits16/onnx/model.onnx` plus its
  `model.onnx_data` weights file, `models/dinov2-small-onnx/model.onnx`,
  `models/resnet50-onnx/model.onnx`, and `models/gpt2-onnx/onnx/model.onnx`.
- The e5 tokenizer resolves from the running binary directory or from
  models/multilingual-e5-small/sentencepiece.bpe.model, and downloads only
  through the explicit acquisition step when absent everywhere; everything
  downstream reuses the locked cache offline. Committed MNIST assets need
  no setup.
- Distribution check: `eng/smoke-pack.ps1` packs the core, installs it into
  a scratch console with no source references, runs inference, and verifies
  the dependency graph and file list.

## Supported ops

`CPUExecutionProvider.SupportedOps` and `OperatorSchemas.All` define the supported
surface, including internal fused operators. Consult the registry in
[OperatorSchema.cs](../src/Lokad.Onnx/OperatorSchema.cs) for the current list and
arity/opset rules. Every listed op needs at least one assertion-complete C# happy
path plus non-happy paths across the operator families. Float reduction fixtures
also cover cancellation, exceptional values and different reduction geometries
against native ORT expectations.

## Memory measurement (buffer-reuse track)

GC allocation bytes are real managed allocations: boxes, temporary arrays and
orchestration alongside payload. A warmed steady-state 30-token e5 run
allocates about 4.0 MB of calling-thread managed bytes
(`ComputationalGraph.LastAllocatedBytes`) while the pool adopts 56 fresh
arrays (2.7 MB via `LastPoolAllocatedNewBytes`), serves 260 rents from
returned buffers (15.8 MB via `LastPoolReusedBytes`), and returns 267 arrays
with zero drops; per-run collection deltas (`LastGcCollections`) are typically
[0-1, 0-1, 0]. Live tensors at run end total 4.3 MB across 97 tensors (max
live tensor 46 KB). The benchmark `allocMB` figure (about 128 MB over 33
warmed iterations) is a shared-process total across both engines plus input
conversion, not per-run attribution. Report all relevant measures: per-run
bytes plus pool service, and never compare figures across these scopes:

- Per-run bytes: `ComputationalGraph.LastAllocatedBytes` after `Execute`.
  Thread-local, so exact for single-threaded runs and a lower bound when
  worker threads allocate.
- Per-run collections: `ComputationalGraph.LastGcCollections`, a fresh
  3-length array of per-generation collection deltas. Deltas come from
  process-wide counters, so concurrent suites can advance them: they bound
  collections during the run, never attribute them.
- Pool service: `ComputationalGraph.LastPoolAllocatedNewBytes` (fresh),
  `LastPoolReusedBytes` (served from returned buffers), and
  `LastPoolPeakOutstandingBytes` (high-water mark of checked-out bytes,
  ArrayPool scratch excluded) after `Execute`.
- Scratch traffic: `ComputationalGraph.LastScratchBytes` (transient im2col
  patches and GEMM panel packing rented from ArrayPool, counted at rent
  time on full graph runs) after `Execute`.
- Copied bytes: `ComputationalGraph.LastCopyBytes` (MatMul-family view
  materialization: provider Speed-mode densify plus kernel RequireContiguous
  and MatMul2D densify paths, counted on full graph runs) after `Execute`.
- Retained peak: `ComputationalGraph.LastPeakLiveBytes` (high-water mark of
  live logical tensor bytes over run bindings, single-node runs excluded)
  after `Execute`.

Procedure (tracked pieces only): run the local e5 model warmed on one CPU,
then read the three diagnostics above off the graph after `Execute` and
confirm the live-at-end census (97 tensors, 4.3 MB, 46 KB max) against the
dumped intermediates.


Rule for future pooled kernels, learned the hard way: float MatMul kernels
accumulate into their destination and rely on zeroed outputs, so pooled MatMul
outputs rent cleared (`TensorBufferPool.RentCleared`); elementwise, Softmax,
Erf, LayerNorm, Transpose and Concat/ChunkCopy assign every element and take
plain `Rent`. Any new pooled kernel must state its contract beside the call,
with a dirty-buffer parity test when it accumulates.

## Reference versions

Timing comparators and correctness references have separate pins. Record the
actual engine version with each result:

- General/e5 benchmarks use C# `Microsoft.ML.OnnxRuntime` 1.23.2, pinned in
  [Lokad.Onnx.Bench](Lokad.Onnx.Bench/Lokad.Onnx.Bench.csproj) and
  [Lokad.Onnx.Campaign](Lokad.Onnx.Campaign/Lokad.Onnx.Campaign.csproj).
- Gemma benchmarks use C# `Microsoft.ML.OnnxRuntime` 1.29.0, pinned in
  [Lokad.Onnx.EmbeddingGemma2Bench](Lokad.Onnx.EmbeddingGemma2Bench/Lokad.Onnx.EmbeddingGemma2Bench.csproj).
- Native Python references use `onnxruntime==1.29.0` and `numpy==2.2.4`.
  Model generation also pins `onnx==1.22.0`. See the
  [e5 requirements](e5/python/requirements.txt),
  [opfuzz generator requirements](opfuzz/generate/requirements-generator.txt)
  and [Gemma requirements](embeddinggemma2/python/requirements.txt).
  Corpus `meta.json` files and
  [ModelManifest.json](Lokad.Onnx.Backend.Tests/ModelManifest.json) record
  per-reference provenance. The Gemma source-model comparison uses its
  separate CPU PyTorch environment, described in the model guide.

Reproduce a discrepancy with matching engine versions before attributing it
to a kernel. Upgrading an oracle is its own commit with regenerated references
and hashes, separate from performance implementation changes.

Replay native e5 with `eng/test-e5.ps1` (add `-RequireIntrinsics` where
supported) and frozen single-op cases with `eng/test-opfuzz.ps1`; both verify
asset hashes first. The ResNet and Gemma guides above describe their full-output
checks and reference generation.

## Model-oracle bisection procedure

Checks vary by model. Some tests use compact mean/spot references; MNIST
checks logits to four decimals and argmax behavior. ResNet retains mean/spot
checks and compares every output component against the committed
[full-vector fixture](Lokad.Onnx.Backend.Tests/fixtures/resnet-ort.json), with
exact managed repeats and input/output ownership checks. The e5 and Gemma
lanes compare full tensors and pooled vectors; Gemma also checks integrated
tokenization, API and CLI results. Parakeet compares complete outputs and
carried state; see [its conformance guide](parakeet/README.md).
Use each lane's recorded inputs, provenance and bounds when an oracle drifts:

- Reproduce the failing case's exact input and execution options. Constant-filled images, ramp images, frozen text cases and audio/state cases have different references.
- Bisect with node-prefix cuts: run the graph truncated at successive nodes (a cut stack, e.g. the DINOv3 RoPE cut stack) in both Lokad.Onnx and native ONNX Runtime through a dtype-generic probe, comparing intermediate tensors to find the first diverging node. Durable equivalents: the tests/opfuzz corpus plus the e5 lane.
- Root-cause at op level and keep a direct op-level regression test alongside the refreshed oracle.
