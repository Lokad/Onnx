# Tests

## Lanes

- Fast offline unit tests: the default `dotnet test` run. No network, no ignored large assets, no source-tree writes. Must be deterministic: repeated runs give identical pass/skip counts.
- Integration tests with committed assets: MNIST model and images under `tests/Lokad.Onnx.Backend.Tests/models` and `images`. Always run, offline.
- Local-model conformance: ignored large assets under `models/` (e5, DINOv3, ResNet50, GPT-2, DINOv2). These tests run whenever their asset files exist and skip with a reason when absent; setting `LOKAD_ONNX_RUN_LOCAL_MODEL_TESTS=1` turns a missing requested asset into a failure instead of a skip. DINOv2 is excluded only from the benchmark full-output gate (see BENCHMARK.md); its compact C# oracle (`GraphExecutionDinoV2Tests`) runs like the other model tests.
- Benchmarks: never correctness tests. See `Bench micro` and `bench.ps1`.
- Single-op differential conformance: frozen seeded corpus under `tests/opfuzz/corpus` compared in scalar, SIMD, and intrinsics modes against frozen ONNX Runtime references. Selected explicitly via `eng/test-opfuzz.ps1`, never part of the default run. See `tests/opfuzz/README.md`.
- Fallback without hardware intrinsics: the same unit gate with `DOTNET_EnableHWIntrinsic=0`; tests that require x86 FMA skip by design with a reason, everything else must pass. The `fallback-tests` CI job runs exactly this.

## Commands

Ordinary gate (from repo root):

    dotnet restore --tl:off -v minimal Lokad.Onnx.slnx
    dotnet build --tl:off --nologo -v minimal -c Release Lokad.Onnx.slnx
    dotnet test --tl:off --nologo -v minimal -c Release --no-build Lokad.Onnx.slnx

Or use `eng/test.ps1`, which runs the same steps and collects Cobertura coverage under ignored `artifacts/coverage`.

Native e5 conformance: `eng/test-e5.ps1` (see `tests/e5/README.md`).

## Local setup

- .NET 10 SDK (verified with 10.0.300-preview.0.26177.108), PowerShell 7, and
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

`CPUExecutionProvider.SupportedOps` is the supported surface (46 entries at last count; the code list is authoritative). It starts: Reshape, Add, Div, Sub, Mul, Pow, Conv, Relu, MaxPool, MatMul, Sqrt, Erf, Transpose, Constant, Cast, Concat, Shape, Gather, Slice, Equal, Where, Expand, Resize, Unsqueeze, ReduceSum, ReduceMean, ReduceMax, Softmax. Every listed op needs at least one assertion-complete C# happy path plus non-happy paths across the operator families.

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

Two different ONNX Runtime versions answer two different questions; never
mix their evidence:

- Timing comparator only: `Microsoft.ML.OnnxRuntime` 1.23.2 (C#, exact pin
  in `tests/Lokad.Onnx.Bench/Lokad.Onnx.Bench.csproj`, printed in every
  benchmark row). It measures the native side of speed comparisons and is
  never an oracle.
- Frozen oracles: Python `onnxruntime==1.29.0` with `onnx==1.22.0` and
  `numpy==2.2.4` (exact pins in `tests/e5/python/requirements.txt` and
  `tests/opfuzz/generate/requirements-generator.txt`). These generate the
  e5 expectations at lane time and the frozen `tests/opfuzz/corpus` references
  (each `meta.json` records its generating environment), plus the
  `ModelManifest.json` oracles (`ortOracle 1.29` with per-model generators).

Policy: a difference between the 1.23.2 and 1.29.0 references is a version
difference, never kernel evidence. Upgrading any oracle is its own commit
with regenerated references and hashes, never bundled with a performance
implementation commit.

Replay: `eng/test-e5.ps1` (add `-RequireIntrinsics` where supported) and
`eng/test-opfuzz.ps1` from the repo root; both verify asset hashes first.

## Model-oracle bisection procedure

All model-level tests share one compact-oracle pattern: exact shape, no NaN/Infinity, mean within 1e-6 plus strided spot values within 1e-4/1e-3 of a frozen native ONNX Runtime reference (MNIST instead asserts exact logits to 4 decimals plus argmax behavior). Reference values are generated with native ORT 1.29 (see tests/Lokad.Onnx.Backend.Tests/ModelManifest.json for per-model provenance), never checked in as tensors. When an oracle drifts:

- Reproduce with a fixed input (0.5-filled tensor of the model input shape; e5 uses the frozen tests/e5/cases.json).
- Bisect with node-prefix cuts: run the graph truncated at successive nodes (a cut stack, e.g. the DINOv3 RoPE cut stack) in both Lokad.Onnx and native ONNX Runtime through a dtype-generic probe, comparing intermediate tensors to find the first diverging node. Durable equivalents: the tests/opfuzz corpus plus the e5 lane.
- Root-cause at op level and keep a direct op-level regression test alongside the refreshed oracle.