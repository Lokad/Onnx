# Tests

## Lanes

- Fast offline unit tests: the default `dotnet test` run. No network, no ignored large assets, no source-tree writes. Must be deterministic: repeated runs give identical pass/skip counts.
- Integration tests with committed assets: MNIST model and images under `tests/Lokad.Onnx.Backend.Tests/models` and `images`. Always run, offline.
- Local-model conformance: ignored large assets under `models/` (e5, DINOv2). Selected explicitly, never part of the default run. Missing requested assets fail only when `LOKAD_ONNX_RUN_LOCAL_MODEL_TESTS=1`; otherwise the test reports a skip with a reason.
- Benchmarks: never correctness tests. See `Bench micro` and `bench.ps1`.
- Single-op differential conformance: frozen seeded corpus under `tests/opfuzz/corpus` compared in scalar, SIMD, and intrinsics modes against frozen ONNX Runtime references. Selected explicitly via `eng/test-opfuzz.ps1`, never part of the default run. See `tests/opfuzz/README.md`.

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
- The e5 tokenizer is copied on first use into the running binary directory
  through the explicit acquisition step and reused from the locked cache
  offline afterwards. Committed MNIST assets need no setup.
- Distribution check: `eng/smoke-pack.ps1` packs the core, installs it into
  a scratch console with no source references, runs inference, and verifies
  the dependency graph and file list.

## Supported ops

`CPUExecutionProvider.SupportedOps` is the supported surface (44 entries at last count; the code list is authoritative). It starts: Reshape, Add, Div, Sub, Mul, Pow, Conv, Relu, MaxPool, MatMul, Sqrt, Erf, Transpose, Constant, Cast, Concat, Shape, Gather, Slice, Equal, Where, Expand, Resize, Unsqueeze, ReduceSum, ReduceMean, ReduceMax, Softmax. Every listed op needs at least one assertion-complete C# happy path plus non-happy paths (Commits 09-10).

## Python migration ledger (historical)

Commit numbers below refer to the retired pre-review roadmap; every row has landed and the legacy suite is retired. Kept as provenance for the C# successor of each old Python test.

Legacy suite under `tests/python` exercised .NET through pythonnet and was retired file by file. Static inventory below (`pytest --collect-only` capture pending a working Python env). Each row names the C# successor or the commit that owns it.

### test_tensors.py -> Commit 08

| Python test | Disposition |
|---|---|
| test_make_tensors | TensorCreationTests (verify) |
| test_convert_tensors | TensorCreationTests, missing dtype cases -> new tests |
| test_arange | TensorCreationTests (verify) |
| test_slice | TensorSliceTests (verify) |
| test_broadcast_dim | TensorBroadcastTests (verify) |
| test_add | TensorOpsElementwiseTests (verify) |
| test_matmul | TensorOpsMatMulTests + TensorOpsMatMulIndependentTests (done, Commit 04) |

### test_tokenizers.py -> Commit 08

| Python test | Disposition |
|---|---|
| test_me5s | New deterministic C# tokenizer tests with literal IDs (Hello world -> 0,35378,8999,2); batch parity covered by Commit 05 equivalence check |

### test_backend.py -> Commits 09-10

| Python test | Disposition |
|---|---|
| test_load_graph | ModelTests.CanParseFile (exists) |
| test_model_run | GraphExecutionMnistTests.CanInferWithMnist (exists) |
| test_model_file_run | ModelTests.CanParseFile + GraphExecutionMnistTests (exists) |
| test_node_run | Named missing C# node-run test -> Commit 09 |
| test_model_node_run | Named missing C# test -> Commit 09 |
| test_mnist_model_run | GraphExecutionMnistTests (exists, incl. dictionary inputs) |
| test_add, test_sub, test_mul, test_div | CpuExecutionProviderOpTests.Add_Sub_Mul_Div_Broadcast_Int32 (exists; strengthen numerics/contracts in Commit 09) |
| test_pow | CpuExecutionProviderOpTests.Pow_Sqrt_Erf_Relu_Float (exists; strengthen in Commit 09) |
| test_transpose | CpuExecutionProviderOpTests.MatMul_Transpose_Softmax (exists) + TensorReshapeTransposeTests |
| test_reshape | CpuExecutionProviderReshapeTests.CanReshape (exists) |
| test_concat, test_unsqueeze | CpuExecutionProviderOpTests.Conv_MaxPool_Reduce_Concat_Gather_Slice_Unsqueeze_Cast_Constant (exists) |
| test_shape | CpuExecutionProviderShapeTests.CanGetShape (exists) |
| test_gather, test_slice | Same big op test (exists; add negative axes/optional inputs in Commit 09) |
| test_constant, test_cast | Same big op test (exists; add invalid-type contracts in Commit 09) |
| test_erf | CpuExecutionProviderOpTests.Pow_Sqrt_Erf_Relu_Float (exists; edge cases in Commit 10) |
| test_reduce_sum, test_reduce_mean | Big op test Reduce coverage (exists; keepdim/axes in Commit 10) |
| test_softmax | CpuExecutionProviderOpTests.MatMul_Transpose_Softmax (exists; numerical edges in Commit 10) |
| test_matmul | Big op test + MatMul suites incl. independent tests (done, Commit 04) |

### test_me5s_model.py -> retired by Commits 06-07

test_model_file_prepare, test_tokenizer, test_model_run download the ONNX file and PyTorch model at import time. Retired with rationale: network-dependent and compares against a different artifact. Superseded by the offline native e5 gate on the exact local assets.

### _tf_test_node.py (159 tests) -> Commits 09-10

Grouped by family; cases for ops outside SupportedOps retire with rationale (op not supported by CPUExecutionProvider). Supported-op cases port per family: shape/manipulation and elementwise in Commit 09; neural (Conv, Relu, MaxPool), MatMul-adjacent, and reductions in Commit 10. Families targeting Abs, Acosh, ArgMax, ArgMin, Asinh, Atanh, BatchNormalization, Ceil, Celu, Compress, ConstantFill, ConstantOfShape, ConvInteger, ConvTranspose, Cosh, Cumsum, DepthToSpace, DequantizeLinear and similar retire as unsupported.

## Memory measurement (buffer-reuse track)

GC allocation bytes are real managed allocations: boxes, temporary arrays and
orchestration alongside payload. A per-op replay attributes ~177 MB to a 30-token
e5 run whose true output bytes are ~30 MB (max live tensor 46 KB); the gap is
per-op framework overhead shared by every execution path, so GC deltas cannot
attribute pooling work on their own. Report all relevant measures: GC bytes plus
true-byte accounting:

- True output bytes per op: replay nodes in file order with exact per-node
  inputs through the tracked `ComputationalGraph.ExecuteNode` API (caller dict
  plus initializers only, exact count match, initializers excluded)
  and sum `Length x element-size` of each produced output. Deterministic and
  immune to framework overhead.
- Pool service: `ComputationalGraph.LastPoolAllocatedNewBytes` (fresh) and
  `LastPoolReusedBytes` (served from returned buffers) after `Execute`.
  Current 30-token e5 floor: ~29.8 MB true outputs, ~25.3 MB served from pool,
  ~2.9 MB fresh pool arrays, ~1.6 MB from never-pooled ops.

Procedure (tracked pieces only): drive the per-node replay through
`ComputationalGraph.ExecuteNode` from a small console referencing `src`
(Release) with the local e5 model plus tokenizer paths. It prints timed-run
GC totals with pool counters followed by the per-op true-byte census.
(Historical precedent: a `C:/Temp` throwaway spike produced the floor numbers
below; it was never committed.)

Rule for future pooled kernels, learned the hard way: float MatMul kernels
accumulate into their destination and rely on zeroed outputs, so pooled MatMul
outputs rent cleared (`TensorBufferPool.RentCleared`); elementwise, Softmax,
Erf, LayerNorm, Transpose and Concat/ChunkCopy assign every element and take
plain `Rent`. Any new pooled kernel must state its contract beside the call,
with a dirty-buffer parity test when it accumulates.

## Model-oracle bisection procedure

All model-level tests share one compact-oracle pattern: exact shape, no NaN/Infinity, mean within 1e-6 plus strided spot values within 1e-4/1e-3 of a frozen native ONNX Runtime reference (MNIST instead asserts exact logits to 4 decimals plus argmax behavior). Reference values are generated with native ORT 1.29 (see tests/Lokad.Onnx.Backend.Tests/ModelManifest.json for per-model provenance), never checked in as tensors. When an oracle drifts:

- Reproduce with a fixed input (0.5-filled tensor of the model input shape; e5 uses the frozen tests/e5/cases.json).
- Bisect with node-prefix cuts: run the graph truncated at successive nodes (a cut stack, e.g. the DINOv3 RoPE cut stack) in both Lokad.Onnx and native ONNX Runtime through a dtype-generic probe, comparing intermediate tensors to find the first diverging node. Durable equivalents: the tests/opfuzz corpus plus the e5 lane; historical precedent is an uncommitted cut-run harness (onnxruntime 1.29).
- Root-cause at op level and add a direct op-level regression test alongside the refreshed oracle. Precedent: ITensor.Unsqueeze normalized negative axes against the input rank instead of the output rank, so a downstream Concat read wrong cells (end-to-end mean abs err 0.085); fixed with the output-rank normalization plus CanUnsqueezeNegativeAxis.
