# Tests

## Lanes

- Fast offline unit tests: the default `dotnet test` run. No network, no ignored large assets, no source-tree writes. Must be deterministic: repeated runs give identical pass/skip counts.
- Integration tests with committed assets: MNIST model and images under `tests/Lokad.Onnx.Backend.Tests/models` and `images`. Always run, offline.
- Local-model conformance: ignored large assets under `models/` (e5, DINOv2). Selected explicitly, never part of the default run. Missing requested assets fail only when `LOKAD_ONNX_RUN_LOCAL_MODEL_TESTS=1`; otherwise the test reports a skip with a reason.
- Benchmarks: never correctness tests. See `lonnx benchmark`.

## Commands

Ordinary gate (from repo root):

    dotnet restore --tl:off -v minimal Lokad.Onnx.slnx
    dotnet build --tl:off --nologo -v minimal -c Release Lokad.Onnx.slnx
    dotnet test --tl:off --nologo -v minimal -c Release --no-build Lokad.Onnx.slnx

Or use `eng/test.ps1`, which runs the same steps and collects Cobertura coverage under ignored `artifacts/coverage`.

Native e5 conformance: `eng/test-e5.ps1` (see `tests/e5/README.md` once Commit 07 lands).

## Supported ops

`CPUExecutionProvider.SupportedOps` currently holds 28 values: Reshape, Add, Div, Sub, Mul, Pow, Conv, Relu, MaxPool, MatMul, Sqrt, Erf, Transpose, Constant, Cast, Concat, Shape, Gather, Slice, Equal, Where, Expand, Resize, Unsqueeze, ReduceSum, ReduceMean, ReduceMax, Softmax. Every listed op needs at least one assertion-complete C# happy path plus non-happy paths (Commits 09-10).

## Python migration ledger

Legacy suite under `tests/python` exercises .NET through pythonnet and is retired file by file in Commits 08-10. Static inventory below (`pytest --collect-only` capture pending a working Python env). Each row names the C# successor or the commit that owns it.

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
