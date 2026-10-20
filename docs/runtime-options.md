# Runtime defaults and intermediate lifetime

Lokad.Onnx enables prepared AVX-512 row sharing, exact softmax/BiasGelu scheduling,
vector transpose copies and reuse of eligible released storage by default.
Existing hardware and shape checks select compatible fallback kernels.

`ExecutionOptions.Default` keeps the existing Speed policy. To allow earlier
release of eligible dead reshape views, select `ExecutionOptions.Memory`:

```csharp
var execution = graph.CreateExecution(ExecutionOptions.Memory);
bool succeeded = execution.Execute(inputs, true, ExecutionProvider.CPU,
    ExecutionOptions.Memory);
```

The CLI equivalent for `run` is `--optimize-memory`. Intermediate bindings may
be cleared after their final use under this policy. Caller inputs, initializers,
graph outputs and previously returned tensors remain valid. Memory mode does
not impose a total process memory ceiling.

A bounded cache retains only arrays that an execution has already released.
By default, each serialized graph or explicit execution context retains at most 128 MiB
and 256 arrays, with at most 32 per element-type/length pair. Separate explicit
contexts do not share caches. Model weights, packed weights, live outputs and
other runtime allocations are outside these limits. Set the released-buffer
control below to `0` before process startup to disable retention between runs.

An explicit context can select its own released-array payload budget:

```csharp
var execution = graph.CreateExecution(ExecutionOptions.Memory,
    maximumReleasedBufferBytes: 512L * 1024 * 1024);
```

Zero disables retention between runs; negative budgets are rejected. The 256-array
and 32-per-type/length caps still apply. This budget covers already-released arrays,
not live tensors or total process memory. Reset preserves eligible cached arrays;
preparation invalidation clears them. Inputs and previously returned outputs keep
their original contents. The [package consumer qualification](../tests/whisper/memory-product-v2/results-20260921.md)
checks reuse at the byte boundary and output ownership through the public API.

## Diagnostic controls

The following controls are read once per process and default to enabled.
Set a value exactly `0` to select the preserved comparison path; `1` explicitly
enables it. Absent or other values retain the enabled default.

| Environment variable | Mechanism |
|---|---|
| `LOKAD_ONNX_PACKED_AVX512_ROWS` | Row sharing for prepared full-column-panel MatMul |
| `LOKAD_ONNX_DEFERRED_RELEASE_CACHE` | Skip repeated failed release probes while alias state is unchanged |
| `LOKAD_ONNX_RELEASED_BUFFER_CACHE` | Bounded reuse of already-released arrays between executions |
| `LOKAD_ONNX_FUSED_TEMP_RELEASE` | Release private MatMul/Div composite temporaries |
| `LOKAD_ONNX_SOFTMAX_EXP_PRUNE` | Omit exponential work discarded by the existing underflow guard |
| `LOKAD_ONNX_SOFTMAX_EXP_INLINE` | Inline the exact pruned exponential; requires pruning |
| `LOKAD_ONNX_SOFTMAX_NONPOSITIVE` | Specialize exponentiation after row-maximum subtraction |
| `LOKAD_ONNX_BIAS_GELU_INLINE` | Inline the existing erf arithmetic |
| `LOKAD_ONNX_VECTOR_TRANSPOSE_FACES` | Use exact-copy vector kernels for supported transpose faces |

These controls are for diagnosis and comparison, not a requirement for normal
use. They do not change the arithmetic precision or select native inference.

Default-off diagnostic controls were retired during 0.4.0 preparation:

| Retired controls (LOKAD_ONNX_ prefix) | Decision |
|---|---|
| PACKED_AVX512_DYNAMIC, PACKED_AVX512_PANELS, PACKED_AVX512_NARROW | Remove unqualified alternative dispatches; preserve qualified prepared and small dynamic-attention kernels. |
| SCALED_MATMUL_INPLACE | Remove the experimental distinct-storage shortcut; preserve normal exact division and temporary release. |
| FINGERPRINT_STRINGS, LAYERNORM_WIDE_OUTPUT | Remove caches/transforms without complete-model performance admission. |
| SOFTMAX_ZERO_BLOCKS, SOFTMAX_WIDE_EXP, SOFTMAX_RECIPROCAL, BIAS_GELU_INTERLEAVED | Remove alternative internal kernels; zero-block softmax missed its acceptance threshold. |
| RELEASE_RESHAPE_VIEWS, SEGMENTED_CONV | Remove environment overrides; retain the public Memory policy and TensorExecutionOptions segmented-convolution option. |
| SOFTMAX_LEGACY, TRANSPOSE_LEGACY | Remove mutable measurement routing and latches; keep direct reference tests and supported shape fallbacks. |

These environment names no longer alter runtime behavior. Original source and
qualification reports are recoverable through [the experiment archive](../archives/README.md).
Public numerical entrypoints and options remain available.

Historical benchmark reports identify their source and exact settings. Their
experimental Speed configurations must not be relabeled as measurements of the
current Default or Memory policy.

The [public-options comparison](../tests/e5/public-options-20260919.md) records
the actual Default and Memory configurations on AMD, including allocations,
GC and retained-memory observations. The disabled-path control uses the same
current core; its observed gains must not be relabeled as a new release/ORT
comparison. [Functional qualification](../tests/e5/defaults-20260919.md) records
shared-model and audio checks, including the known numerical failures.

The later [fresh native comparison](../tests/e5/public-ort-20260919.md) measures
both public choices against ORT in ninety balanced, sequential processes.
It reports all current per-case gaps and process ranges without claiming
calibrated parity or combining historical timings.
