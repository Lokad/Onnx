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
Each serialized graph or explicit execution context retains at most 128 MiB
and 256 arrays, with at most 32 per element-type/length pair. Separate explicit
contexts do not share caches. Model weights, packed weights, live outputs and
other runtime allocations are outside these limits. Set the released-buffer
control below to `0` before process startup to disable retention between runs.

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

Other experiments, including narrow/panel/dynamic packed MatMul, interleaved
BiasGelu, wider or reciprocal softmax and segmented convolution, remain off
unless explicitly enabled with `1`. `LOKAD_ONNX_RELEASE_RESHAPE_VIEWS=1` remains
an experimental override of the lifetime policy; prefer the public
`ExecutionOptions.Memory` option for application code.

Historical benchmark reports identify their source and exact settings. Their
experimental Speed configurations must not be relabeled as measurements of the
current Default or Memory policy.
