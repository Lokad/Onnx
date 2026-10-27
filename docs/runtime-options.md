# Runtime behavior and options

Lokad.Onnx executes graphs on the CPU. Automatic dispatch uses supported scalar,
SIMD and hardware-intrinsic kernels according to element type, shape and hardware.
The [core API guide](../src/Lokad.Onnx/README.md) describes loading, tensors and
execution diagnostics.

## Execution and ownership

`ExecutionOptions.Default` selects the Speed policy and automatic tensor dispatch.
`Scalar`, `Simd` and `Intrinsics` select explicit tensor modes.
`TensorExecutionOptions.MaxDegreeOfParallelism` defaults to one; eligible float
MatMul kernels support opt-in parallel execution.

`ExecutionOptions.Memory` additionally allows eligible dead intermediate views to
be released after their final use. The CLI `run` equivalent is `--optimize-memory`.

```csharp
var execution = graph.CreateExecution(ExecutionOptions.Memory);
bool succeeded = execution.Execute(inputs, true);
```

Caller inputs and initializers retain their contents. Capture output tensors
from `Outputs` before `Reset` or another execution repopulates the dictionary;
held outputs remain valid without cloning. Eligible `IntermediateOutputs` entries
may already be cleared during execution. The live-reference index drops final
array references so discarded unpooled intermediates can collect during a run.

`Reset` clears transient bindings and live-reference bookkeeping while preserving
input bindings, eligible released-buffer caches and completed-run diagnostics.
An application retaining a context must replace or clear its own input bindings
when their payloads are no longer needed. Private Whisper contexts clear their
inputs after each window.

One graph or execution context accepts one execution at a time. Use separate
contexts for concurrent calls. Contexts share prepared weights and synchronization,
but have independent bindings, diagnostics and released-buffer caches. Public
graph mutation invalidates affected preparation; preparation cannot run during
execution. Neither Speed nor Memory imposes a total process memory ceiling.

## Memory budgets and diagnostics

Each graph/context's released-buffer cache defaults to **128 MiB**, at most
**256 arrays**, and at most **32 per element-type/length pair**. Only arrays
already released by execution enter this cache. Select a context's payload budget:

```csharp
var execution = graph.CreateExecution(ExecutionOptions.Memory,
    maximumReleasedBufferBytes: 512L * 1024 * 1024);
```

Zero disables retention between runs; negative budgets are rejected. Array-count
caps still apply. Reset preserves eligible cached arrays; preparation invalidation
clears them. Model weights, packed weights, caller-held outputs and other live
allocations are outside this budget. Loading APIs separately accept a
`maximumPackedWeightBytes` limit for retained packed matrix weights.

`LastAllocatedBytes`, `LastPool*`, `LastScratchBytes`, `LastCopyBytes` and
`LastPeakLiveBytes` describe different costs: managed allocations, pool service,
scratch/copy traffic and live payload per binding. Aliases can count more than
once in payload metrics. These counters are not process resident memory.
Caller scratch/copy reporters receive byte events; graph-owned counters are
per-run and child graphs report separately through `LastSubgraphExecutions`.
See [measurement scopes](../tests/README.md#memory-measurement-buffer-reuse-track).

## Tensor geometry limits

Shape, index and range parameters have an inclusive absolute magnitude limit of
**2^48 (281474976710656)**; shape products also stay within 2^48. Oversized
geometry is rejected before count arithmetic, narrowing, iteration or allocation.
Dimensions, strides and backing-array lengths still have tighter Int32 limits;
a valid geometry value does not guarantee an allocation is possible. Negative
dimensions are invalid except symbolic metadata and the reshape `-1` marker.

`Range` / `Arange` require finite start, exclusive limit and nonzero delta within
the signed bound. A step facing away from the limit yields an empty range.
Output counts must fit Int32. Floating ranges retain repeated-addition semantics.
Slice's exact Int64 minimum/maximum start/end markers are normalized before
validation; other oversized values, including axes and steps, are rejected.

This limit applies to geometry, not numeric payloads: Int64 tensors may contain
Int64 extrema and floating tensors retain IEEE exceptional values. Casts, external
file offsets and memory budgets have their own contracts. Direct APIs throw;
graph-dispatch errors follow ordinary operation failure handling.

## Diagnostic controls

These comparison controls are read once per process. They default to enabled;
set exactly `0` before startup to disable one. Other values retain the default.
They are not required for normal use and do not select native inference.

| Environment variable | Mechanism |
|---|---|
| `LOKAD_ONNX_PACKED_AVX512_ROWS` | Row sharing for prepared full-column-panel MatMul |
| `LOKAD_ONNX_DEFERRED_RELEASE_CACHE` | Skip failed release probes while alias state is unchanged |
| `LOKAD_ONNX_RELEASED_BUFFER_CACHE` | Reuse already-released arrays between executions |
| `LOKAD_ONNX_FUSED_TEMP_RELEASE` | Release private MatMul/Div composite temporaries |
| `LOKAD_ONNX_SOFTMAX_EXP_PRUNE` | Omit exponentials discarded by the underflow guard |
| `LOKAD_ONNX_SOFTMAX_EXP_INLINE` | Inline the pruned exponential; requires pruning |
| `LOKAD_ONNX_SOFTMAX_NONPOSITIVE` | Specialize exponentiation after row-maximum subtraction |
| `LOKAD_ONNX_BIAS_GELU_INLINE` | Inline the existing erf arithmetic |
| `LOKAD_ONNX_VECTOR_TRANSPOSE_FACES` | Exact-copy vector kernels for supported transpose faces |
