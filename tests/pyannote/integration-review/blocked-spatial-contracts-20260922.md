# Contracts for a prepared spatial convolution integration

The isolated blocked-channel prototype is not yet part of normal product dispatch.
A passing component comparison would admit a separate product qualification;
complete application improvement must still be measured against the selected
implementation and Microsoft ORT. This review describes the actual integration
boundaries in root source after commit `02ad833c`.

`TensorOps.ConvPool.cs:Conv2DFloatCore` validates execution options, allocates an
owned or pooled output, materializes dense operands, and dispatches pointwise,
segmented or spatially tiled convolution. It has no prepared convolution-weight
map. The prototype prepares weights outside its hot timer, so inserting a call
that repacks weights on every convolution would be a different implementation
with a different cost. Preparation must be measured at graph/model creation and
its retained storage charged to the existing graph limit.

`GraphPacking.cs:PackMatMulWeights` currently owns all reported packed residency.
It starts its accounting at zero, reuses valid existing records, discards stale
ones and admits new clones up to `MaximumPackedWeightBytes`. A second packer
cannot independently use that same maximum: both maps must share a single total.
Refreshes must account for both kinds of retained clone. Exact boundary, zero
budget, oversized-first/smaller-later and repeated preparation behavior already
have useful examples in `PackedWeightBudgetTests.cs`. Parakeet's configured
256 MiB budget must continue to be a total bound.

`ComputationalGraph.cs:InvalidatePreparation` drops folded and packed artifacts.
`RefreshLifetimeAnalysis` folds transposes and invokes matrix packing. A
convolution map must participate in both transitions. Each record should identify
the source tensor, backing storage, shape, instruction width and clone. Initializer
replacement or a runtime weight override must not resolve an old clone. In-place
mutation follows the existing documented requirement to invalidate preparation.
Shape aliases, sliced buffers and foreign prepared records need explicit fallback
coverage. Preparation cannot silently change original initializer data.

`GraphExecution.cs` shares prepared maps and plan objects with the parent graph,
but owns request bindings, pools and released-buffer caches. New prepared weights
should follow that sharing boundary; temporary input/output layout buffers must
remain per invocation. Independent contexts, retained outputs, repeated calls,
failure recovery and cancellation must preserve the existing ownership behavior.
Every scratch rental must be returned on failure and counted through the existing
`ScratchReporter`. Replacing output allocation with pooled storage also requires
guard tests for partial writes and reuse.

The component combines convolution, bias, optional residual addition and ReLU.
The current product has `ConvRelu` and `AddRelu`, but no `ConvAddRelu` operator.
`CPUExecutionProvider.Fusion.cs` currently runs convolution/addition, then applies
ReLU in place on the owned result. Its production behavior therefore differs
from the component's public allocating ReLU call. Component timing cannot be
projected directly onto application timing.

A minimal product composition can accelerate all eligible convolutions while
retaining the existing residual and activation nodes. If integrating the combined
epilogue, either adapt the existing ConvRelu path and explicitly qualify it, or
introduce a separately reviewed residual fusion. The latter must preserve the
single-consumer and graph-output restrictions in `GraphFusion.cs:TryMatchEpilogue`,
domain/opset validation, intermediate-output exposure and graph invalidation.
Simply removing residual/activation nodes from the imported graph is insufficient.

The prototype is qualified for batch one, group one, 3x3 kernels, unit dilation,
one-element padding, strides one or two and its channel divisibility bounds.
Product admission must check every condition after the existing public validation.
Scalar policies, unsupported hardware, other shapes, noncontiguous buffers,
nonfinite operands and exhausted preparation budgets retain the selected fallback.
No model filename or particular node name should select the optimization.

Microsoft ORT's retained optimized-graph review establishes blocked-channel
convolution and fused sum/activation in the embedding graph. Its layout can stay
blocked across consecutive nodes; the prototype converts at every call. This is
an architectural difference, not an estimate of achievable application speed.
See the separate [native graph review](../nchwc-review/results-20260922.md) and
the actual AMD native-layout evidence. Neither source inspection nor an internal
component speedup establishes whole-application parity.

The next product qualification must exercise the normal project references and
public requests, every captured layer/model output, both supported instruction
widths and hardware fallback, shared models, long meetings, complete backend and
tensor suites, and an independent package consumer. Only a fresh matched AMD
production/candidate/ORT campaign can replace the application table in BENCHMARK.md.
