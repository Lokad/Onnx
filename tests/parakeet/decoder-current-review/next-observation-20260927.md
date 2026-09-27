# Resolve the decoder projection question with one bounded observation

The decision is whether to let the final one-row projection consume its existing
prepared B layout. Keep the current vector width, four accumulators, reduction
order, scalar tail and ownership fixed if an implementation follows. The observed
shape is A `[1,1,1,640]`, B `[640,8198]`; the constant is `onnx::MatMul_230` and
the original node is `/joint/joint_net/joint_net.2/MatMul`.

The retained current wall profile establishes complete node time and the exact
constant edge, but does not record its runtime mapping, per-node allocation or
executed kernel. The old OwnedWeightCensus and final-row counters inspect encoder
feed-forward weights, not this decoder mapping. The September 21 packing census
does inspect the decoder, but binds older DLLs. None independently establishes
the current decoder's prepared mapping. The LSTM capture proves an approach for
retaining original decoder operands without editing the model; its saved operands
are recurrent inputs and outputs, not the final projection input.

The current staged profiler also cannot isolate kernel cost here. In
TensorOps.MatMul.cs, the Math stage begins before RunBatchedFloatMatMul, which
still performs RequireBatchOperand, batch-stride planning, map selection,
pinning and loop dispatch. The non-pooled overload also allocates its output
after entering that stage. A large Math interval therefore does not prove a
large arithmetic-kernel interval. LastAllocatedBytes and LastCopyBytes are
whole-execution totals, not per-node measurements.

After the fixed sigmoid release qualification, use its actual qualified product
and the retained full-model decoder trajectories. Load the original decoder with
the same 64 MiB preparation budget and ExecutionOptions.Memory. Preserve the
model's nodes, attributes, original outputs and recurrent decisions. Retain the
actual projection operands only through execution-local output bindings, with
ordinary control executions and complete output/ownership checks as in the
existing decoder capture.

Record the source-array identity, packed-map entry and reference relationships,
tensor types, strides, dimensions, options, and hashes before and after execution.
Record whole-decoder copy/allocation counters with their actual scope. Zero
whole-decoder counted copies can rule out a counted final-weight materialization;
nonzero totals cannot be assigned to this node without further evidence. Reuse
existing managed execution sampling/code capture to identify the invoked one-row
kernel while running the original decoder. Retain all versions, samples and
ordinary controls; do not call an instrumented time a performance score.

Stop once the observation distinguishes the decisions: if B is already mapped,
the observed one-row route bypasses that mapping, and samples place the material
cost inside that kernel, one layout-only intervention is justified. If mapping
is absent or copying dominates, address that observed cause instead. If sampling
does not identify the path, record the missing evidence and change the observation
mechanism; repeating the same capture is not a diagnosis.

ORT's retained one-row samples still lack a per-node caller join. Its original
node profile and pinned MatMul implementation support prepared-B use; avoid
attributing all one-row sample weight to this projection. The existing packed
helper's accumulator caveat also remains in force. No new implementation,
inference, timing or release claim is introduced by this note.

Do not make a new native trace a prerequisite unless that missing association
could change the layout decision. If it is needed, collect node intervals and
instruction samples in the same process and execution. The existing profile and
perf captures were separate runs and cannot supply that association. Repeating
the leaf-only capture cannot repair the missing information.

The clocks also need an explicit correspondence. At pinned ORT revision
`2e2543fbe9fae542f921d47a72d21d5a4ef0b710`, `Profiler::GetStartTimeNs()` returns
the high-resolution clock's epoch count; node `ts` and `dur` are truncated to
microseconds. It does not promise the Linux monotonic epoch used by the retained
perf capture. The actual decoder profile start is `1790241475510600688` ns,
whereas the first application's monotonic call starts at `607186763379246` ns.
Adding `profile_start_ns` to `ts * 1000` therefore does not produce a perf
timestamp. A future combined capture must establish its clock mapping and
precision, verify process/thread and call boundaries, retain uncertain boundary
samples separately, and refuse association if those checks fail. No historical
timestamps have been shifted or combined here.

References: [current decoder partition](README.md),
[native one-row evidence](native-row1-20260927.md),
[accumulator caveat](packed-row-caveat-20260927.md),
[current MatMul implementation](../../../src/Lokad.Onnx/TensorOps.MatMul.cs),
[whole-graph accounting](../../../src/Lokad.Onnx/ComputationalGraph.cs),
[existing decoder capture](../decoder-lstm-capture-amd/Capture.cs), and
[encoder-only counters](../packed-final-row-counters/Program.cs.txt).

Clock sources: `onnxruntime/core/common/profiler.cc`, `profiler.h`, and
`include/onnxruntime/core/common/common.h`, read with `git show` at the revision
above; retained `artifacts/parakeet-ort-diagnosis-amd-20260924/collected/profile/observation.json`;
[original observer](../ort-diagnosis-amd/observer.py) and
[perf clock validation](../ort-diagnosis-amd/audit_native.py). This is source and
retained-data review, not a completed combined capture.
