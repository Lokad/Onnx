# Parakeet scalar-mask selection: next source hypothesis

Inspect `Where` as a separate optimization after rejecting ordered matrix
blocking. No new product implementation or performance result is selected here.
The current release remains source `81f75c38`, Core `672e5f30` / Data `065b7a7f`.

The retained encoder graph has 73 Where nodes. In 72 float nodes the true input
is a scalar: 48 select zero and 24 select -10000. False inputs come from attention
division, softmax and convolution multiplication. This establishes graph structure;
concrete runtime tensor classes, strides and broadcast runs remain to be captured.
The next diagnostic must record those layouts with the selected product before
freezing candidate scope and complete-call fixtures.

The earlier complete-request profile attributed 3.780% and 3.895% of sampled
leaf weight to Where. That profile used `94a550de`; it is neither a current
latency prediction nor proof of achievable gain. Current
[`TensorOps.Elementwise.cs`](../../../src/Lokad.Onnx/TensorOps.Elementwise.cs)
and [`BroadcastedTensor.cs`](../../../src/Lokad.Onnx/BroadcastedTensor.cs) remain
byte-for-byte identical to the sources inspected with that profile. Their hashes
are `78fb682e983598663e415e43460d70a97dea6e52ed5055870b7890fe52d946e8` and
`ffdc2b46f6bcf84f857c98fa82b9de83aedbf1b4568c99ef363c2cf84e754134`.

Where currently broadcasts all three inputs, creates an owned output and uses
GetValue for each selected element. BroadcastedTensor decomposes the flat index
using division and remainder at every axis. Even a broadcast scalar repeats this
work. A narrow dense path could cache the scalar and select contiguous runs,
advancing outer broadcast coordinates once per run.

Microsoft ORT's official Where source at
`2e2543fbe9fae542f921d47a72d21d5a4ef0b710` uses broadcast spans, separate
selections and merging. The retained file is
`artifacts/parakeet-memory-source-review-20260923/ort/onnxruntime/core/providers/cpu/tensor/where_op.cc`,
SHA-256 `1f34fc25892f558d36b43309548849bcd616f03a1461a6de4f9bce9f40aeae2d`.
This source motivates processing runs, but does not establish installed-wheel
dispatch or authorize copying merge arithmetic. Lokad must preserve selected
NaN payloads, signed zero, exact shape and independently owned output.

The proposed candidate is limited to float32, exact standard DenseTensor classes,
a scalar true input and dense false input already matching the output shape.
Keep the original broadcast path for views, reversed storage, custom subclasses,
unsupported shapes and other dtypes. Runtime layout capture, numerical tests,
compiled-code inspection, stable complete-call gains and full application/release
qualification are still required. Matrix arithmetic, padding, graph policies and
weight budgets remain separate work.

Evidence: [complete prior attribution](../current-profile-results/results-20260923.md),
[graph and pinned source observations](../current-profile-results/memory-source-observations-20260923.json),
and [rejected ordered-block screen](../ordered-wide-blocks-results/screen-20260923.md).
