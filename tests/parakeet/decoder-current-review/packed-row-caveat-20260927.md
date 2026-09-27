# Keep accumulation behavior fixed when testing the one-row weight layout

The existing `PackedFinalRowKernel.Multiply` is not a direct equivalent of the
source-inferred decoder kernel with only the B layout changed. Its source reads
and writes each output vector inside every reduction step. The current
`mm_m1_kblocked` instead names four AVX2 accumulator variables, keeps them through
the reduction loop and stores them afterward. Both use ascending reduction order
and FMA for complete vectors; the final six columns of this geometry are scalar.

These are source-level observations. Generated code could change the actual load
and store behavior, so this does not establish the helper's measured performance.
It does mean that merely routing one row into the existing helper would not, from
source alone, isolate packed weights as the cause of a gain or loss.

After confirming the actual constant mapping, dispatch and complete-call cost,
a layout experiment should preserve the current four accumulators, vector width,
reduction order, scalar tail and output ownership while changing only the weight
addressing to the already prepared panel layout. Alternatively, first establish
that the existing helper emits equivalent accumulation behavior. Do not introduce
wider vectors, different unrolling or graph fusion into that experiment.

This is an additional restriction on the [next decoder decision](README.md), not
a new selected implementation. The [observed ORT one-row branch](native-row1-20260927.md)
still has no per-node caller association. The profiled final projection/bias
excess remains about 1.128 seconds; it is an opportunity bound, not a prediction.

Source: [packed final-row helper](../../../src/Lokad.Onnx/PackedFinalRowKernel.cs),
[current one-row multiply](../../../src/Lokad.Onnx/MathOps.cs),
and [prepared-weight dispatch](../../../src/Lokad.Onnx/TensorOps.MatMul.cs).
No source change, model execution or new timing was performed for this review.
