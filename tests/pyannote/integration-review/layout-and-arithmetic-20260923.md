# Pyannote: layout continuity and convolution arithmetic

The selected product still converts each eligible convolution input and output
between ordinary and blocked channel order. ORT keeps compatible connections
blocked across the embedding graph. This is established by both source review
and the [actual AMD graph and execution capture](../native-layout-amd/results-20260922.md),
which finds 36 blocked convolutions, 16 fused residual additions, 33 ReLU
attributes and one output reorder. It does not identify the native assembly
microkernel or establish that ORT uses Winograd.

The pinned ORT revision is `2e2543fbe9fae542f921d47a72d21d5a4ef0b710`.
In `onnxruntime/core/optimizer/nchwc_transformer.cc`, `InsertReorderInput`
reuses the mapping for an ordinary graph value. Convolution handling at
lines487–525 consumes an existing blocked value when available. Add/Sum and
activation handling at lines705–740 and909–945 require suitable single-use
producers and respect already-fused operations. `Finalize` at lines1348–1364
adds an ordinary-layout output only where an original-format consumer remains.
The source copy is retained under
`artifacts/pyannote-layout-continuity-review-20260923`; transformer SHA256 is
`a69e8fd310dd75c599b3016b45788a6006f8aa233f851ba2fe5c35ebc6947995`.
The local ORT checkout was not changed.

In selected Lokad source, `TensorOps.ConvBlocked.cs:TryConvBlockedSpatial`
rents per-call input/output scratch. `ConvBlockedSpatial.Execute` packs input,
computes convolution, then calls `UnpackEpilogue`. Its graph caller currently
passes no residual and disables the private ReLU epilogue.
`CPUExecutionProvider.Fusion.cs:ConvRelu` applies ReLU to the fresh ordinary
output afterward. `GraphConvPacking` owns immutable prepared weights;
`GraphExecution` shares those maps but owns request buffers. A layout change
must preserve that distinction and all exposed intermediate outputs.

The [current sampled profile](../current-profile-results/results-20260922.md)
helps assess which mechanism deserves a prototype:

| Exclusive leaf, full dialogue | Capture A seconds | Capture B seconds |
|---|---:|---:|
| Direct blocked convolution kernel | 28.463672 | 28.468072 |
| Input packing | 0.520351 | 0.490223 |
| Output unpacking/epilogue | 0.797745 | 0.779853 |
| Finite-operand scan | 0.339647 | 0.302495 |
| All sampled request time | 43.855161 | 43.866097 |

The kernel accounts for about64.9%; the two conversion helpers together
account for about3.01%/2.90%. These are diagnostic samples, affected by
inlining and profiler overhead. They are neither exact phase clocks nor a
bound on the benefit of changing layouts. Layout continuity and fused
epilogues remain useful options, but arithmetic is the larger concentration.

After the rejected M28 application and M29 component trials, the next isolated
experiment is [small-tile Winograd convolution](../winograd-prototype/README.md).
It uses the mathematical transforms from
[Lavin and Gray, section4.1](https://arxiv.org/pdf/1509.09308), with sixteen
transformed products for four outputs instead of36direct products per channel
pair. Transformation and memory costs still have to be measured. No external
implementation is copied, and no product dispatch changes at this stage.

The arithmetic changes finite rounding. Its prospective numerical gate keeps
the original `1e-4` scaled native bound for every one of87eligible captured
calls, in both instruction widths, with an independent double-precision
reference for synthetic cases. Historical bit-exact loop experiments retain
their original verdicts. A numerical success would only admit a separate
performance screen; complete application checks still determine integration.
