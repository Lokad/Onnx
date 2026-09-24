# Parakeet masking and padding: complete matched work

The 72 float selection operations and48 padding operations, including their
input construction, cost **6.207072 profiled seconds in Lokad versus0.144281
in Microsoft ORT**: a **6.062791-second difference** per twenty-clip corpus.
These are retained selected-release profiles, not a fresh candidate benchmark.
No model inference or new optimization ran for this analysis.

| Matched family | Operations | Lokad seconds | ORT seconds | Excess seconds |
|---|---:|---:|---:|---:|
| Attention mask |24|1.228600|0.036694|1.191906|
| Attention cleanup |24|1.223299|0.023666|1.199634|
| Convolution mask |24|0.821441|0.033787|0.787654|
| Attention padding |24|2.201273|0.034881|2.166392|
| Convolution padding |24|0.732725|0.015294|0.717430|

Family totals overlap at shared constants and mask views. The combined figure
above counts every ancestor once:541 managed nodes and122 ORT nodes. The earlier
[convolution attribution](convolutions-20260924.md) already includes convolution
masking/padding; do not add these reports together without removing overlap.
The tiny Int64 shape-selection Where is outside this120-operation scope.

The comparison matches each output and its data/mask boundaries, scalar bits,
padding values, attributes and all19 native runtime lengths (51 through225).
Each operation has60 measured calls; the shape census also retains the warmup
pass. Original small integer expressions are evaluated solely as graph metadata
to verify ORT's folded padding constants. No model arithmetic is executed.

ORT shares the mask views, removes redundant boolean casts and folds the padding
shape construction. Those differences explain only about0.05s of Lokad cost.
The actual Where kernels account for **3.169154s** of excess and Pad kernels for
**2.842889s**. More folding alone would leave almost all of this difference.

The source gives specific mechanisms to verify. Lokad's Where repeatedly maps
flat indices through broadcast views; PadCore performs division and remainder
per element and per axis. Pinned ORT Where processes broadcast spans through
selection and merge passes; constant Pad copies contiguous inner blocks and
fills padding around them. ORT is faster here despite making multiple Where
passes. This evidence favors investigating element access and bulk copying;
it does not identify the exact generated native instructions or prove the
benefit of a proposed managed change.

The matched geometries are:

| Family | Data/output | Condition or padding |
|---|---|---|
| Attention Where | `[1,8,T,T]` | condition `[1,1,T,T]`; scalar true value -10000 or +0 |
| Convolution Where | `[1,1024,T]` | condition `[1,1,T]`; scalar true value +0 |
| Attention Pad | `[1,8,T,2T-1]` → `[1,8,T,2T]` | one leading zero on the final axis |
| Convolution Pad | `[1,1024,T]` → `[1,1024,T+8]` | four zeros on each side of the final axis |

The next observation must establish concrete managed layouts and mask values
for **all20 corpus clips and every target node**. The older two-clip Where
capture proves all-false masks only for its six exported layer-zero fixtures;
it does not supply that full-corpus observation. Keep the optimized graph and
public transcription flow intact, retain original result/input/ownership checks,
and make no performance claim from an instrumented capture.

Prior padding and Where screens retain their failed controls and rejection
verdicts. In particular, the6.772% padding regression was a synthetic cropping
fallback case, not a measured full-application regression. The identical-binary
Where control also failed. These outcomes do not establish a full-model gain
or falsify the complete mechanism; they do rule out treating those screens as
valid admissions or repeating their unchanged designs.

Source references: [Lokad Where](../../../src/Lokad.Onnx/TensorOps.Elementwise.cs),
[broadcast index mapping](../../../src/Lokad.Onnx/BroadcastedTensor.cs),
[Lokad Pad](../../../src/Lokad.Onnx/CPUExecutionProvider.Shape.cs),
[pinned ORT Where](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/providers/cpu/tensor/where_op.cc),
[pinned ORT Pad](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/providers/cpu/tensor/pad.cc).
Microsoft sources were read from the retained Git revision, not the older checkout.

Closure: `0299d2e1d273b2651d45209cb7c9dffb623ae4149cf0cb5eed23eaf89d842e76`.
Raw evidence: `artifacts/parakeet-masking-padding-attribution-20260924`.
[Every matched operation](masking-padding-20260924.csv),
[shared work, source identities and limitations](masking-padding-observations-20260924.json),
and [analysis implementation](../managed-phase-amd/compare_masking_padding.py).
