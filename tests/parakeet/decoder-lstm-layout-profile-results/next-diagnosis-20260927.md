# Narrow the next Parakeet investigation to pointwise matrix execution

The [fresh complete partition](diagnosis-20260927.md) changes the priority.
Decoder time is 1.933698 seconds managed versus 2.008664 seconds in ORT.
Further decoder tuning has a weaker justification than the remaining encoder
work. These are profiled times; the independently qualified application remains
46.931723 versus 39.513180 seconds, ratio 1.188.

The 2.704226-second constant-projection difference is spread across several
families. The breakdown below joins each of 217 native matrix nodes to its exact
managed node and the 48 multiply edges fused by ORT. It retains all 265 managed
nodes, all original clocks, constant-weight dimensions and every runtime shape.
Each native node has 80 calls including warmups and 60 measured calls. All 217
expose only A in the native runtime-input records. The applicable float MatMul
source uses a prepared B buffer; this supports preparation without claiming a
new dump of internal buffers or per-node native kernel sampling.

| Projection family | Matrices | Managed seconds | ORT seconds | Difference |
|---|---:|---:|---:|---:|
| Feed-forward expansion | 48 | 11.304318 | 10.582636 | 0.721682 |
| Feed-forward contraction, including scale | 48 | 10.982254 | 10.607148 | 0.375106 |
| Attention key | 24 | 1.562219 | 1.321999 | 0.240219 |
| Attention output | 24 | 1.569955 | 1.324667 | 0.245288 |
| Attention positional | 24 | 3.212655 | 2.635378 | 0.577277 |
| Attention query | 24 | 1.575842 | 1.320432 | 0.255410 |
| Attention value | 24 | 1.558756 | 1.322560 | 0.236196 |
| Stem output | 1 | 0.273825 | 0.220776 | 0.053049 |

[Exact members and shape checks](projection-breakdown-20260927.json) come from
[the read-only calculation](projection_review.py). No historical clocks or new
inference contribute. The 120 square attention projections together differ by
1.554389 seconds. A generic matrix-kernel change would mix their prepared and
unprepared routes with the separate owned feed-forward path.

The 48 pointwise convolutions are a larger common execution family: 6.419682
seconds managed versus 4.390779 seconds ORT, difference 2.028903. Their observed
input is [1,1024,T]; the first 24 use [2048,1024,1] filters, the second 24 use
[1024,1024,1]. Both observe the same 19 values of T:
51, 61, 83, 88, 89, 102, 106, 112, 114, 120, 151, 156, 157, 158, 167, 169, 190,
222 and 225. Original graph attributes and parameter edges were validated in the
[convolution correspondence](../managed-phase-results/convolutions-20260924.md);
the fresh partition validates unchanged managed descriptors, native metadata,
runtime shapes and counts. Its old timings are not used here.

## What the source establishes for these calls

At installed ORT revision `2e2543fbe9fae542f921d47a72d21d5a4ef0b710`,
`onnxruntime/core/mlas/lib/convolve.cpp:1646-1660` chooses
`MlasConvAlgorithmGemmDirect` for this unit-stride, unpadded, size-one kernel.
The working convolution buffer is zero-sized. At lines 1248-1250, the direct
route invokes GEMM with filter rows as A and the original input as B. The two
matrix geometries are [2048,1024] × [1024,T] and [1024,1024] × [1024,T]. B varies
on each call. The prepared constant-B behavior of ordinary encoder MatMul nodes
does not describe these convolution products.

Lokad also bypasses patch expansion. In
[Conv2DFloatCore and RunPointwiseBatchesFloat](../../../src/Lokad.Onnx/TensorOps.ConvPool.cs),
the size-one route creates matrix views over input, filter and output, then
calls MatMul2D. With the qualified default flags and these shapes, the source
predicts ordinary input-panel packing and the two-row packed kernel in
[RunFloatMatMulKernel](../../../src/Lokad.Onnx/TensorOps.MatMul.cs).
The isolated wide-projection route requires at least 1,024 output columns;
these products have T <=225. Neither matrix row count is divisible by three.

The [retained native binary proof](../ort-diagnosis-results/ort-kernels-20260924.md)
identifies MLAS's AVX-512 SGEMM family, which contains a twelve-row kernel.
Those samples cover the whole application. They do not establish the executed
row block of each of these 48 convolution nodes. The next source inspection
must follow the unpacked `MlasSgemmOperation` route, including its input packing
and reduction blocking; the packed-constant LSTM route is a different case.

That source follow-up is now recorded in
[the exact pointwise review](pointwise-source-20260927.md). ORT uses four or
eight reduction slices for these widths. Lokad's existing AVX-512 helper rejects
all 19 widths, even if its dynamic flag is enabled. The retained managed stack
exports do not separately identify the pointwise boundary, so their older
percentages cannot choose between arithmetic and data movement here.

## One bounded diagnostic question

Determine how much of the pointwise difference lies in input packing and matrix
arithmetic, and whether the expected two-row versus MLAS row-sharing routes
actually execute. Account for input materialization and output clearing before
attributing the difference to arithmetic. Reuse retained source and traces where
they answer this question; observe only missing execution detail that could
change the intervention. Preserve original graph outputs and complete results.

A candidate requires one causal explanation and a predicted reduction in the
affected complete calls. The 2.028903-second profiled difference is an opportunity
estimate, about 4.3% of qualified application time, not promised savings.
No new kernel, flag change, packing-budget change or optimization trial is selected
by this report. Keep Parakeet first, Pyannote second and Whisper deferred.

The subsequent [scoped cost observation](pointwise-costs-20260927.md) is now
complete. It assigns 95.77% of pointwise time to arithmetic, confirms the actual
two-row leaf and dense inputs, and selects one column-remainder intervention
from source and measured shape dependence. That report supersedes this open
diagnostic question; no optimization has yet been scored.
