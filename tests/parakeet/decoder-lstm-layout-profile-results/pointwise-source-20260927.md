# Exact pointwise matrix paths and the remaining attribution question

The next observation should separate arithmetic from materialization, clearing
and packing inside the 48 encoder pointwise convolutions. Their fresh profiled
times are 6.419682 seconds in Lokad and 4.390779 seconds in ORT. The source
comparison identifies specific differences, but it does not establish which
difference explains those 2.028903 seconds.

The [read-only review](pointwise_review.py) checks all 48 nodes and their 3,840
calls, including warmups, against the closed fresh native profile. It verifies
all 19 output widths and the relevant managed files against qualified source
`edc1a6c8`. It also verifies both retained stack exports against their recorded
hashes and reconciles each Speedscope export with its Chromium counterpart.
[Machine-readable results](pointwise-source-observations-20260927.json) include
the exact source and evidence identities. No inference or optimization ran.

## Applicable source paths

The matrices are `[2048,1024] × [1024,T]` and `[1024,1024] × [1024,T]`.
There is no bias operand. Both runtimes bypass patch expansion and use the
activation input as the right matrix, as established in the
[preceding diagnosis](next-diagnosis-20260927.md).

At ORT revision `2e2543fbe9fae542f921d47a72d21d5a4ef0b710`,
`onnxruntime/core/mlas/lib/sgemm.cpp:1181` starts the unpacked operation with
128 output columns and 128 reduction elements per block. The balancing loop
at line 1193 changes these to 64 columns and 256 reduction elements for
`T=51` and `T=61`. All other observed widths retain 128/128. Thus the
1,024-element reduction takes four or eight slices. Widths above 128 also
require two column slices. The source does not apply the packed-constant-B
256-element rule uniformly to these calls.

For each slice, `MlasSgemmCopyPackB` copies the activation panel into temporary
storage, then `MlasSgemmKernelLoop` traverses every filter row. The AVX-512
assembly in `x86_64/FgemmKernelAvx512FCommon.h:481` handles 12 rows whenever
at least 12 remain. Its 32-column block shares two loaded B vectors across
24 accumulators. With one unsplit matrix operation, 1,024 rows therefore use
85 groups of 12 and a final group of four; 2,048 rows use 170 groups of 12,
then six and two. These are source-derived counts. The retained native binary
proof identifies this routine across the application; it does not tag each
pointwise invocation with its executed row block.

Lokad's `RunFloatMatMulKernel` instead packs all 1,024 reduction elements and
uses `mm_unsafe_vectorized_intrinsics_2x4packed_bump` under the qualified default
options. Each 32-column block shares four AVX2 B vectors across two rows.
The full reduction is accumulated before storing the block. Its remaining
columns use eight-column vectors and a masked narrow tail. Row sharing,
vector width and reduction blocking are distinct differences; a candidate
must state which mechanism it changes.

The existing `TryPackedAvx512Rows` helper requires `k % 32 == 0`, where `k`
is the output width. **None of the 19 observed widths satisfies that guard.**
Enabling its dynamic flag would leave all these products on the existing
kernel. That flag is not a useful experiment for this workload.

Before the matrix call, `Conv2DFloatCore` creates or rents a cleared output
and calls `ToDenseTensor` on input and weights. Row-major `DenseTensor`
instances return themselves; other layouts may materialize. The pointwise
routine then creates three dense matrix views without copying their buffers.
The destination-taking `MatMul2D` clears the output again. Those statements
identify possible costs; they do not measure the input layout or the time
spent clearing in these actual calls.

## Why the retained managed samples do not settle the choice

Both September 23 exports contain the two-row packed kernel and the broader
`Conv2DFloatCore` frame. Neither contains a `RunPointwiseBatchesFloat` frame.
Those convolution stacks cover other convolution paths as well. This review
therefore does not assign their historical kernel or packing percentages to
the 48 current pointwise nodes. Adding another unchanged whole-application
stack capture would not guarantee that the missing boundary becomes visible.

The next diagnostic should reuse the complete public-request consumer with a
scoped observer for these exact pointwise shapes. Record actual input layouts,
materialization, destination clearing, input packing, selected leaf and elapsed
arithmetic separately, retaining the complete pointwise total and its remaining
time. Include every layer and clip; exclude the different stem convolutions by
geometry. Reconcile 48 calls per encoder request, original public results and
unmodified graph outputs. Measure observer overhead against its control and
retain it rather than subtracting it from individual costs.

If arithmetic dominates, the first candidate must address the observed kernel
limitation while holding unrelated packing and materialization behavior fixed.
If data movement accounts for the excess, address that measured work first.
Do not combine reduction blocking, a packing-layout change and a new row kernel
in one trial. Any arithmetic-preserving candidate must retain the existing
reduction and FMA operand order, including the distinct non-FMA narrow tail.

This remains diagnosis. The qualified application score is unchanged:
46.931723 seconds versus ORT 39.513180, ratio 1.188. The profiled pointwise
gap is an opportunity estimate of about 4.3% of application time, not a
predicted or measured speedup.
