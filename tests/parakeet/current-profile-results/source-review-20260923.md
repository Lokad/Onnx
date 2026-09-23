# Parakeet matrix multiplication: current Lokad and ORT source

The current source offers a focused hypothesis: reuse the existing AVX-512
row-sharing kernels for eligible large matrices that are packed during a call.
This could reduce computation without raising Parakeet's retained-weight budget.
Selection requires the current AMD profile and a separate complete-call trial.
No performance result follows from this source review.

The inspected Lokad product is `94a550de`, measured as Core `521bae17` and Data
`f3b9aa81`. ORT source is pinned to
`2e2543fbe9fae542f921d47a72d21d5a4ef0b710`; the installed comparison is ORT1.29.0.
Source inspection identifies mechanisms, not the installed wheel's actual dispatch.
The separate checkout at `external/onnxruntime` remains unchanged at `a83fc4d5`.

## Where the current paths differ

[`GraphPacking.FitsPackBudget`](../../../src/Lokad.Onnx/GraphPacking.cs) accepts
reduction length strictly below4096. Graph preparation also respects a shared
retained-weight byte cap, initializer ownership, layout and consumer eligibility.
[`ParakeetTranscriber`](../../../src/Lokad.Onnx.Data/ParakeetTranscriber.cs) sets
256MiB for the encoder and64MiB for the decoder. Its request includes frontend,
encoder, greedy decoding, input validation and independently owned contexts/results.

[`ResolvePackedKernel`](../../../src/Lokad.Onnx/TensorOps.MatMul.cs) requires
SIMD/intrinsics/FMA and even rows or exact three-row groups. With a usable prepared
mapping, `RunPreparedPackedRows` enables AVX-512 row sharing by default.
[`TryPackedAvx512Rows`](../../../src/Lokad.Onnx/MathOps.PackedAvx512.cs) handles
full32-column panels with12- and8-row tiles plus established2/3-row remainders.
Its lane accumulators still traverse the reduction in increasing order.

Without that mapping, `RunFloatMatMulKernel` can rent scratch and pack B on every
call once its row threshold and256MiB element-derived scratch bound permit it.
The existing AVX-512 consumer in these branches is disabled by default through
`EnablePackedAvx512Dynamic`. The ordinary path therefore uses the AVX2
`mm_unsafe_vectorized_intrinsics_2x4packed_bump` or `3x4packed` consumer. An odd
remaining row uses the original unpacked single-row kernel. Large axes with too
few rows keep their older fallback. These conditions must stay explicit in any
trial: a stored mapping alone does not establish which kernel executes.

The historical twenty-clip census found37of217encoder weights prepared and no
prepared `[4096,1024]` weights. That census was a Windows diagnostic of an older
product. It motivates inspecting per-call packing and consumption on the current
AMD workload; it does not quantify their current cost.

## Microsoft ORT mechanisms

ORT's pinned [MatMul implementation](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/providers/cpu/math/matmul.cc)
prepares constant input B through `GemmPackBFp32`. Compute uses the stored B shape
and passes the packed-buffer flag to MLAS with beta0. The
[packing helper](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/providers/cpu/math/gemm.cc)
accepts rank-two weights, derives N/K with transpose handling, and asks MLAS for
the required storage. It has no corresponding hardcoded K<4096 boundary.

[MLAS SGEMM](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/mlas/lib/sgemm.cpp)
has separate packed and unpacked traversal. The unpacked path packs bounded
column/reduction blocks; the packed path iterates column slices and reduction
blocks over its retained buffer. The first reduction block can overwrite C when
beta0; later blocks accumulate. Lokad's current clear-and-accumulate contract
must not silently inherit those overwrite semantics.

The GNU x86-64
[AVX-512 kernel macros](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/mlas/lib/x86_64/FgemmKernelAvx512FCommon.h)
load two B vectors and reuse them across as many as twelve row accumulators.
Each A broadcast supplies both output vectors. This is the same broad reuse
mechanism already available in Lokad's prepared-weight tiles. ORT additionally
uses assembly scheduling, reduction blocking and different packed layouts;
enabling row sharing alone does not imply equivalence or parity.

## What to test next

If the current profiles show substantial AVX2 packed-consumer cost, first test
existing AVX-512 row sharing in a bounded large-matrix per-call scope. Include
scratch rental, packing, destination clearing, consumption and odd-row tails in
the measured boundary. Keep the retained256/64MiB budgets and all raw kernel
arithmetic unchanged. Test both instruction widths, unsupported shapes and
nonzero accumulators before complete native/public/application qualification.

Treat packing admission and larger residency budgets as separate alternatives.
The voice-branch `c679bf99` boundary adaptation already has isolated numerical
qualification; its larger-budget Windows application trial stopped at the memory
guard. That trial supplies no admitted speedup and must not be recycled as one.
Likewise, the conditioned wide-kernel Windows trial failed its control limits.
Neither result authorizes a blanket import or an unchanged favorable retry.

The existing dynamic switch also has an earlier e5 trial, retained in
`artifacts/dynamic-packed-core-20260918`. Its twenty-worker complete-model
comparison produced only a0.68% median change at512tokens, smaller than the
repeat spreads, with mixed shorter cases; it stayed default-off. A Parakeet
candidate must establish value on its distinct wide projection shapes. This
review does not reopen the earlier e5 verdict or justify enabling the broad
dynamic switch globally.

Exact reviewed source hashes are in [source observations](source-observations-20260923.json).
Full source copies are retained under
`artifacts/parakeet-current-source-review-20260923`. No product edit, VM worker,
new benchmark ratio or native dispatch assertion is part of this review.
