# Constant packing in the selected Parakeet release and ORT

This review concerns selected source `81f75c38`, Core `672e5f30` / Data `065b7a7f`,
and retained Microsoft ORT source `2e2543fbe9fae542f921d47a72d21d5a4ef0b710`.
It identifies a bounded hypothesis; it is not current attribution or a speed
result. The separately frozen complete-request profile must finish first.

[`GraphPacking.FitsPackBudget`](../../../src/Lokad.Onnx/GraphPacking.cs) requires
`n < 4096`. Here `n` is B's row count, the reduction length; `k` is B's column
count, the output width. A `[4096,1024]` constant is excluded by that boundary.
Preparation additionally requires a plain row-major rank-two float initializer,
eligible consumers, independent packed storage and the remaining graph budget.
The same predicate also gates use through `ResolvePackedKernel`, so changing
the boundary would affect both preparation and dispatch.

[`ParakeetTranscriber`](../../../src/Lokad.Onnx.Data/ParakeetTranscriber.cs) gives
the encoder 256 MiB and decoder 64 MiB of retained packing. Preparation consumes
that shared cap in traversal order. Admitting a previously excluded weight can
displace another one. The older Windows census changed 37 retained encoder
weights to 28 at the same 256 MiB; it is preserved in the
[earlier admission report](../packing-admission/results-20260921.md).
Those older counts and correctness results do not qualify this AMD product.

A prepared mapping does not prove that a particular call uses it.
[`ResolvePackedKernel`](../../../src/Lokad.Onnx/TensorOps.MatMul.cs) also requires
SIMD/intrinsics/FMA and an even row count or a multiple of three. Prepared calls
then use the existing AVX-512 row-sharing dispatcher when its guards admit them,
with established AVX2 fallbacks. Without a usable mapping, the selected
[`wide-entry path`](../../../src/Lokad.Onnx/Zzz.WideProjectionEntry.cs) can enter
the isolated per-call packing and AVX2 consumers. That path already belongs to
the qualified release. An inclusive 4096 boundary would change which existing
path runs; it would not merely eliminate a memory copy. Current invocation
shapes, residency, fallback coverage and full application effects all matter.

ORT's pinned [MatMul implementation](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/providers/cpu/math/matmul.cc)
prepares constant B through `GemmPackBFp32` and passes a packed-buffer flag to
MLAS. Its [preparation helper](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/providers/cpu/math/gemm.cc)
checks rank two, derives reduction/output dimensions with transpose handling,
and asks MLAS for the required storage. That helper has no corresponding
hardcoded reduction-length-4096 exclusion. This source comparison does not
identify the installed wheel's actual dispatched assembly kernel.

[MLAS SGEMM](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/mlas/lib/sgemm.cpp)
has distinct packed and unpacked traversal. Its beta-zero first reduction block
can overwrite C; subsequent blocks accumulate. Lokad's current clear-and-ordered-
accumulation behavior and nonzero-destination contracts must remain intact.
Different layouts, scheduling and arithmetic contracts prevent importing a
kernel fragment on the strength of a matching tile size.

If the new profile supports another packing trial, adapt voice-branch `c679bf99`
as a separate change at unchanged 256/64 MiB budgets. First prove actual bounded
residency and the exact compiled delta, then numerical/ownership behavior and
complete Parakeet/native/public/application performance. Shared models, e5,
Pyannote, normal-root suites and package qualification remain promotion gates.
Global dynamic AVX-512, padding and reduction-blocking trials remain closed;
this review does not reopen them or authorize larger resident budgets.

[Reviewed source identities](packing-source-observations-20260924.json) retain
the current file hashes and the pinned ORT copy identities. No product source,
public API or benchmark ratio changes as part of this inspection.
