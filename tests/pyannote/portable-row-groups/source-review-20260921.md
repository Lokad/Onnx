# Packed row grouping: Lokad and Microsoft ORT source review

The complete public-request diagnostic identifies Lokad's packed two-row
matrix kernel as the largest remaining pyannote leaf. This review motivates
the adjacent experiment; it does not establish a performance improvement.

The exact qualified source is retained under
`artifacts/pyannote-convolution-pool-v5-20260921/source/src/Lokad.Onnx`.
`MathOps.cs` defines both `mm_unsafe_vectorized_intrinsics_2x4packed_bump`
and `mm_unsafe_vectorized_intrinsics_3x4packed`. Both consume the same
`PackPanelsB` layout: blocks of32output columns across the complete reduction,
then a row-major column remainder. Each output accumulates in ascending
reduction order. Complete eight-column vectors use FMA; the final fewer-than-eight
columns use separate multiply and add.

The two-row full-panel loop holds eight vector accumulators and shares each
right-hand vector between two output rows. The three-row loop holds twelve
accumulators and shares it between three rows. `RunFloatMatMulKernel` selects
the latter only when the entire row count is divisible by three. The captured
embedding convolutions instead have32,64,128or256output channels. They reach
the two-row path whenever packed dispatch applies.

The proposed composition takes the largest multiple of six rows through the
existing three-row kernel and sends the remaining zero, two or four rows to
the existing two-row kernel. This avoids a single-row remainder and shifts only
the left operand and destination pointers. Neither kernel nor packed layout
changes. Reduced operand loading is a hypothesis: register allocation, call
overhead and cache behavior still require measurement.

Microsoft ORT1.29 source at revision
`2e2543fbe9fae542f921d47a72d21d5a4ef0b710` supplies a related design in
[FgemmKernelFma3Common.inc](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/mlas/lib/amd64/FgemmKernelFma3Common.inc).
`ComputeBlockFma3By2` loads two YMM vectors from packed B, broadcasts an A value
per row, and updates up to twelve accumulators for six rows. Its dispatch
processes up to six available rows, including explicit one-through-five-row
paths. Its store logic distinguishes overwrite and accumulation and supports
masked final columns. This shows that sharing packed operands across more rows
is part of an established native design. It does not establish which kernel
the installed ORT wheel executes or predict a .NET speedup.

The retained assembly file is19,156bytes, SHA256
`08ff22acc5fc5204152d58265157b56d879b60536ca943cf75b882b03447fc3c`.
The accompanying `FgemmKernelCommon.inc` is7,310bytes, SHA256
`4b361dc95547b7a55c91781cc3dced26de3c95df517898a353e22ded493d6885`.
Preparation independently verifies both against their original
`artifacts/parakeet-reduction-source-20260921/kernel-sources.json` receipt.

The probe derives sixteen full/final tile shapes from the retained model census,
includes packing and destination clearing, and uses the exact qualified Core
assembly. Synthetic operands test geometry and ordered arithmetic; captured
model outputs and complete public application timing remain required before
any production or AMD admission. The separate frozen AVX-512 convolution
candidate is unaffected.
