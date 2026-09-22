# Convolution loop structure: pinned ORT source and selected Lokad kernel

At ORT revision `2e2543fbe9fae542f921d47a72d21d5a4ef0b710`, the
[common convolution macro](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/mlas/lib/x86_64/SconvKernelCommon.h#L173)
loads kernel height and width at runtime. `ProcessNextRow` and
`ProcessNextColumn` decrement their counters and branch. For the NCHWc
16-channel format, it emits sixteen `ComputeBlock` expansions at each spatial
kernel position. Thus this source unrolls the channels within a block, not
the complete 3x3 spatial kernel. The
[AVX512 implementation](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/mlas/lib/x86_64/SconvKernelAvx512F.S#L111)
defines broadcasts, weight loads and fused multiply-adds for those expansions.
This is source evidence, not proof of installed-wheel dispatch or latency.

Lokad's current `ConvBlockedSpatial.Kernel512` reduces input channel, kernel
row, kernel column, in that order. Its six-position/two-output-block tile has
twelve independent vector accumulators. Retained selected Tier1-OSR listings
of 2,176 and 2,192 bytes still contain the full-tile column counter and
backedge; their tail paths already unroll that loop. The kernel method is
unchanged by the subsequently integrated M22 LSTM improvement.

A different experiment can explicitly expand all nine 3x3 positions of the
full tile while retaining the input-channel loop and the exact row/column
order. Keep the six positions, two output blocks, FMA chain, weight increments,
stores, scalar/nonfinite fallbacks, AVX2 implementation and tails unchanged.
No extra weights or scratch are required. This is a Lokad fixed-geometry
specialization; it does not copy ORT's channel/spatial reduction order.

The hypothesis is that fewer branches and constant offsets reduce complete
convolution cost. Code growth and register allocation may negate that benefit.
The rejected address-hoisting experiment improved complete calls only 3.55%,
so source-level simplification alone is weak evidence. Require actual AMD
numerical coverage and generated-code inspection, then the unchanged fixed
complete-call and full-application gates. No speed improvement is claimed here.

Files were read using `git show` against the pinned revision, preserving the
local checkout at `a83fc4d58cb48eb68890dd689f94f28288cf2278`.
Source copies and manifest: `artifacts/pyannote-kernel-loop-review-20260922`.

| Source | Bytes | SHA-256 |
|---|---:|---|
| SconvKernelCommon.h | 26,183 | `7077407221b380e20ff5368048538cc0876b7161cbc8786aac5dff3ea491d9b1` |
| SconvKernelAvx512F.S | 27,153 | `6c36d30d2c3fd290b73416fa37f9fe5ba93383818605866ba10fed21a576872a` |

Retained Lokad listings:
`artifacts/pyannote-spatial-weight-codegen-20260922/listings.json`, SHA-256
`7d7b7c315281882bf3ad1585d7ba7437c3964c2b8ba5a8c111d65d134a79f3e4`.
