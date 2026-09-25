# Parakeet stem: most of the difference is depthwise convolution

The two 3×3 depthwise convolutions explain **80.66% of the matched stem
difference**. They take **1.534429 s in M78 versus 0.043986 s in ORT** per
20-clip corpus. This is a more specific target than the whole stem or another
matrix-kernel search. No new inference ran for this analysis.

| Stem operation | M78 (s) | ORT (s) | Difference (s) |
| --- | ---: | ---: | ---: |
| Initial 1→256 convolution, stride 2, ReLU | 0.247606 | 0.035035 | 0.212571 |
| 256-channel depthwise 3×3, stride 2 | 1.214746 | 0.034179 | 1.180567 |
| 256→256 pointwise convolution, ReLU | 0.316659 | 0.200619 | 0.116040 |
| 256-channel depthwise 3×3, stride 2 | 0.319683 | 0.009807 | 0.309876 |
| 256→256 pointwise convolution, ReLU | 0.080070 | 0.049729 | 0.030341 |
| Native output reorder | — | 0.001601 | −0.001601 |
| **Complete matched boundary** | **2.178764** | **0.330970** | **1.847793** |

The analysis verifies the exact original/managed/native chain, convolution
attributes, initializer names/dimensions, fused ReLU edges and all observed
runtime shapes. Both chains start at `/pre_encode/Unsqueeze_output_0` and end
at `/pre_encode/conv/conv.1_2/Relu_output_0`. Every operator contributes all 60
measured calls. Native shape records include all 80 calls, including warmup.
The native output reorder is included once; its time is not omitted or added
to another row. These are the already closed profiles, not a fresh application
comparison or a release result.

## What the exact installed ORT source explains

All five executed native convolution nodes have domain `com.microsoft.nchwc`.
The initial convolution consumes the ordinary single-channel input directly
and produces blocked-channel output. The following convolutions consume that
layout; the sole final `ReorderOutput` restores ordinary NCHW. There are no
intermediate input/output reorder nodes within this boundary.

At installed revision `2e2543fbe9fae542f921d47a72d21d5a4ef0b710`, the
[NCHWc transformer](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/optimizer/nchwc_transformer.cc)
prepares the initial/depthwise filters in `OIHWBo` layout and pointwise filters
in `OIHWBiBo` layout. Preparation belongs to graph transformation, not each
request. The [NCHWc dispatcher](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/mlas/lib/snchwc.cpp)
chooses the dedicated depthwise algorithm when both input and output channels
per group equal one. It works directly on blocked-channel vectors, with no
expanded patch matrix or per-channel public matrix calls. Its platform-selected
[AVX-512 kernel](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/mlas/lib/x86_64/SconvKernelAvx512F.S)
keeps output accumulators in registers and applies bias at the end.

The executed NCHWc nodes and their shapes are observed. Internal kernel/ISA
selection is a source-derived prediction, not a new native leaf observation.
The optimized graph's temporary external weights were previously retired;
this analysis does not claim to compare reordered filter bytes or independently
observe the channel block size. The exact source copies and their hashes are
retained with the diagnosis.

## Why Lokad takes a different route

The exact M78 source's existing blocked convolution path requires `group == 1`
and at least 16 input channels per filter. Both depthwise nodes fail those
conditions. None of the five stem weights qualifies for the existing prepared
3×3 convolution representation: the spatial filters have one input channel,
and the remaining filters are 1×1.

The depthwise nodes reach `RunTiledBatchFloat`. The spatial scratch planner
budgets all channels together, then enforces a minimum 32-column panel. For
each panel it expands a patch, constructs three tensor views per channel, calls
the matrix dispatcher and copies the temporary output through the bias epilogue.
The direct-output and portable-row helpers reject the resulting `[1,9] × [9,N]`
products. This is a source explanation; actual internal counts still need a
runtime observation.

| Source-predicted work per corpus | Stem depthwise pair | 24 nine-tap module convolutions | Combined |
| --- | ---: | ---: | ---: |
| Matrix calls | 1,712,128 | 2,236,416 | 3,948,544 |
| Tensor views | 5,136,384 | 6,709,248 | 11,845,632 |
| Float values written into patches | 492,761,088 | 592,109,568 | 1,084,870,656 |

These are checked loop-count predictions from every actual shape and frequency,
not allocation measurements. Requested scratch is 327,680 bytes for each stem
depthwise call and 1,310,720 bytes for each module depthwise call; ArrayPool
bucket sizes have not been measured. Four boundary tests cover whole patches,
partial panels and the minimum-panel rule.

## The bounded next observation

The stem pair alone costs less than 3% of the complete M78 application time,
even before accounting for the work that must remain. It cannot justify the
existing 3% application-gain requirement by itself. The same managed tiling
mechanism serves the 24 nine-tap depthwise convolutions, which add 1.428194 s
versus ORT's 0.150435 s. Together, the 26 operators take **2.962622 s versus
0.194421 s**, a 2.768202-second difference with no overlapping Pad/SiLU work.

ORT's one-dimensional nodes use ordinary `Conv`, not NCHWc. The exact source
predicts segmented expansion and one unpacked M1 product per channel, as
documented in the [earlier dispatch analysis](../managed-phase-results/depthwise-dispatch-20260924.md).
Do not attribute the stem's blocked depthwise kernel to those nodes.

The next action is one instrumented execution of exact M78: observe the actual
input layouts, tiling, panels, tensor views, matrix calls and selected managed
leaf for all corpus shapes. Retain the complete public-result checks and exact
consumer/Data binaries. This capture is for counts; its elapsed times cannot
become a performance score. If the counts contradict the prediction, investigate
that discrepancy before selecting code. No optimization is selected yet.

The voice branch contains direct depthwise helpers, but they are reference
material only. They seed accumulation with bias, force compiler optimization,
change pooling/parallel behavior, and leave stride-two 2D execution scalar.
Any later reuse must preserve the current sum-then-bias contract and be justified
by the observed mechanism; the branch is not a ready-made qualified import.

The rejected vector Sigmoid screen remains closed. The root product and
`BENCHMARK.md` remain unchanged, including M78's independent short-e5 admission
failure. Repository allocation was last measured at 45.31 GB; this diagnosis
adds only source and report files.

[All stem means](stem-20260925.csv), [all shapes, counts and source identities](diagnosis-20260925.json),
[reconciliation code](review.py). Raw evidence is under
`artifacts/parakeet-stem-diagnosis-20260925`.

Closure: `469e436bf873876ff547db99ec1f42f2f6afd4da7669e83f911f039848944ca9`.
