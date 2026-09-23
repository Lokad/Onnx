# Parakeet padding, selection and packed-matrix traversal

The next small, well-defined Parakeet experiment is constant padding of the
last axis. All 48 encoder Pad nodes use this case, and PadCore accounts for
3.78%/3.85% of the two current AMD request profiles. A contiguous row copy can
avoid the existing per-element coordinate divisions. This is a source-backed
hypothesis; no new implementation or performance result is selected here.

The reviewed product is `94a550de`. The M43 candidate changes matrix dispatch
and first-use compilation, leaving these padding and selection methods exact.
Its release qualification is still running. The observations below do not
change the current release table or the order of that qualification.

## The actual encoder graph

Static inspection of the retained encoder graph finds 4,491 nodes. Its padding
expressions contain only small constants and shape operations; evaluating those
expressions loads no external weights and runs no model inference.

| Family | Nodes | ONNX pads (beginnings, then endings) | Mode / fill |
| --- | ---: | --- | --- |
| Attention | 24 | `[0,0,0,1,0,0,0,0]` | Constant / omitted, hence zero |
| Depthwise convolution | 24 | `[0,0,4,0,0,4]` | Constant / omitted, hence zero |

Both families pad only the last axis, with nonnegative widths. This census
establishes eligibility, not the individual nodes' runtime shapes or cost.

The graph also has 73 Where nodes. In 72, the true branch is a scalar constant;
48 select zero and 24 select -10,000. The remaining Where handles shape
construction. Lokad's general Where first
broadcasts all three operands, then calls GetValue/SetValue for each output.
BroadcastedTensor computes offsets using division and remainder per dimension.
This suggests a later scalar-selection path using contiguous runs, subject to
the actual condition layout. It must still support arbitrary views and preserve
selected NaN payloads and signed zero. No Where change is combined with padding.

## Source comparison with Microsoft ORT

Lokad's `CPUExecutionProvider.PadCore` materializes the source, fills a new
destination, and computes the multidimensional destination index separately for
every source element. Negative pads crop; reflection uses a distinct branch.
Normal dense sources already materialize without a copy. TensorSlice instead
inherits the generic coordinate-based materializer, so removing all copying is
neither necessary nor a valid ownership shortcut.

At pinned ORT commit `2e2543fbe9fae542f921d47a72d21d5a4ef0b710`,
[Pad](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/providers/cpu/tensor/pad.cc)
flattens eligible inner dimensions and copies innermost contiguous blocks.
This motivates a bounded Lokad path after the existing validation and fill:
require row-major materialized source, nonnegative last-axis padding and zero
padding on every outer axis; copy each source row into its destination slice.
Keep the existing crop, reflection and general-layout fallback unchanged.
Zero source width must return the already-filled destination without division.

ORT's pinned
[Where](https://github.com/microsoft/onnxruntime/blob/2e2543fbe9fae542f921d47a72d21d5a4ef0b710/onnxruntime/core/providers/cpu/tensor/where_op.cc)
uses broadcast spans, but also creates separate true/false selections and
merges them. This source alone establishes neither a one-pass implementation
nor which implementation an optimized installed graph dispatches. A new Lokad
path should preserve its own precise selection semantics rather than copy that
merge mechanism.

## Matrix blocking remains a separate hypothesis

The pinned MLAS header fixes packed column/reduction block sizes at 128/256.
Its SGEMM loop traverses a column slice, then ascending reduction blocks, then
row groups; later blocks accumulate into the destination. Lokad's two hot
AVX2 consumers traverse a 32-column panel, row group, then the full reduction.
A 4,096-term Lokad panel contains 512 KiB; a 256-term subpanel contains 32 KiB.
Smaller traversals could improve locality but add destination loads/stores.
Those byte counts do not establish actual cache misses or a performance gain.

The retained `tests/parakeet/wide-matmul/Blocked.cs` already prototypes ascending
128/256/512-term blocks with the same FMA chain and nonzero accumulator. Its
Windows comparisons failed control limits and candidate performance gates;
none is promoted. A future AMD experiment would need a separately qualified
complete-call protocol and actual candidate source, not a retry of those scores.
The separate partial-sum reduction experiment changes arithmetic and also failed
its AMD application gate; it is not the same mechanism.

The [complete observations](memory-source-observations-20260923.json) pin every
Pad/Where node, the graph, eleven source files and the existing profile report.
Source copies are retained in `artifacts/parakeet-memory-source-review-20260923`.
The external ORT checkout and all product files remain unchanged.
