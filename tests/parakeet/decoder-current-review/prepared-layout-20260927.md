# The existing Lokad projection buffer uses 32-column panels

This source review specifies the buffer a possible one-row consumer would read.
It does not establish that the current decoder has this mapping or executes a
particular kernel. The [bounded observation](next-observation-20260927.md) still
decides whether an implementation is justified.

`GraphPacking.PackMatMulWeights` calls `MathOps.PackPanelsB` for eligible constant
weights. For B `[640,8198]`, that routine produces **256 panels of 32 columns**,
followed by a six-column tail. A full panel occupies 81,920 bytes. The entire
buffer occupies 20,986,880 bytes: exactly the original element count, with no
padding to a multiple of the vector width.

For reduction index `j` and output column `c`, offsets below are in floats:

| Storage | Offset |
|---|---|
| Original row-major B | `j * 8198 + c` |
| Prepared B, `c < 8192`, with `b = 32 * floor(c / 32)` | `b * 640 + j * 32 + (c - b)` |
| Prepared six-column tail, `c >= 8192` | `8192 * 640 + j * 6 + (c - 8192)` |

The existing `mm_m1_kblocked` holds four 256-bit accumulators for 32 outputs,
visits reduction indices in ascending order, and stores after the reduction.
All six remaining columns use scalar multiply/add in ascending reduction order.
A layout-only experiment can retain those choices and replace B addressing with
the offsets above, provided the ordinary mapping and operand guards succeed.
There is no need to change the preparation format to make that comparison.

ORT's [observed one-row loop](native-row1-20260927.md) instead consumes two
sixteen-column streams with two 512-bit accumulators and reduction unrolling.
Its stream convention is not a specification for Lokad's existing buffer.
Importing that reader would combine layout, vector width and loop changes.
Likewise, the [existing final-row helper](packed-row-caveat-20260927.md) must not
be assumed to preserve the current accumulator behavior.

Source at qualified root `5a216b09`: [preparation](../../../src/Lokad.Onnx/GraphPacking.cs),
[packing and one-row accumulation](../../../src/Lokad.Onnx/MathOps.cs), and
[dispatch](../../../src/Lokad.Onnx/TensorOps.MatMul.cs). Those files remain
unchanged in the fixed rational-sigmoid candidate. No implementation, compilation,
inference, timing, or additional candidate follows from this review.
