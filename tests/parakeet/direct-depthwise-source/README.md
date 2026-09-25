# One direct nine-tap depthwise candidate

The measured M78 baseline writes expanded patches and makes millions of tiny
matrix calls for Parakeet depthwise convolution. This candidate keeps the dense
layout and writes direct results into the original independently allocated output.
Only `Conv2DFloatCore` gains a dispatch call; four private helpers are added.
No shared matrix, packing, graph, output allocation or compiler settings change.

The route requires batch one, one channel per group, equal input/output channels,
at least two channels, degree one, SIMD/intrinsics and AVX2/FMA. It covers the
observed 1×9 stride-one no-pad and 3×3 stride-two pad-one geometries, both with
unit dilation. Explicit segmented selection keeps priority. All other cases
retain the existing code. The 1D loop uses contiguous vectors; the spatial loop
gathers stride-two interior positions and handles borders directly.

Tap order and sum-then-bias stay unchanged. Vector positions use FMA; the original
final flattened remainder uses multiply/add. Spatial row boundaries do not alter
that choice. Padding contributes actual zero arithmetic, including NaN behavior.

Eight focused tests compare the candidate with the exact original assembly
loaded separately: every observed geometry, borders/tails, memory offsets, special
values, unsupported options/geometries, logical layouts/ownership, provider 1D
and pooled spatial output, and runtime/product identity. Run both normal hardware
and hardware-disabled modes. A full public mechanism observation, native model
qualification and performance comparison remain necessary afterward.

From the repository root, before the one-time preparation:

    C:/Python313/python.exe -X utf8 -B -m unittest discover -s tests/parakeet/direct-depthwise-source -v
    C:/Python313/python.exe -X utf8 -B tests/parakeet/direct-depthwise-source/prepare.py

Root product code stays unchanged. The isolated M78 e5 regression still prevents
release promotion independently of this candidate.
