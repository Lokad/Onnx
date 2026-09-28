# Reuse the existing transpose tile for two diagnosed layouts

The actual encoder and fresh profiles identify 96 nodes with 0.858205 seconds
of excess time. Pinned ORT source reduces both permutations to batched matrix
transpose and uses contiguous tiles. Lokad currently uses generic indexing or
strided vector construction. See ../attention-owned-profile-results/
transpose-diagnosis-20260928.md for exact shapes, source and limitations.

Change only Tensor<T>.TransposeInto to reuse the existing 8-by-8 shuffle for
float rank-three [0,2,1] and rank-four [0,2,3,1], with a full tile in both
collapsed axes. Preserve small-face, nonfloat, disabled-vector and missing-AVX
paths, other permutations, alias validation, input immutability and output
ownership. Empty destinations return before flattening dimensions.

The six new portable facts cover all 76 observed family/geometric combinations,
special float bits, partial tiles, empty axes (including a large zero-element
suffix), batches, source views, destination guards/aliasing, ownership and
fallbacks. Existing transpose and source-policy contracts remain required.

Freeze these tools, then from the repository root run once:

    C:/Python313/python.exe -X utf8 -B tests/parakeet/transpose-axis-source/prepare.py

The isolated snapshot binds all 446 current qualified source files to root
ee4a38ff, adds one fixture, and changes no root product file. Build and test on
the exclusive AMD VM. Predict at least 0.45 seconds saved per twenty-clip corpus;
require the original independent application comparison (>=1% gain, <=5%
per-clip regression, all 63 controls/21 gates) before release qualification.
No arithmetic kernel, flag or tile-size sweep is part of this candidate.
