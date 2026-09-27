# Reuse prepared storage for the 92 uncached attention weights

The measured current cost is 0.817785 seconds of repeated packing per corpus.
Extend only the opt-in GraphOwnedPacking eligibility policy: exact square
attention name/shape pairs join the existing feed-forward pairs. Preserve all
single-use, alias, visibility, captured-graph and existing-cache exclusions.
The expected actual-model delta is exactly 92 owned weights, with the existing
37 cache entries and 256 MiB budget unchanged. No arithmetic method changes.

This source snapshot is a candidate, not a release. Bind all 445 original inputs
to qualified root fc116763 and diagnosis 5ff17ed5. Reversible source edits change
only PrepareOwnedMatMulWeights and its documentation; add one portable fixture.
The fixture covers five families, paired name/shape exclusions, seven sharing
guards, unchanged cached weights, all 38 T/2T-1 corpus geometries, logical bits,
held input/output ownership, alias rejection, collection of old arrays and
unsupported-execution fallbacks. Existing feed-forward tests remain unchanged.

From repository root, after freezing these tools, run once:

    C:/Python313/python.exe -X utf8 -B tests/parakeet/attention-owned-source/prepare.py

Inspect candidate.patch under artifacts/parakeet-attention-owned-source-20260928.
Build and run numerical contracts on the exclusive AMD VM, in normal and
disabled-AVX512 modes plus focused disabled-intrinsics checks. First qualify the
actual compiled scope; then verify the exact model census and complete original
public results before the independent application comparison. At least 1% gain,
all original repeatability controls and no clip regression above 5% are required.
The roughly 0.82-second opportunity is a prediction, not a measured improvement.

Do not overwrite the snapshot or modify the root product before admission.
Preserve failures and investigate retained evidence before recovery. Global
packing-budget and kernel/flag sweeps remain closed.
