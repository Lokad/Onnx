# Padding dispatch with first-use optimized fallback

M49 starts from qualified selected source 94a550de, with all 420 parent inputs
verified. M47's complete-call screen failed. M48 traces identify different
fallback compilation states; the selected release remains the parent.

Public Pad changes only four private call targets. A new private PadDispatch
selects eligible constant, nonnegative last-axis padding and copies whole rows.
It delegates reflection, reversed input, cropping and outer padding to the
original PadCore body. Eligibility is decided before materializing
input, so fallback does not pay a duplicate materialization. Normal dense,
sliced and broadcast inputs retain the existing ToDenseTensor behavior; the
copy uses the same flat buffer and original data dimensions as PadCore.

The private helper allocates/fills an owned destination and preserves zero
dimensions and zero-padding ownership. Original PadCore receives exactly one
AggressiveOptimization annotation (implementation flag 0 to 512). Its body,
locals and branches stay exact. All other method flags, arithmetic, matrix code
and public APIs remain selected. The dispatcher and six tests are byte-identical
to M47. Losing later profile-based optimization is a tradeoff to qualify.
Source preparation alone establishes no measured speedup.

Use `C:/Python313/python.exe -X utf8 -B prepare.py` from this directory, or its
repository-relative path from the root. It refuses an existing output and writes
an isolated 422-file snapshot under artifacts/parakeet-pad-first-use-source-20260923.
Then qualify the new build and unchanged complete-call workload before a model trial.
