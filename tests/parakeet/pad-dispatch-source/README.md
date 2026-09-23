# Isolated padding dispatch candidate

M47 starts from qualified selected source 94a550de, with all 420 parent inputs
verified. M46's complete-call screen failed; none of its product code is a parent.

Public Pad changes only four private call targets. A new private PadDispatch
selects eligible constant, nonnegative last-axis padding and copies whole rows.
It delegates reflection, reversed input, cropping and outer padding to the
entirely unchanged original PadCore. Eligibility is decided before materializing
input, so fallback does not pay a duplicate materialization. Normal dense,
sliced and broadcast inputs retain the existing ToDenseTensor behavior; the
copy uses the same flat buffer and original data dimensions as PadCore.

The private helper allocates/fills an owned destination and preserves zero
dimensions and zero-padding ownership. No arithmetic, matrix, public API or
compiler-flag change is introduced. The six targeted tests are exact copies
from M46. Source preparation alone establishes no measured speedup.

Use `C:/Python313/python.exe -X utf8 -B prepare.py` from this directory, or its
repository-relative path from the root. It refuses an existing output and writes
an isolated 422-file snapshot under artifacts/parakeet-pad-dispatch-source-20260923.
Then qualify the new build and unchanged complete-call workload before a model trial.
