# Reuse the retained observer after transpose dispatch

This read-only review reconciles the retained observer through qualified root
ee4a38ff and the single measured transpose change c471f5d1. All 697 Data method
bodies and flags remain identical; both assemblies' public declarations and
attributes are preserved. Candidate Data is byte-identical to current root Data.
Keep consumer 38ab5c7e and observed Data 51b6d2bb. After release qualification,
use the actual built Core in every managed capture mode, actual Data for control
and the retained observed Data for phase and wall.

Read retained compiled inventories and candidate binaries; build nothing and
execute no inference. Require exactly one changed Core method, TransposeInto,
with arithmetic leaves unchanged. Original observer proofs and failures keep
their verdicts. This review does not qualify new performance or admit release.

From repository root prefix C:/Python313/python.exe -X utf8 -B and run
tests/parakeet/transpose-axis-profile-review/review.py --publish once.
Subsequent read-only verification omits --publish. Evidence is under
artifacts/parakeet-transpose-axis-profile-review-20260928.

Fresh profiling still requires actual graph, complete Pyannote application and
root/package admission. Combine complete new managed/native attribution over
the same twenty clips, then inspect pinned ORT source for the largest remaining
measured excess before selecting one causal intervention. This review selects
no further optimization.
