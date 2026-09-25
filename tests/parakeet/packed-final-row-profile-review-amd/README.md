# Correct the M78 observer review's zero-warning false positive

The observer build and collection succeeded. Its frozen reviewer then rejected
the two ordinary `0 Warning(s)` summaries because it searched for any line
containing `warning`. No compiler diagnostic or nonzero summary was present.
The rejection occurred before writing or transferring a build review and before
inference. Keep the original reviewer unchanged; do not rebuild the product.

`review.py` preserves the original source and generates a corrected copy under
`artifacts/parakeet-packed-final-row-profile-amd-20260925`. It permits exactly one
source change: exclude the exact stripped `0 Warning(s)` summary. It verifies
the actual rejected lines and checks that real warning text and nonzero summaries
still fail. All source, compiled-method, runtime, resource and transfer checks
remain unchanged. The review records the corrected reviewer's actual hash.

The correction has already run successfully. Do not rerun it or the original
review. Receipt `fc821b73` preserves the exact change, original source and logs.
Build review `1f6ff9bb` proves all 164 runner methods unchanged, 696 of 697 Data
methods unchanged, the one Execute hook exactly reversible, all 50 helper methods
equal to the certified observer, and the packed-weight constructor unchanged.
There are no warnings. Original Core `49901366` is retained; diagnostic Data is
`51b6d2bb`, with original control Data `01e9e784`.

The review was transferred before capture launched. Resume only the existing
capture through `tests/parakeet/packed-final-row-profile-amd/run.py observe capture`.
Once terminal, collect capture, audit and compare according to the main PLAN.md.
The frozen source-preparation and campaign tools remain unchanged. This correction
changes no model, inference loop, scoring limit or release verdict.
