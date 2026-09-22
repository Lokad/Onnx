# M22 complete Pyannote qualification on AMD

This follows the admitted complete-LSTM screen and closed normal Linux
build/suite/package qualification. It runs the exact measured selected
Core `3c2f16b0` / Data `6318cf48` and candidate Core `208371f6` / Data `b9358370`.
It makes no new application performance claim.

From repository root use `C:/Python313/python.exe -X utf8 -B` with
`selftest.py`, then `run.py prepare`, `run.py stage`, `run.py launch`.
Observe the recorded owner using `run.py observe`; only after terminal state,
run `run.py collect` and `audit.py`. Existing destinations are refused.

Five fixed workers restore/build the candidate consumer, inspect its methods,
then run selected and candidate full Pyannote qualification. The consumer source
changes exactly one expected Data identity literal. Require 95 of 96 methods
identical, and only that literal in Main changed, before model workers start.
The selected consumer and both product binaries remain unchanged.

Each model worker checks all 18 segmentation/embedding output arrays,
2,917,107 values and 16 public diarization requests. Both products retain the
native scaled-error limit 1e-4, complete speaker timelines and centroid bounds,
exact repeats, readonly inputs and retained outputs. Every candidate graph
output and complete public result must match fresh selected AMD execution
exactly. Numerical references are the existing pinned native arrays; fresh
native execution remains a later gate. Original node profiles and every
diagnostic clock are retained, without becoming a performance score.

Assets and graph references come from the verified closed M17 payload. Derived
manifests change only Core/Data identities and the stale descriptive product
label; all model/audio hashes, cases, expected outputs and reference fields
remain exact. Models already present on the VM are used by their pinned absolute
paths. The complete preceding independent graph/public auditor is reused byte
for byte, with additional exact managed and public-result comparisons.

Workers inherit CPU2 before CLR startup; the monitor runs on CPU0. No overrides
are allowed in numerical workers. SDK/runtime dependencies and all source/assets
are frozen before launch. Preserve 12 GiB available / 3 GiB tmpfs preflight,
8 GiB owned RSS, 900 seconds per worker, 1 GiB minimum available memory/tmpfs and
output, and 2 GiB artifacts. The local machine does no build or inference.
