# M22 normal Linux product and package qualification

This phase follows the admitted 13.72% complete-LSTM screen. It changes no
candidate arithmetic and makes no new timing claim. All work uses the authorized
AMD VM, worker CPU2 before CLR startup and monitor CPU0.

From repository root, run `C:/Python313/python.exe -X utf8 -B` with
`selftest.py`, then `run.py prepare`, `run.py stage`, `run.py launch`.
Observe the recorded owner with `run.py observe`; after all owners are terminal,
use `run.py collect` and `audit.py`. Existing destinations are refused.

The source is the exact locally qualified M22 candidate. Its initial archive
did not include tensor tests or the solution marker used by test root discovery,
so those unchanged files are separately archived from base commit `9533cd67`.
Every original source hash and the supplemental
archive are frozen. The measured Core `208371f6` / Data `b9358370` remain intact.
An independently qualified IL inspector checks all 3,163 Core and 697 Data
method bodies and public declarations against the normal Linux build.

Fourteen fixed jobs verify SDK 10.0.204; restore/build CLI, backend and tensor
projects; inspect compiled methods; run full backend/tensor suites; pack the
actual normal Core; and restore/build/run an independent PackageReference
consumer. The feed is the existing pinned offline feed. Expect exactly 3,432
backend passes and 41 existing AMD skips, and 343 tensor passes. Reconcile the
entire prior Linux census plus 36 new LSTM cases, including the mandatory
AVX512 independent-oracle test. Missing tests cannot be masked by totals.

The package must contain the exact rebuilt Core and only Google.Protobuf
3.33.5 as a dependency. The unchanged package consumer checks public operators,
MNIST import, input/held ownership and two prepared ConvRelu requests with
1,056 values each, 18,432 retained weight bytes and 8,384 scratch bytes.

Keep preflight 12 GiB available / 3 GiB tmpfs, 8 GiB owned RSS, 1 GiB minimum
available memory/tmpfs, 900 seconds per worker and 2 GiB artifacts. Builds use
isolated cache directories and `--tl:off`. No local build/inference is permitted
while local disk remains below its 20 GiB threshold. All logs, resource samples,
test results, package, runtime and method inventory are retained for audit.

`release_transfers.py` separately reclaimed only two duplicate upload archives
after verifying complete local copies, closed evidence, terminal owners and no
external dependency references. It left all extracted inputs/results intact.
Its receipt is `artifacts/pyannote-lstm-product-space-20260922/receipt.json`.
