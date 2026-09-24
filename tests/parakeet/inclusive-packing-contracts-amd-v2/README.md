# Focused inclusive-packing contracts on AMD

Run only after isolated build closure `50f2a3a8` verifies the single boundary
comparison. Build one small test assembly against the frozen product DLLs;
no product is rebuilt. The four test files are exact copies from source
`a535266a`: FoldBudgetTests, PackedBoundaryTests, PackedWeightsTests and
PackedWeightBudgetTests. Their 61 cases cover admission, aggregate caps,
invalidation, source ownership, fallback, row/column tails and independent
matrix products. One additional case verifies the actually loaded product
hashes and directory, CPU2, .NET 10.0.8 and instruction availability.

Six jobs: SDK, offline test restore, test build, selected-release negative
control, candidate normal tests and candidate tests with AVX-512 disabled.
The control runs the new Reduction4096_IsAdmitted test and the identity case:
exactly one expected assertion failure (524288 retained bytes versus zero),
one pass, exit 1. Both candidate modes must pass all 62 cases without skips.
Every selected/candidate output file is immutable after test build. All runtime
product files and the actual loaded identities are checked independently.

The existing build controller retains its resource limits: 12 GiB available /
3 GiB tmpfs before each job, 8 GiB owned RSS, 1 GiB remaining memory/tmpfs,
900 seconds per job, four hours total, 1 GiB output/job and 2 GiB artifacts.
CPU2 compute and CPU0 monitoring; no Windows build or inference. One expected
test-process exit 1 is required, never mistaken for a passing product test.

Freeze tools, then use `C:/Python313/python.exe -X utf8 -B` from the repo root:

    tests/parakeet/inclusive-packing-contracts-amd-v2/run.py prepare
    tests/parakeet/inclusive-packing-contracts-amd-v2/run.py stage
    tests/parakeet/inclusive-packing-contracts-amd-v2/run.py launch
    tests/parakeet/inclusive-packing-contracts-amd-v2/run.py observe
    tests/parakeet/inclusive-packing-contracts-amd-v2/run.py collect
    tests/parakeet/inclusive-packing-contracts-amd-v2/audit.py

Collect only after all exact PID/birth owners are terminal. Local output:
`artifacts/parakeet-inclusive-packing-contracts-amd-v2-20260924`; VM:
`/dev/shm/lokad-parakeet-inclusive-packing-contracts-v2-20260924`. Refuse existing
namespaces and retain failures. This establishes focused correctness only;
actual model residency, native/public correctness and timing remain separate.

The first namespace closed at `17376d23` before restore or tests because its
isolated source omitted `global.json` and selected an installed preview SDK.
This version adds the unchanged selected-source `global.json`. Test sources,
expected failure, loaded-product checks, counts, numerical assertions, commands
and resource limits are unchanged except for namespace paths. The original
failure and all owner records remain retained.
