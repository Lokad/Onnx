# Inclusive packing isolated AMD build

Build source `a535266a` from selected release `81f75c38`, without changing root
product files. The complete 22-instruction `GraphPacking.FitsPackBudget` body
must differ only at offset 14: signed `bge.s` becomes `bgt.s`, with the same
encoded branch destination. This admits reduction length 4096. Every other
instruction, header, local, exception region, method flag and public interface
must match; all other 3,188 Core and 697 Data methods remain exact.

The unchanged four-job controller checks SDK 10.0.204, restores offline, builds
the Release CLI and inventories both assemblies on AMD. Compute runs on CPU2,
monitoring on CPU0. All build commands use `--tl:off --nologo -v minimal`.
Bounds: 12 GiB available / 3 GiB tmpfs preflight, 8 GiB owned RSS, 1 GiB remaining
memory/tmpfs, 900 seconds per job, four hours per campaign, 1 GiB output per job,
2 GiB total artifacts. No Windows build or inference.

Freeze tools, then run from the repository root with
`C:/Python313/python.exe -X utf8 -B`:

    tests/parakeet/inclusive-packing-build-amd/test_checks.py
    tests/parakeet/inclusive-packing-build-amd/run.py prepare
    tests/parakeet/inclusive-packing-build-amd/run.py stage
    tests/parakeet/inclusive-packing-build-amd/run.py launch
    tests/parakeet/inclusive-packing-build-amd/run.py observe
    tests/parakeet/inclusive-packing-build-amd/run.py collect
    tests/parakeet/inclusive-packing-build-amd/audit.py

Observe until the exact PID/birth owner is terminal before collection. Output:
`artifacts/parakeet-inclusive-packing-build-amd-20260924`; VM:
`/dev/shm/lokad-parakeet-inclusive-packing-build-20260924`. Existing namespaces
are refused. Preserve failures and do not alter frozen inputs.

This qualifies build scope only. Boundary and ownership tests, actual bounded
residency, complete model/native/public correctness and application performance
are separate mandatory gates in `.agent/m63-parakeet-inclusive-packing-20260924.md`.
