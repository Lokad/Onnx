# Actual Parakeet packing residency at unchanged budgets

After build `50f2a3a8` and focused contracts `e85e5f5f`, load the actual encoder
at 256 MiB and decoder at 64 MiB with selected and candidate products, in both
ordinary and AVX-512-disabled modes. Eight fresh processes use existing model
paths; no new model or tensor copies are exported. This is load-only evidence,
without inference or application timing.

The consumer adapts `packing-residency/Program.cs`: keep the actual graph mapping
and accounting checks, restrict budgets to 256/64 MiB, hash every retained source
and clone, verify independent backing storage, invalidate and rebuild, and require
identical mapping/content afterward. Report every name, source key, shape, byte
count and digest. Compare both instruction modes and every shared weight across
products. Do not prescribe old counts or infer that every retained weight is used.

Eleven jobs: stable SDK check, offline consumer restore/build and eight loads.
The unchanged selected-source `global.json` is included. No product rebuild.
Use CPU2 for all compute, CPU0 for monitoring. Bounds: 12 GiB available / 3 GiB
tmpfs preflight, 8 GiB owned RSS for build jobs / 12 GiB for loads, 1 GiB minimum
remaining memory/tmpfs, 900 seconds/job, four hours total, 1 GiB output/job,
2 GiB artifacts. No Windows build or inference. All .NET build flags include
`--tl:off --nologo -v minimal`.

Freeze tools, then run with `C:/Python313/python.exe -X utf8 -B`:

    tests/parakeet/inclusive-packing-residency-amd/run.py prepare
    tests/parakeet/inclusive-packing-residency-amd/run.py stage
    tests/parakeet/inclusive-packing-residency-amd/run.py launch
    tests/parakeet/inclusive-packing-residency-amd/run.py observe
    tests/parakeet/inclusive-packing-residency-amd/run.py collect
    tests/parakeet/inclusive-packing-residency-amd/audit.py

Refuse existing output `artifacts/parakeet-inclusive-packing-residency-amd-20260924`
and VM `/dev/shm/lokad-parakeet-inclusive-packing-residency-20260924`. Collect only
terminal PID/birth owners; preserve all failed results and guards. Complete native
tensors, public requests and application performance remain later gates.
