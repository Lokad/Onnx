# M54 actual root, full suites and NuGet qualification

First require the admitted complete Parakeet, Pyannote and eight-case graph
comparisons. `integrate.py` verifies all 420 selected root inputs, preserves
TensorOps.MatMul.cs and applies exactly the three-file admitted change, adding
Zzz.IsolatedShortMatMul.cs and Zzz.WideProjectionEntry.cs. Both additions must
be absent before integration. All 422 resulting inputs must match the measured
isolated snapshot. A partial
integration is retained in `intended.json`; inspect it before resuming or
restoring the one saved file and removing only the two verified additions.

Run a normal SDK 10.0.204 restore/build on AMD, then compare all 3,189 Core and
697 Data method bodies, implementation flags and public interfaces against the
measured candidate. No arbitrary normalization or changed method is permitted.

Run the complete backend and tensor suites in ordinary and AVX512-disabled
modes. The ordinary census must exactly match selected root evidence: 3,449
backend passes/41 existing skips and 343 tensor passes. Source inspection fixes
the disabled census prospectively: 83 AVX512-only cases become skipped and
three unsupported-hardware checks become active, yielding 3,369 backend passes
and 121 skips. Every other test outcome remains identical; tensors stay 343/0.
The disabled set is the packed 12-row case, 44 narrow-row cases, 36 panel
composition cases and two six-row AVX512 cases. No new skip is accepted.

Pack the normal core and restore/build/run the existing independent consumer
through actual `PackageReference`. Verify package DLL identity, the sole
Google.Protobuf 3.33.5 dependency, model import, prepared convolution, Winograd,
owned results and input preservation. No timing is inferred from this lane.

Use `C:/Python313/python.exe -X utf8 -B` with `integrate.py`, then `run.py
prepare`, `stage`, `launch`, `observe`, `collect`, then `audit.py`. Freeze tools
and actual source before transfer. CPU 2 computes and CPU 0 monitors; preflight
10 GiB available/3 GiB tmpfs, 8 GiB RSS and 900 seconds per job, four-hour
campaign bound. Preserve failures and do not relax any gate. Update release
benchmarks only after this lane closes successfully.

The graph prerequisite is the M54 comparison using the qualified warmed consumer: 600 fixed
warmups and 180 measurements per timing process, with unchanged 1.10 stability
and 1.05 regression limits. Require its exact frozen payload and admitted
closure. The earlier M43/M45 candidate remains rejected and cannot admit this
integration. No root source change occurs merely by preparing these tools.

Before integration, run `source_scope.py` and `consumer_scope.py` to verify the
source snapshot and retained qualification checks without changing the product.
Run `python -B -m unittest discover -s tests/parakeet/wide-entry-first-use-root-amd
-p test_guards.py -v` for eight tests that reject unrelated source changes,
missing additions, and substituted selected/candidate binaries in either audio
comparison or the graph campaign.
