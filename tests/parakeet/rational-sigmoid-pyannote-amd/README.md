# Pyannote correctness with the existing qualification consumer

Validate the fixed rational-sigmoid product, Core `946ddfb6` / Data `dbe95936`,
against current root `8bb22038` / `d02dbf55`. Parakeet model/application checks and
shared/e5 qualification have closed successfully. This lane introduces no product
variant and produces no performance score. The rejected operator screen and its
unresolved double latency remain part of the release evidence.

Reuse exact GraphQualification `d78c45b9`, already built and qualified for the
previous padding release. Its expected Core and Data identities are arguments,
and its guards remain intact. `consumer_reuse.py` binds the original qualified
product through the closed 3,979-method compatibility review to this pair, checks
every retained consumer/runtime file against its closed receipt, and preserves
all public bindings and old implementation flags. No product/consumer build,
package restore, model download or new compiled-consumer comparison is needed.
The original consumer comparison proved 95/96 methods unchanged and only the
specified Main identity-argument edits; its inspector did not record flags.
Reuse depends on exact binary identity, not a stronger retrospective claim.

Three jobs run serially: identity-probes, selected correctness, candidate
correctness. Four fresh deliberately wrong-Core/wrong-Data probes must reject
before output creation. Their PID/birth identities must all be terminal before
collection. Both real products then run all 18 arrays / 2,917,107 values and 16
complete public calls. Inputs and held outputs must remain unchanged.

All original native finite-output, shape, timeline, speaker, centroid and 1e-4
scaled numerical checks are unchanged. The arithmetic candidate is prospectively
allowed candidate/current float-array and centroid differences within that same
scaled bound. Record every numerical difference and bit-identity result. Public
intervals, speaker assignments, status, duration, window count and embedding state
remain exact. This replaces the padding-specific extra float bit-equality check;
it does not weaken native agreement or discrete public behavior.

Reuse the exact original native auditor, identity-probe tool, assets and fixtures.
The scope verifier preserves the worker's resource/ownership monitoring and all
original public semantics. Keep 11 GiB available RAM / 3 GiB tmpfs preflight,
8 GiB RSS, 900 seconds/job, 1 GiB remaining RAM/tmpfs, 1 GiB/job output and 2 GiB
campaign files. CPU 2 computes, CPU 0 monitors; only one VM workload runs at a time.

From the repository root, prefix these commands with
`C:/Python313/python.exe -X utf8 -B`:

    tests/parakeet/rational-sigmoid-pyannote-amd/consumer_scope.py
    tests/parakeet/rational-sigmoid-pyannote-amd/consumer_reuse.py
    -m unittest discover -s tests/parakeet/rational-sigmoid-pyannote-amd -p test_*.py
    tests/parakeet/rational-sigmoid-pyannote-amd/run.py prepare
    tests/parakeet/rational-sigmoid-pyannote-amd/run.py stage
    tests/parakeet/rational-sigmoid-pyannote-amd/run.py launch
    tests/parakeet/rational-sigmoid-pyannote-amd/run.py observe

Tools freeze at prepare. Observe the same supervisor until terminal, then collect
and audit.py once. Put audit stdout outside the artifact directory. Preserve any
failure; do not replay a completed campaign. Local namespace:
`artifacts/parakeet-rational-sigmoid-pyannote-amd-20260927`. Remote:
`/dev/shm/lokad-parakeet-rational-sigmoid-pyannote-20260927`.
Graph and complete Pyannote application regressions, then actual-root/package
qualification, remain necessary before integration or BENCHMARK.md promotion.
