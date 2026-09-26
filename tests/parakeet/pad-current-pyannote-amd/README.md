# Pyannote correctness for the current-root padding candidate

This checks the existing copy-padding candidate, Core a74acb17 / Data be954dc4,
against the qualified root f3992f40 / a8e0b583. Full Parakeet correctness,
complete application admission and shared/e5 correctness must already pass.
No product, native reference, numerical requirement or timing protocol changes.

Reuse the original Pyannote assets and graph references. Both fresh products
must pass all 18 arrays / 2,917,107 values and 16 complete public calls, native
scaled error <= 1e-4, unchanged inputs and held outputs. Candidate tensors and
complete public results, including centroids, must equal current root exactly.
This campaign produces no performance score.

Build one common GraphQualification consumer on the VM. The reviewed two-line
source change accepts an expected Data hash alongside the existing expected Core
hash. Both actual assembly hashes are still checked. Compare all 96 compiled
methods to the original consumer: 95 unchanged, and Main exactly the specified
argument count, usage and Data operand transformation, including adjusted branch
and exception offsets. The inspector does not record implementation flags.

Before inference, run four negative probes: wrong Core and wrong Data for each
product. Each must fail at the corresponding identity guard before creating an
output directory. Keep command, PID/birth, exit status and stdout/stderr. Core
dumps are disabled only for these expected failures. Collection independently
checks all four child identities are terminal, including short-lived children
that might finish between supervisor samples. The consumer runs unchanged for
both actual numerical jobs, with both expected identities from the frozen payload.

Six jobs run serially: restore, build, compiled inventory, identity probes,
selected correctness, candidate correctness. Reuse the original monitor and
checks. Keep 11 GiB available / 3 GiB tmpfs preflight, 8 GiB owned RSS, 900 seconds
per job, 1 GiB remaining RAM/tmpfs, 1 GiB per-job output, 2 GiB campaign files and
four hours overall. Compute on CPU 2, monitor on CPU 0. No Windows .NET execution
or model downloads. Existing assets are hardlinked; product DLLs are never built.

From the repository root, use `C:/Python313/python.exe -X utf8 -B` with:

    tests/parakeet/pad-current-pyannote-amd/consumer_scope.py
    -m unittest discover -s tests/parakeet/pad-current-pyannote-amd -p test_*.py
    -m unittest discover -s tests/pyannote/parameterized-qualification-consumer -p test_*.py
    tests/parakeet/pad-current-pyannote-amd/run.py prepare
    tests/parakeet/pad-current-pyannote-amd/run.py stage
    tests/parakeet/pad-current-pyannote-amd/run.py launch
    tests/parakeet/pad-current-pyannote-amd/run.py observe
    tests/parakeet/pad-current-pyannote-amd/run.py collect
    tests/parakeet/pad-current-pyannote-amd/audit.py

Prepare, stage, launch, collect and audit only once; observe while active. Keep
audit stdout outside the artifact folder. Preserve failure and all evidence;
never weaken the frozen checks or retry unchanged. Local namespace:
`artifacts/parakeet-pad-current-pyannote-amd-20260926`. Remote namespace:
`/dev/shm/lokad-parakeet-pad-current-pyannote-20260926`.

This correctness pass must precede fresh Pyannote application timing, graph
regressions and actual root/package qualification. No release promotion follows
from building the consumer or preparing these tools.
