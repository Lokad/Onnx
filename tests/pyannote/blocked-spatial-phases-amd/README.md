# AMD complete phase diagnostic

Qualify the exact instrumented executable on all 108 actual model calls, then
run two fresh AVX-512 diagnostic processes. Each keeps one full warmup and three
measured passes in frozen crop/node order. Reconcile all 864 calls, including
768 eligible and 96 fallback calls, all output hashes and every inclusive clock.

These clocks are instrumented attribution. They are not an application timer,
a new comparison against ORT, or a repeat of the closed failed speed screen.
The original arithmetic kernel and model qualifier remain unchanged; clocks can
still affect generated code, so actual numerical qualification is required.

From root with `C:/Python313/python.exe -X utf8 -B`:

    tests/pyannote/blocked-spatial-phases-amd/run.py prepare
    tests/pyannote/blocked-spatial-phases-amd/run.py stage
    tests/pyannote/blocked-spatial-phases-amd/run.py launch
    tests/pyannote/blocked-spatial-phases-amd/run.py observe
    tests/pyannote/blocked-spatial-phases-amd/run.py collect
    tests/pyannote/blocked-spatial-phases-amd/audit.py

Artifact: `artifacts/pyannote-blocked-spatial-phases-amd-20260922`.
Remote: `/dev/shm/lokad-pyannote-blocked-spatial-phases-20260922`.
Refuse existing outputs, freeze sources/binaries/fixtures and require all eight
audit tests before packaging. CPU2 before CLR, monitor CPU0, 12 GiB available /
3 GiB tmpfs preflight, 8 GiB RSS, 1 GiB available/tmpfs/output, 2 GiB artifacts,
900 seconds per worker and four-hour campaign limit. Collect only terminal owners.
Preserve temporary account lingering and service masks through the authorized VM
window; restore their recorded prior states after the whole window ends.
