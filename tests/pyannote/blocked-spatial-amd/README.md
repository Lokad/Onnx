# AMD numerical qualification of direct spatial convolution

Run the exact closed v2 consumer in separate AVX2 and AVX-512 processes. Each
must pass 2,648 original cases, twenty isolated non-finite/overflow cases and
ten invalid/alias checks. Require exact finite scalar/public results, explicit
non-finite fallback, guards, read-only operands and owned outputs. Both AMD
reports must match every Windows observation, with the expected instruction
width and .NET 10.0.8 identities. This is not a performance comparison.

From repository root use `C:/Python313/python.exe -X utf8 -B`:

    tests/pyannote/blocked-spatial-amd/run.py prepare
    tests/pyannote/blocked-spatial-amd/run.py stage
    tests/pyannote/blocked-spatial-amd/run.py launch
    tests/pyannote/blocked-spatial-amd/run.py observe
    tests/pyannote/blocked-spatial-amd/run.py collect
    tests/pyannote/blocked-spatial-amd/audit.py

Freeze source, binaries and existing interpreter/runtime/psutil files. CPU2
before CLR; monitor CPU0. Keep 12 GiB available / 3 GiB tmpfs preflight, 8 GiB
RSS, 1 GiB available/tmpfs/output floors/limit, 2 GiB artifacts and 900-second
workers. Preserve every resource observation. Verify actual terminal owners
before collection; never relaunch after a timeout. Existing outputs are refused.

Artifacts: `artifacts/pyannote-blocked-spatial-amd-20260922` and remote
`/dev/shm/lokad-pyannote-blocked-spatial-20260922`. No product dispatch changes,
runtime overrides, new dependencies, push or package publication.
