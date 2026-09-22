# Actual-shape AMD qualification

Execute the exact Windows-qualified consumer in separate AVX2 and AVX-512
processes, each covering 108 calls and 119,823,360 values. Require exact selected
production and Windows observations, native endpoints within `1e-4`, immutable
operands and owned repeated outputs. This is not a speed measurement.

Use `C:/Python313/python.exe -X utf8 -B` from root:

    tests/pyannote/blocked-spatial-model-amd/run.py prepare
    tests/pyannote/blocked-spatial-model-amd/run.py stage
    tests/pyannote/blocked-spatial-model-amd/run.py launch
    tests/pyannote/blocked-spatial-model-amd/run.py observe
    tests/pyannote/blocked-spatial-model-amd/run.py collect
    tests/pyannote/blocked-spatial-model-amd/audit.py

Local artifact: `artifacts/pyannote-blocked-spatial-model-amd-20260922`.
Remote: `/dev/shm/lokad-pyannote-blocked-spatial-model-20260922`.
Refuse existing output, pin transferred fixtures and binaries, retain every
resource observation, and collect only after actual terminal ownership.
Preserve CPU2 initialization, monitor CPU0, 12 GiB available/3 GiB tmpfs preflight,
8 GiB RSS, 1 GiB available/tmpfs/output, 2 GiB artifacts and 900-second workers.
No product changes, new dependencies or performance candidate selection.
