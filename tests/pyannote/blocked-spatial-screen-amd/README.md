# Fixed AMD screen for complete spatial convolution calls

First qualify the new screen executable on all 108 actual model calls. Then run
four fresh processes, in order: production, candidate, candidate, production.
The candidate always uses sixteen-channel AVX-512. Each process executes one
complete warmup pass and three measured passes in frozen crop/node order.
Retain all 1,728 calls, including 432 warmups and every fallback.

Timers include ordinary NCHW input to owned NCHW output, operand scans, scratch
rental, clearing, direct spatial reduction, conversion, bias, residual and ReLU.
Production returns its ordinary owned tensor without an added copy. Hash checks
and journal writes occur outside the timer. Both roles load identical fixtures
and retain 32 prepared weights (21,086,208 bytes) outside hot-call timers.
Preparation is separately measured in every process, one warmup plus three
measurements of every constant. This common harness residency is not a claim
that production normally needs prepared spatial weights. Future application
qualification must charge candidate preparation per model/context construction.

Use exact integer-clock fractions. Sum each of the 108 calls once per measured
pass, preserving original call frequencies across three crops. Require each
role's two-process max/min at most 1.10 for the aggregate and 1.20 for every form.
Candidate/production must be at most 0.90 overall and 1.05 for each eligible
form. Include all four fallback convolutions per crop in the aggregate. Retain
every failed gate; no unchanged retry. Admission permits product qualification,
not a whole-application speed claim or an ORT comparison.

From root, with `C:/Python313/python.exe -X utf8 -B`:

    -m unittest discover -s tests/pyannote/blocked-spatial-screen-amd -p test_score.py
    tests/pyannote/blocked-spatial-screen-amd/run.py prepare
    tests/pyannote/blocked-spatial-screen-amd/run.py stage
    tests/pyannote/blocked-spatial-screen-amd/run.py launch
    tests/pyannote/blocked-spatial-screen-amd/run.py observe
    tests/pyannote/blocked-spatial-screen-amd/run.py collect
    tests/pyannote/blocked-spatial-screen-amd/audit.py

Local artifact: `artifacts/pyannote-blocked-spatial-screen-amd-20260922`.
Remote: `/dev/shm/lokad-pyannote-blocked-spatial-screen-20260922`.
Before staging require closed actual-model AMD qualification and closed Windows
screen-executable qualification. Refuse existing output and freeze all inputs.
CPU2 before CLR; monitor CPU0. Keep 12 GiB available/3 GiB tmpfs preflight,
8 GiB RSS, 1 GiB available/tmpfs/output, 2 GiB artifact and 900-second worker
bounds, with a four-hour campaign limit. Collect only actual terminal owners.
