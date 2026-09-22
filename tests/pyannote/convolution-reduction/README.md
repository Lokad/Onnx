# Pyannote three-row convolution reduction panels

This isolated prototype wraps only the existing full 32-column panel's row
loop in ascending blocks of 128 reduction terms. It uses the exact selected
product DLL; no root product file changes. Reductions at most 128, columns
below 32, short-reduction fallback and final two-row remainder keep the selected
route. The generated kernel's vector and narrow tail source stays identical.

All 22 observed convolution geometries and reduction/SIMD boundary cases are
checked against complete selected-kernel bits and an independent scalar oracle,
with offset buffers, guards, nonzero destinations and input preservation.
Timing includes clearing, packing and multiplication. Four fresh AMD processes
run baseline, candidate, candidate, baseline. Each conditions all shapes, then
warms each shape again before retaining six measured blocks.

Fixed screening gates: every same-role process ratio at most 1.10; geometric
mean candidate/baseline at most 0.95; every shape at most 1.05. Shapes have equal
weight for this component screen, not application frequency. The earlier e5
12/8-row AVX-512 blocking trial regressed and remains rejected. This different
three-row convolution hypothesis needs its own evidence.

From the repository root use `C:/Python313/python.exe -X utf8 -B` with:

    tests/pyannote/convolution-reduction/prepare.py
    tests/pyannote/convolution-reduction/transport.py stage
    tests/pyannote/convolution-reduction/transport.py launch
    tests/pyannote/convolution-reduction/transport.py observe
    tests/pyannote/convolution-reduction/transport.py collect
    tests/pyannote/convolution-reduction/audit.py

Commands require terminal successful dependencies and refuse existing output
destinations. Observe an existing owner; do not launch twice. Artifacts are
`artifacts/pyannote-convolution-reduction-20260922`, with the matching
`/dev/shm/lokad-pyannote-convolution-reduction-20260922` directory on the
exclusive AMD VM. Stage and collect pinned archives; never use its full root
disk. Process affinity is CPU 2 before CLR, monitor CPU 0; every observed native
thread is checked. Per-process limits: 900 seconds, 2 GiB RSS, 1 GiB available
and tmpfs free. Preflight requires 8 GiB available and 3 GiB tmpfs free; all
new artifacts are bounded by 1 GiB. Keep every failure and observation.

Even a passing screen permits only further full-model qualification. Accepted
application timings and the Microsoft ORT comparison remain unchanged.
