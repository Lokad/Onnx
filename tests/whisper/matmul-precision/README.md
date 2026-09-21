# Whisper matrix-reduction precision diagnostic

This local, unfused NumPy/SciPy graph experiment changes only MatMul accumulation
from float32 to float64. It rounds each matrix result back to float32, preserving
all original operator boundaries. It asks whether matrix reduction error is a
useful target for a later managed-kernel prototype. It is not a product variant,
a latency benchmark or a new numerical acceptance contract.

The fixed three previously selected natural recordings plus a repeat use managed
features. Both variants retain all 41 encoder boundaries and compare every value
with both previously qualified float64 reference routes. Original native and
managed final comparisons are retained separately. Selection and numerical
limits are frozen before inference; results never trigger extra requests.

Run from the repository root with Python 3.13 and the already installed numerical
packages referenced by `tests/whisper/full-reference/common.py`:

    C:/Python313/python.exe -X utf8 -B tests/whisper/matmul-precision/test_precision.py
    C:/Python313/python.exe -X utf8 -B tests/whisper/matmul-precision/probe.py prepare
    C:/Python313/python.exe -X utf8 -B tests/whisper/matmul-precision/run.py
    C:/Python313/python.exe -X utf8 -B tests/whisper/matmul-precision/audit.py

Preparation, execution and audit are single-use evidence writers. Existing
artifacts refuse replay. All models and reference arrays are local; no downloads
or AMD VM workload are required. Do not modify frozen scripts while a worker is
live. Console logs must remain outside the artifact being inventoried.

The mechanism screen requires at least halving the final maximum error in every
selected case, with no increase in values failing the unchanged `1e-4` limit,
against both references. Passing nominates further work only. The full-reference
checks, direct native disagreement and intermediate regressions are retained
regardless of the screen. This controlled unfused operator sequence differs from
Lokad's fused graph, so it supplies no claim about an unimplemented managed kernel.
