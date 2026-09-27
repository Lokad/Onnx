# Check Pyannote with the fixed prepared-row candidate

Qualify Core `af19b3b4` against the current release `65f15a41`, both with
unchanged Data `da72ca54`. The single prepared-row intervention has passed exact
Parakeet model checks, the independent application comparison (3.122916% gain,
1.207329 times ORT), and shared/e5 correctness. This campaign selects no new
implementation and measures no performance. Preserve the rejected mixed-layout
operator screen and its failed first-call-only explanation.

Reuse exact GraphQualification `d78c45b9`, most recently qualified with the rational
sigmoid release. Its Core and Data identity arguments and guards are unchanged.
The closed 3,980-method compatibility review connects that product through the
actual current root to this candidate: only the two prepared-MatMul dispatch
methods change and one internal row kernel is added. All original method flags,
public bindings and Data methods remain identical. The consumer itself is reused
byte for byte. Its original inspector established 95/96 methods unchanged and
only Main identity-argument edits; it did not record flags. No retrospective
claim about those original consumer flags is made.

Run three serial jobs: identity-probes, selected correctness, candidate correctness.
Four wrong-Core/wrong-Data probes must reject before output creation, and their
PID/birth identities must be terminal. Each product then checks 18 arrays /
2,917,107 values and 16 complete public requests. Preserve all original native
finite-output, shape, timeline, speaker, centroid and 1e-4 numerical checks.
The addressing-only candidate must additionally produce byte-identical arrays
and exact complete public results, including centroids, against the current
release. Inputs and held outputs remain immutable.

`consumer_scope.py` verifies the original padding checker, identity probes,
native auditor and public semantics unchanged. The worker retains its original
resource monitoring; build jobs were removed in the already-qualified sigmoid
adapter. No model download, package restore or product/consumer build occurs.
Keep 11 GiB available RAM / 3 GiB tmpfs preflight, 8 GiB RSS, 900 seconds/job,
1 GiB remaining RAM/tmpfs, 1 GiB/job output and 2 GiB campaign files. CPU 2
computes, CPU 0 monitors, with only one VM workload active.

From the repository root, prefix commands with `C:/Python313/python.exe -X utf8 -B`:

    tests/parakeet/decoder-packed-row-pyannote-amd/consumer_scope.py
    tests/parakeet/decoder-packed-row-pyannote-amd/consumer_reuse.py
    -m unittest discover -s tests/parakeet/decoder-packed-row-pyannote-amd -p test_*.py
    tests/parakeet/decoder-packed-row-pyannote-amd/run.py prepare
    tests/parakeet/decoder-packed-row-pyannote-amd/run.py stage
    tests/parakeet/decoder-packed-row-pyannote-amd/run.py launch
    tests/parakeet/decoder-packed-row-pyannote-amd/run.py observe

Freeze tools at preparation. Observe the same owner until terminal, then collect
and audit once, placing audit stdout outside the artifact directory. Preserve
failure evidence; never replay a completed namespace. Local evidence goes in
`artifacts/parakeet-decoder-packed-row-pyannote-amd-20260927`; VM evidence goes in
`/dev/shm/lokad-parakeet-decoder-packed-row-pyannote-20260927`.

All graph and complete Pyannote application comparisons, followed by actual-root
and package qualification, remain necessary before source or BENCHMARK promotion.
