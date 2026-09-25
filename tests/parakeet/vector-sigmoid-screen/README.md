# One vector Sigmoid screen

This prospective protocol tests the single-method candidate whose compiled and
numerical qualification is closed in `vector-sigmoid-results/contracts-20260925.md`.
The current product is isolated M78, not the qualified release. Its e5 admission
failure remains unchanged. There is no native operator timing or application score
in this screen; ORT's observed shapes guide the managed comparison.

`census.py` reads the closed exact-profile analyses. All 96 encoder activation
nodes contribute 38 shapes and 1,920 calls per 20-clip corpus. The 72 fused native
SiLU nodes each correspond to a managed Sigmoid and a separate multiply. We time
only the complete public Sigmoid call, including validation, layout materialization,
allocation and computation. Input values are synthetic deterministic finite floats,
not captured intermediates. Eight additional cases cover scalar options, doubles,
tails, empty/scalar inputs and reversed, sliced and broadcast layouts.

One common consumer binary runs current/candidate/candidate/current in four fresh
ordinary .NET 10.0.8 processes on CPU 2; the resource supervisor stays on CPU 0.
The exact frozen Core binaries are the only differing runtime files. No environment
ISA/JIT overrides are used. Fixtures and expected outputs are prepared outside
timing. A batch consists of `max(1,min(256,65536/elements))` complete calls, using
integer division and treating an empty input as one element for this formula.
Results are retained in a preallocated array. All cases receive 600 warmup rounds
before any case is measured; 180 subsequent rounds are measured, in census order.
This is fixed before the first timing. Earlier failed Pad protocols and scores
remain unchanged. Every timestamp, warmup and batch size is retained.

Every result's status is checked outside timing. Numerical checks at rounds 0,
599 and 779 use the original scalar expression and existing float 1e-6 / double
1e-12 bounds. End-of-run checks establish immutable inputs and independently owned
held outputs. Same-product process output hashes must match. Cross-product scalar
option, double, scalar and empty outputs must be exact; SIMD output rounding may
differ within the existing bound. The prior 2,048,769-value numerical sweep per
mode remains a prerequisite.

Exact rational arithmetic computes each case's 180-sample mean and the corpus
frequency-weighted sum. Every case and weighted total must repeat within 10%
for each product (94 controls). No case may regress by over 5% (46 gates).
The candidate must reduce the weighted sum by at least 75%, and both candidate
process totals must be below both current totals. Any failed gate rejects this
screen. No outlier trimming, new warmup rule, selected-shape reporting or unchanged
retry is allowed. Passing only permits full native/model qualification and the
original full Parakeet application comparison; it does not establish release
admission or parity. A failed prediction stops this candidate, without ISA,
polynomial, compiler-flag or fusion variants.

The small operator workload requires 2 GiB available memory and 1 GiB tmpfs free
before each job, stays below 3 GiB process-tree RSS and 512 MiB staged/output size,
and leaves at least 1 GiB memory and tmpfs free. Build jobs have 180-second limits;
each timing process has a 900-second limit. Half-second resource samples retain
process birth identity and every thread's affinity. Before/after process CPU
accounting must pass the existing 1% foreign-CPU rule, with its explicit limitation
for exited short-lived processes. Only one VM workload runs at a time.

From the repository root, use `C:/Python313/python.exe -X utf8 -B` for the following
Python commands, in order. Observe actual deployment ownership to terminal status
before collecting. Preparation, build, capture, collection and closure are each
one-time actions. Failed outputs must be preserved.

    -m unittest discover -s tests/parakeet/vector-sigmoid-screen -v
    tests/parakeet/vector-sigmoid-screen/run.py prepare
    tests/parakeet/vector-sigmoid-screen/run.py stage
    tests/parakeet/vector-sigmoid-screen/run.py launch build
    tests/parakeet/vector-sigmoid-screen/run.py observe build
    tests/parakeet/vector-sigmoid-screen/run.py collect build
    tests/parakeet/vector-sigmoid-screen/audit.py build
    tests/parakeet/vector-sigmoid-screen/run.py launch capture
    tests/parakeet/vector-sigmoid-screen/run.py observe capture
    tests/parakeet/vector-sigmoid-screen/run.py collect capture
    tests/parakeet/vector-sigmoid-screen/audit.py capture

The result is stored under `artifacts/parakeet-vector-sigmoid-screen-amd-20260925`.
Inspect `analysis.json`'s `admitted`, every failed control/case and the weighted
prediction separately from `passed`, which denotes complete evidence validation.
