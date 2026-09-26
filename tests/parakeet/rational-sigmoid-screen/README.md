# One fixed rational-sigmoid performance screen

Compare qualified current root Core 8bb22038 against the single ORT-derived
rational-sigmoid candidate. Preparation requires its closed compiled, numerical
and generated-code qualification in rational-sigmoid-build. Both products use
the same consumer, ordinary compiler settings and input/output contract.
No further implementation variants are permitted by this protocol.

The original census, Screen.cs, worker, scorer, tests and auditor are reused
unchanged from vector-sigmoid-screen. The adapter changes only product references
and isolated paths. The prior vector-exp screen remains failed; this is a different
arithmetic candidate predicted by the executed-code diagnosis. The existing
75% weighted target and every original gate remain. No instrumentation is enabled
in the timed processes. The projected application benefit (~3.6%) is unproved.

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
differ within the existing bound. The new candidate's 2,048,769-value numerical sweep per
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
    tests/parakeet/rational-sigmoid-screen/run.py prepare
    tests/parakeet/rational-sigmoid-screen/run.py stage
    tests/parakeet/rational-sigmoid-screen/run.py launch build
    tests/parakeet/rational-sigmoid-screen/run.py observe build
    tests/parakeet/rational-sigmoid-screen/run.py collect build
    tests/parakeet/rational-sigmoid-screen/audit.py build
    tests/parakeet/rational-sigmoid-screen/run.py launch capture
    tests/parakeet/rational-sigmoid-screen/run.py observe capture
    tests/parakeet/rational-sigmoid-screen/run.py collect capture
    tests/parakeet/rational-sigmoid-screen/audit.py capture

The result is stored under `artifacts/parakeet-rational-sigmoid-screen-amd-20260927`.
Inspect `analysis.json`'s `admitted`, every failed control/case and the weighted
prediction separately from `passed`, which denotes complete evidence validation.

Before preparation, verify the adapter imports the original worker/scorer/auditor,
all copied consumer bytes are identical, and both closed product identities match.
VM namespace: /dev/shm/lokad-parakeet-rational-sigmoid-screen-20260927.
Audit stdout belongs outside the artifact directory. Do not modify tools after
preparation or repeat completed inference to repair collection or interpretation.
