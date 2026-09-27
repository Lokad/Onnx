# Locate the persistent unmapped-call penalty

The packed-row screen 3e6a7562 remains rejected. The target improved 28.9032%,
but the unmapped control doubled and two narrow controls failed regression or
repeatability. This diagnostic changes no product, numerical tolerance, case,
order, batch size, warmup count or admission limit. It does not rescore that
screen or admit the candidate, even if its instrumented clocks differ.

Resolve one ambiguity: does the first unmapped call after the mapped projection
account for the extra cost, or are all seven fallback calls slower? Current
reuses original row-major B across both cases; candidate switches from a separate
20,986,880-byte packed B to original B. Cache residency is a plausible cause.
Sustained fallback cost or different emitted code are competing explanations.
This observation distinguishes the temporal pattern, not cache misses or JIT
instruction identities. The narrow process plateaus remain independently open.

Start from the exact successful e9afd0b6 consumer source. A reversible patch adds
two preallocated arrays, individual start/duration timestamps around the seven
unmapped calls in every round, and serialization of those arrays. Original batch
clocks, allocations, copies, scratch, numerical and ownership checks all remain.
No clock arrays are allocated during timing. Other case calls are unchanged.
Per-call timing adds overhead, so all new clocks are diagnostic only.

Reuse the unchanged worker, supervisor, transport and zero-warning build audit.
Compile one consumer against current; swap only Core (65f15a41/af19b3b4).
Four fresh ordinary CPU2 processes run current/candidate/candidate/current.
All six cases remain in fixed order for 600 warmup and 180 measured rounds, with
seven wide and 102 narrow calls per batch. Keep all 723,840 calls, 18,720 original
batch clocks and 21,840 additional individual clocks, including warmup.
Validate individual intervals are ordered, inside their original batch, positive
and exhaustive. Every returned output, input, ownership and resource check from
the original consumer and auditor remains in force.

Report all seven mean durations per process and across paired products, plus
every original 60-round block. First-call concentration is supported only if
the unmapped batch slowdown reproduces (>1.05), all six later-call aggregate
ratios are <=1.05, the first-call ratio is >1.05, and the first call accounts for
at least 80% of the sum of positive per-position candidate-minus-current excess.
These are descriptive, prospectively fixed hypothesis checks, never performance
admission. If later calls are also slow, inspect emitted native fallback code;
if the diagnostic no longer reproduces the slowdown, preserve the discrepancy.
Never discard calls or lengthen warmup to make this explanation fit.

Freeze before execution. Local namespace:
artifacts/parakeet-decoder-unmapped-calls-amd-20260927; VM:
/dev/shm/lokad-unmapped-calls-20260927. Preserve original 32 MiB output limit,
3 GiB RSS, 900-second job limit, CPU0 supervision, CPU2 execution and foreign CPU
accounting. The last inventory leaves 200.6 MB allocation headroom below 50 GB.

From repository root, prefix each with C:/Python313/python.exe -X utf8 -B:

    tests/parakeet/decoder-unmapped-calls/test_observations.py
    tests/parakeet/decoder-unmapped-calls/run.py prepare
    tests/parakeet/decoder-unmapped-calls/run.py stage
    tests/parakeet/decoder-unmapped-calls/run.py launch build
    tests/parakeet/decoder-unmapped-calls/run.py observe build
    tests/parakeet/decoder-unmapped-calls/run.py collect build
    tests/parakeet/decoder-unmapped-calls/audit.py build
    tests/parakeet/decoder-unmapped-calls/run.py launch capture
    tests/parakeet/decoder-unmapped-calls/run.py observe capture
    tests/parakeet/decoder-unmapped-calls/run.py collect capture
    tests/parakeet/decoder-unmapped-calls/audit.py capture

Observe the same owner until terminal before collecting, with audit output outside
the artifact. Never replay completed build/capture work. The rejected product
remains isolated; qualified source and BENCHMARK.md stay unchanged.
