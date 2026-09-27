# The admitted pointwise candidate preserves Pyannote correctness

All three jobs pass: four identity-rejection probes, current correctness and
candidate correctness. Together the products check **36 arrays / 5,834,214 values
and 32 complete public diarization requests**. Every candidate array and complete
public result matches current exactly, including speaker centroids and timelines.
Input immutability and independently held outputs pass.

Maximum scaled error against the original Microsoft ORT arrays is
0.00003728270531 for both products, below the unchanged 0.0001 bound. The same
GraphQualification consumer `d78c45b9` checks current Core `47984318` and candidate
Core `7cac6788`, both with Data `dd56902f`. No product, consumer or model was built
or downloaded. All four wrong-Core/wrong-Data probes reject before creating
outputs, and their child identities are terminal.

All 452 resource samples pass; peak owned RSS is 1,207,160,832 bytes. Supervisor
1229174 / birth1790536270.4 and all children are terminal. The 174 files were
collected once and independently audited. Durations and retained operator profiles
are correctness-run observations, not a Pyannote performance comparison.

The existing worker, validators, probes and 24 adapter tests remain unchanged.
The auditor changes only its candidate label. Product compatibility retains all
original public bindings and method flags; the reused consumer's original
95/96-method body proof had no method-flag inventory, and no new claim is made
about those absent records.

The [Parakeet gain](pointwise-tail-app-20260927.md) remains 2.132562% with a
1.155498 candidate/ORT ratio. Graph regression, complete Pyannote application
regression, portable boundary tests and actual-root/package qualification remain
before source or BENCHMARK.md promotion. The rejected component screen remains
explicit and unchanged.

[Identities, native errors, probes and resources](pointwise-tail-pyannote-observations-20260927.json),
[protocol](../pointwise-tail-pyannote-amd/README.md).

Closure: `f284cc72de08b0113daa8a5482e2a2dbea2bd6a89ea00e128a569036394eebb8`.
Raw evidence: `artifacts/parakeet-pointwise-tail-pyannote-amd-20260927`.
