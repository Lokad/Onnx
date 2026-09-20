# Locally balanced whole e5 requests

This distinct experiment follows the [failed global-order controls](../fingerprint-model/aa-results-20260920.md)
and [retained-data label diagnostic](../fingerprint-model/label-diagnostic-results-20260920.md).
It tests whether the qualified exact fingerprint cache improves actual public
`ExecutionOptions.Memory` calls. The original evidence remains unchanged.

Each fresh worker runs **48 cycles in eight locally balanced six-cycle blocks**.
Every block contains all six permutations of A/B/C, shuffled by Fisher-Yates
using a continuing 32-bit LCG (multiplier 1664525, increment 1013904223), seeded
with `20260920 + 100 * visit + caseIndex`. No diagnostic passing seed is selected.
There are four visits to each of 8/30/padded128/128/512 tokens; alternating visits
reverse the case order. Batches contain 32/16/4/4/2 calls respectively. All 33,408
measured calls per phase count, including GC tails. More samples do not establish
independence or statistical confidence.

One graph, core, weight set, released-buffer set and immutable cache serve every
role. The owner-local cache property changes outside timing; the cache remains
resident even while disabled. Four enabled settling calls precede alternating
off/on conditioning for 15 cumulative Execute seconds per setting. All calls
record Execute and enclosing Reset-plus-Execute ticks, allocation and GC deltas.
There is no forced collection, profiling, tiering override or discarded sample.

First run twenty workers with A/B/C all disabled. For **each case and both
boundaries**, every pair must be within 0.5% overall, 1% in every worker and 1%
at every pooled batch position. Max/min same-position ratio must be <=1.01.
These unchanged limits are descriptive eligibility screens, not confidence
intervals. Any failure ends this protocol without a candidate phase.

Only an independently closed passing A/A gate permits the frozen comparison:
A/B stay off, C is on. A/B must pass the same controls. C / mean(A,B) must be
<=0.98 at eight tokens, <=0.99 at thirty, and <=1.01 at all other cases, for both
boundaries; every worker ratio must be <=1.02. Passing common-state evidence
still needs a separately declared isolated deployment/native comparison before
default promotion. This experiment does not update an ORT latency ratio.

The product is archived `faf2844`, production-equivalent to `a34720e`, core SHA256
`48ca1d62ee2586d81072b8e347a671013d00abf8fe637c89eff65314e13cc710`.
The probe is a locally built, source-pinned diagnostic; no probe archive-build
claim is made. All complete before/after arrays must match bits and native
references at scaled error <=1e-4. Inputs, held outputs, exact fingerprints and
the immutable cache must remain unchanged.

AMD workers inherit CPU2 before CLR startup (.NET10.0.8/AVX512); supervisor uses
CPU0. Guards: 300 seconds per worker, 6 GiB group RSS, 1 GiB available memory,
observed foreign CPU <=2% of machine capacity and guest steal <=0.5%. Snapshots
miss some exited activity and do not observe hypervisor neighbors. Collection
requires the absence of every original PID/creation-time identity. Observation
timeouts never restart a process. Successful writers refuse existing outputs.

Build `Probe.csproj` with `FrozenProductDirectory` pointing to the qualified
product replay's Release directory, and use `--tl:off --nologo -v minimal`.
`smoke.py` checks 72 local calls spanning two complete blocks; `test_audit.py`
checks actual output, schedule reconstruction, local balance, drift, bias and
retained tails. `test_closure.py` checks ownership/resource refusals.
`prepare.py` freezes source, binaries, inputs and both phases. Use
`vm.py launch|poll|collect --artifact <artifact> --phase aa|compare`, then
`audit.py`, `close_phase.py`, and finally `finalize.py` as described in the
prospective plan `.agent/m2-fingerprint-balanced-20260920.md`.
The new artifact is `artifacts/e5-fingerprint-balanced-20260920`.

The orchestration and gates are derived from `../fingerprint-model` and preserved
here as a distinct frozen protocol. Only ordering, fixed cycles, smoke coverage
and protocol/assembly/artifact names change; no production source changes.
