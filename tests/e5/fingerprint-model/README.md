# Whole e5 requests with one prepared graph

This prospective experiment measures the qualified fingerprint cache in actual
public `ExecutionOptions.Memory` calls. It uses one prepared graph, core and set
of weights. Three role labels share that state; the owner-local cache property
is set outside timing. The prepared immutable cache remains resident even for
disabled roles. This differs from the earlier failed two-engine paired A/A and
does not replace an isolated deployment or native ORT comparison.

The archived product is `faf2844`, production-equivalent to `a34720e`; its core
SHA256 is `48ca1d62ee2586d81072b8e347a671013d00abf8fe637c89eff65314e13cc710`.
The probe is built from the frozen source files against those exact binaries;
its build predates the tools commit. It is not claimed to be an archive build.
All models and native reference arrays are existing, immutable local assets.

First run twenty fresh identical-control workers: four visits to each of five
lengths. All three measured labels disable the cache. Each worker first settles
four enabled calls, then alternates settings for 15 cumulative Execute seconds
each. Twenty-four measured cycles contain all six role permutations four times
in seeded order. Role batches contain 32/16/4/4/2 calls at 8/30/padded128/128/512
tokens. Every call records Execute and Reset-plus-Execute ticks, allocation and
all GC deltas. Storage is allocated beforehand; no sample is excluded.

For both boundaries and each case, every pair of identical labels must be
within 0.5% overall, 1% in every worker and 1% at each pooled batch position.
The largest divided by smallest same-position ratio must be <=1.01. These
are descriptive eligibility screens, not confidence intervals. If any fails,
no candidate phase follows under this protocol.

Only the independently audited A/A receipt permits a second fixed twenty-worker
phase. A/B remain off; C enables the cache. A/B must pass the same controls.
C divided by the arithmetic mean of A/B must improve both boundaries by at
least 2% at eight tokens and 1% at thirty. The other cases may regress at most
1% overall; every worker/case may regress at most 2%. Failed controls make the
performance conclusion inconclusive. No default changes automatically.

All complete before/after arrays must match bits and native references at
scaled error <=1e-4. Inputs, held outputs, exact graph fingerprints and the
prepared cache stay unchanged. Workers inherit CPU2 before CLR startup;
supervision uses CPU0. Limits: 300 seconds, 6 GiB group RSS, 1 GiB available
memory, observed foreign CPU <=2% of machine capacity, guest steal <=0.5%.
Process snapshots miss some exited activity; hypervisor neighbors are unobserved.
Every original PID/birth must be absent before collection. Observation timeouts
never restart a process, and all successful writers refuse existing outputs.

Build `Probe.csproj` with `FrozenProductDirectory` pointing to the qualified
product replay's Release directory, using `--tl:off --nologo -v minimal`.
`smoke.py` runs a bounded local functional check; `test_audit.py` checks actual
smoke arrays, malformed records, exact scheduling, bias detection and retained
tails. `prepare.py` freezes committed tools, binaries, inputs and both schedules.
`vm.py launch|poll|collect --artifact <artifact> --phase aa|compare` manages
the exclusive VM; `audit.py` independently reconstructs all evidence and screens.
The local artifact is `artifacts/e5-fingerprint-model-20260920`.
