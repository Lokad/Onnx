# Complete LayerNorm banks with minimum-work conditioning

The [completed result](results-20260920.md) passes every original control, gain,
regression, correctness and resource check. Real complete-bank means are
15.11–16.49% lower. All 13,824 measured batches and 18,432 conditioning batches
remain in the [observations](observations-20260920.json), with exact outputs and
zero measured allocation/GC counts. This supports subsequent product integration
and model qualification; it does not establish a whole-model or native ORT gain.

The [preceding experiment](../layernorm-conditioned/results-20260920.md) passed
duplicate controls but failed the candidate regression screen in one 8-token
worker. Its time-only conditioning admitted only three complete startup cycles.
This distinct protocol adds a fixed minimum of **128 complete four-role cycles**
alongside the existing three seconds of timed kernel work. It stops at the first
cycle satisfying both conditions. All conditioning observations are retained.

Everything else is unchanged: exact archived kernels and captures, all nine
banks and four fresh workers, role/case order, repetitions512/128/32/32/8 for
real banks and128 for diagnostics, then96 measured cycles. There are13,824
measured batches, at least18,432 conditioning batches and144 first calls.
Every original control, gain and regression threshold still applies. No sample
exclusions, forced GC, runtime override, product change or whole-model claim.

Run `prepare.py stage --artifact artifacts/e5-layernorm-minimum-20260920`.
This verifies the closed v2 receipt, generates thirteen host/tool files from its
exact sources and records all before/after pins and the full diff. It reuses515
existing pinned dependencies without copying the capture bank.

Build generated `host/Probe.csproj` Release with `--tl:off --nologo -v minimal`,
`FrozenProductDirectory=<hardware artifact>/collected/product`,
`FrozenSourceDirectory=<new artifact>/payload/source`, output`payload/bin`;
retain `build.log`. Run generated `tools/local_check.py --artifact <artifact>`
for actual identity and independent input mapping checks without inference.
Run generated `tools/test_audit.py`, retaining `unit-tests.log`. Tests exercise
both minimum-work and elapsed-time cutoff boundaries, including extra cycles.

Commit, then `prepare.py freeze --artifact <artifact>` verifies source/build
identities and every archive entry. Generated `tools/vm.py launch/poll/collect`
with `--artifact` checks all three closed prerequisites, original process births,
remote captures and resources. Four workers run on AMD CPU2/.NET10.0.8 with
the same600-second/3-GiB/2-GiB and foreign-activity limits.

After every original process terminates, run generated `tools/audit.py --payload
<artifact>/collected --origin artifacts/e5-layernorm-amd-proof-20260920/collected
--output <artifact>/audit.json`, then `tools/close.py --artifact <artifact>`.
The closer separately reconstructs raw integer totals, both conditioning bounds
and all decisions before writing reports/receipt. Rehash all closed files and
reports and verify actual terminal births independently. Successful writers are
single use; all prior unsuccessful results remain unchanged.
