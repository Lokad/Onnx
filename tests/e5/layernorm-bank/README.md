# Complete LayerNorm-bank timing

The [completed result](results-20260920.md) is **inconclusive**: every candidate
gain/regression check passes, but three banks fail per-worker duplicate controls.
All aggregate controls and complete output/resource checks pass. Real-bank means
are 13–17% lower with the wider transform. These are promising component
observations, with no product integration or whole-model speedup established.
All samples and visits remain in the [observations](observations-20260920.json).

This experiment reuses the exact kernels and all 125 real captures from the
[closed AMD arithmetic/code proof](../layernorm-amd-proof/results-20260920.md).
It measures all 25 complete normalizations for each e5 case, including centered
statistics and output stores. Four extra thirty-row banks exercise widths
383/385 with and without bias; columns map modulo the original width 384.

Four sequential fresh workers use Product, two identical Copy labels and Wide.
Every bank has 16 fixed warmup cycles and 48 measured cycles, rotating four
positions. Samples repeat 128/32/8/8/2 banks for 8/30/padded128/128/512 tokens,
and 32 for diagnostic banks. All first calls, warmups, measured ticks, allocations,
GC counts and output hashes are retained. The actual first output is held through
later calls. Diagnostic product references are saved for independent scalar checks.

Controls must agree within 1% overall and 2% per worker. Wide must improve
30/padded128/128 banks by at least 2% against Product and both copies, with no
aggregate regression above 1% or worker regression above 2% on any bank. This
engineering screen supplies no calibrated confidence or complete-model claim.

Use `prepare.py stage --artifact <new-directory>`, then build `Probe.csproj`
with `--tl:off --nologo -v minimal`, frozen product/source directories and output
`<artifact>/payload/bin`. `local_check.py --artifact <directory>` runs only host
identity and input construction, independently checking all nine banks without
kernel inference. Retain build and `test_audit.py` logs, commit tools, then
`prepare.py freeze --artifact <directory>` binds sources and dependencies.

`vm.py launch/poll/collect --artifact <directory>` verifies the closed hardware
prerequisite, exact existing remote captures and actual terminal births. Timing
uses CPU2/.NET10.0.8/AVX-512, clean runtime settings and fixed resource/activity
guards. Collection streams locally after all original processes terminate.

Run `audit.py --payload <artifact>/collected --origin
artifacts/e5-layernorm-amd-proof-20260920/collected --output <artifact>/audit.json`,
then `close.py --artifact <directory>`. The closer separately reconstructs integer
timing totals and all decisions before writing reports and a complete receipt.
Successful writers are single-use; preserve failed attempts without replaying
completed inference. No product arithmetic or default changes in this experiment.
