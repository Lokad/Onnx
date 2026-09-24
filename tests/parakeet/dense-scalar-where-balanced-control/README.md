# Fixed identical-binary control after balanced warmup qualification

Reuse qualified consumer `bfd850bc` (59,392 bytes), selected Core `672e5f30`
and Data `065b7a7f` in both nominal roles. Native qualification `ef63ec76` and
review `2a6d0ed0` pass all 22 whole-method entry checks in two processes.

Five serial jobs: SDK check, then current0, candidate1, candidate2, current3.
Role names indicate positions only. No restore/build and no managed flags.
The exact compiled consumer performs 600 round-robin warmup rounds over all
220 cases, followed by 180 measured samples per case in original case order.
Batch sizes, public Where boundary, inputs, oracle and ownership checks are fixed.
Retain 686,400 clocks, 880 setups and 232,929,840 complete calls, including
53,753,040 calls within measured intervals. No trimming or adaptive stopping.

The original scorer is unchanged except its protocol identifier. All four
identical processes must satisfy aggregate max/min <= 1.10 and each case <= 1.20.
Every middle-pair/outer-pair ratio must lie within [20/21, 21/20], for all 220
cases and five aggregates: all220, target6, other_uniform42, fallback74, added98.
Every bound is required. Eleven adversarial scorer tests cover these limits.
Failure parks M59; no unchanged retry, subset selection or relaxed threshold.

From the repository root, use `C:/Python313/python.exe -X utf8 -B` with
`tests/parakeet/dense-scalar-where-balanced-control/run.py`, sequentially:
`prepare`, `stage`, `launch`, `observe`, `collect` after terminal owners, then
run `audit.py`. Refuse existing namespaces and never observe a closed campaign.
The audit checks all journals in their new round-robin order, exact outputs,
input hashes, ownership, immutable files, PID/birth identities and resources.

Bounds: CPU2 for workers/threads, CPU0 monitor; 12 GiB available / 3 GiB tmpfs
preflight, 8 GiB RSS, 1 GiB remaining memory/tmpfs, 900 seconds/job,
four hours/campaign, 1 GiB output/job, 2 GiB total artifacts. Keep root product
unchanged. This control can admit a future prospectively frozen candidate race;
it cannot establish a candidate speedup or change BENCHMARK.md.
