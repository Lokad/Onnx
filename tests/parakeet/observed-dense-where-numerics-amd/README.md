# Qualify the observed-mask composition on the current release

Reuse the exact retained [235-case census](../dense-scalar-where-numerics-amd/README.md),
consumer, provider contracts, coordinate oracle, fixtures, resource worker and
full numerical auditor. Preparation checks that these files and every generated
case are identical. Only product identities, prerequisite bindings, namespace
and the prospectively declared 11 GiB memory preflight differ.

Current Core is `37c24375`, Data `cc37b19e`; candidate Core is `f95a13c5`,
Data `a893952f`. Joint build/inventory qualification `7d5296b2` proves the same
3,250 other Core methods, all 697 Data methods, flags and public interfaces.
The new provider and two helper bodies exactly match their retained versions.
The initial inspection dependency failure stays recorded; no rebuild occurred.

All 235 cases run through both APIs with both products and instruction modes;
15 errors, 20 additional provider contracts, original hashes, raw Boolean
truth, NaN bits, input immutability, ownership and ordered validation stages
remain mandatory. Separate code-generation workers retain all 220 valid cases.
No result here is a speed measurement or application admission.

Nine serial AMD jobs use SDK 10.0.204 / runtime 10.0.8, CPU 2 compute and CPU 0
monitor. Preflight 11 GiB available / 3 GiB tmpfs, RSS below 8 GiB, at least
1 GiB remaining memory/tmpfs, 900 seconds/job, four hours/stage, 1 GiB output/job
and 2 GiB total. The 11 GiB preflight was specified before preparation in M70;
all numerical and resource checks remain unchanged thereafter.

Use `C:/Python313/python.exe -X utf8 -B` with `run.py prepare`, `stage`,
`launch`, `observe`, `collect`, then `audit.py`. Keep auditor stdout outside
the audited directory. Freeze all tools before preparation. Local namespace:
`artifacts/parakeet-observed-dense-where-numerics-amd-20260924`;
VM: `/dev/shm/lokad-parakeet-observed-dense-where-numerics-20260924`.
