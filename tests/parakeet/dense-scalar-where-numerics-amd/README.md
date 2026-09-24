# Dense scalar Where numerical qualification

Source a49c010c and AMD build 1df68687 produce candidate Core 0c3e6168 /
Data 494413f5. The selected release remains Core 672e5f30 / Data 065b7a7f.
The provider's float arm changes; all 3,188 other existing Core methods,
697 Data methods, implementation flags and public interfaces stay exact.
The new two-method helper handles mixed dense masks as broadcast runs.
Build scope alone does not qualify that behavior. No timing occurs here.

The prospective census retains all 203 provider qualification cases and adds
32 cases: four broadcast layouts at each rank 1–8. Their masks are respectively
full/first, last-axis-scalar/last, trailing-only/alternating, and alternating
singleton axes/alternating. Rank 1 uses 4,096 elements. Higher ranks use first
dimension 8, last dimension 512, with intervening dimensions alternating 1, 2.
All 32 cross the unchanged 4,096-element provider threshold.

There are 235 cases, including 15 expected errors (12 incompatible broadcasts
and three null inputs). The 220 valid cases check exact selected bits, independent
coordinate results, full input stores and guards, mutated inputs, held outputs,
and independently owned results. The original 131 output/mutated-output hashes
must remain exact. NumPy independently checks captured arrays, noncanonical
Boolean bytes, and all 32 added shapes using integer representations of floats.
Signed zero, infinities, NaN payloads and subnormal bits are retained.

Dense layout and shape eligibility admits 152 cases, including mixed masks.
The provider attempts 123 cases and admits 95 after its existing size guard.
Unsupported types, custom/view/reversed layouts, invalid shapes and nonscalar
true operands keep their fallback/error behavior. Twenty additional provider
contracts cover options, missing inputs, type failures and validation order.
Every valid numerical provider call checks its complete ordered profiler stages:
one validation for helper success, four normally, five on attempted refusal.
Profiler durations are never comparative measurements.

One consumer exercises both Tensor<T>.Where and CPUExecutionProvider.Where
with both selected/candidate binaries, in ordinary and AVX512-disabled modes.
That yields 608 numerical helper admissions across both boundaries and modes.
Two separate ordinary code-generation workers exercise all 220 valid cases
through each boundary 81 times, retaining complete provider, generic Where,
Try, SelectMixed and consumer assembly. These diagnostics cannot establish
tiers in a future timed process.

Nine serial jobs use CPU2 for compute and CPU0 for monitoring, SDK 10.0.204 /
runtime 10.0.8, and the existing offline feed. Limits remain 12 GiB available /
3 GiB tmpfs before each job, 8 GiB RSS, 1 GiB remaining memory/tmpfs, 900 seconds
per job, four hours per campaign, 1 GiB output per job and 2 GiB total artifacts.
The 32 retained fixture files are verified and hardlinked; no model is copied.
There is no Windows build/inference or root product change.

Freeze these tools, then invoke `C:/Python313/python.exe -X utf8 -B` with
`tests/parakeet/dense-scalar-where-numerics-amd/run.py` and sequential commands
`prepare`, `stage`, `launch`, `observe`, `collect`; finally run `audit.py`.
Existing namespaces are refused. Collection requires terminal PID/birth owners.
Local evidence: `artifacts/parakeet-dense-scalar-where-numerics-amd-20260924`.
VM evidence: `/dev/shm/lokad-parakeet-dense-scalar-where-numerics-20260924`.
The living plan is `.agent/m59-parakeet-dense-scalar-where-20260924.md`.
