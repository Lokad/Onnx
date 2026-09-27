# Graph release checks for the single attention ownership candidate

All actual prerequisites are bound: Parakeet application 600e9e67 admits
1.128490% gain, shared/e5 0f5d4baa and Pyannote correctness ad6afcc3 pass.
Complete Parakeet models 2d64fad9 and prior graph closure faf553e7 are retained.
Every preceding VM owner is terminal. Current Core a6f7d9f9 and candidate ee5218db differ in one
preparation-policy method; all arithmetic leaves, flags and public bindings agree.

Reuse the original three ReleaseBenchmark consumers: d827e3b9 for ordinary
cases, 0b228b2d for 30-token e5 and e437850d for eight-token e5. Load Core only;
no consumer/product builds, new kernels, flags or downloads. The four-export /
eight-case census proves coverage without claiming execution of a specific route.

Preserve all 72 original jobs over five e5 inputs, DINOv3, ResNet50 and GPT-2:
three correctness workers and six timed processes per case, ordered current,
candidate, ORT, ORT, candidate, current. Keep 6000 warmups for eight-token e5,
1200 for 30-token e5 and 600 otherwise, then 180 measurements. Retain all
73,512 calls, 8,640 measured clocks and 72 setup intervals. Native ORT is 1.29.0.

The original worker, numerical checks, scoring and auditor are unchanged.
Require finite values, native scaled error <=1e-4, exact current/candidate arrays,
ownership, <=5% per-case regression and <=10% process disagreement. Preserve
all clocks and failures. These are release regression checks, not new variants.

Keep CPU2 compute / CPU0 monitor, 11GiB RAM / 3GiB tmpfs preflight, 8GiB RSS,
1GiB free RAM/tmpfs, 900 seconds/job, 2GiB campaign files and four hours total.
Check retained input bytes and terminal owners; use verified existing runtimes.

After binding real preceding closures, run all local prerequisite tests and
freeze tools. Prefix C:/Python313/python.exe -X utf8 -B from the repository root:
tests/parakeet/attention-owned-graphs-amd/run.py prepare, stage, launch, observe.
Observe the same owner until terminal, collect once, then audit.py once.
Artifacts: artifacts/parakeet-attention-owned-graphs-amd-20260928.
VM: /dev/shm/lokad-attention-owned-graphs-20260928. Complete Pyannote application
and root/package qualification remain required before release promotion.
