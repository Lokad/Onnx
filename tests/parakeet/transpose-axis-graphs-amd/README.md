# Graph release checks for the admitted transpose candidate

Bind actual complete Parakeet models 8568a229, admitted application 2e4741a6
and shared/e5 fa74521c. Pyannote correctness must close before preparation;
an empty digest rejects the lane before creating outputs. The previous graph
closure is 0878b709. The current Core 4e97e2ae and candidate c471f5d1 differ only
in TransposeInto. The 3,985-method compatibility proof preserves every other
method, flag and public binding, and connects actual root ee4a38ff to the
previous measured graph product. Focused contracts c0c063bb pass all 487 checks.

Reuse the original ReleaseBenchmark consumers: d827e3b9 for ordinary cases,
0b228b2d for 30-token e5 and e437850d for eight-token e5. These load Core only.
No consumer/product build, new variant, flags or downloads. The four-export /
eight-case census proves coverage, without claiming a particular runtime route.

Keep all 72 jobs over five e5 inputs, DINOv3, ResNet50 and GPT-2: three correctness
workers and six timed processes per case, ordered current, candidate, ORT, ORT,
candidate, current. Retain 6000 warmups for eight-token e5, 1200 for 30-token e5
and 600 otherwise, then 180 measurements. Retain all 73,512 calls, 8,640 measured
clocks and 72 setup intervals. Native ORT is 1.29.0.

Original worker, numerical checks, scoring, auditor and fixture preparation are
unchanged. Require finite values, native scaled error <=1e-4, exact product
arrays, ownership, <=5% per-case regression and <=10% process disagreement.
Preserve all clocks and failures. This checks release regressions for the single
admitted candidate; it does not select an additional optimization.

Keep CPU2 compute / CPU0 monitor, 11GiB RAM / 3GiB tmpfs preflight, 8GiB RSS,
1GiB free RAM/tmpfs, 900 seconds/job, 2GiB campaign files and four hours total.
Check retained bytes and terminal owners; use existing verified runtimes.

After binding the actual Pyannote closure, run all local prerequisite tests,
read-only prepare.previous_closed() and freeze the tools. From repository root,
prefix C:/Python313/python.exe -X utf8 -B and use
tests/parakeet/transpose-axis-graphs-amd/run.py prepare, stage, launch, observe.
Follow the same owner to terminal, collect once, then audit.py once. Never replay.
Artifacts: artifacts/parakeet-transpose-axis-graphs-amd-20260928.
VM: /dev/shm/lokad-transpose-axis-graphs-20260928. Complete Pyannote application
and actual-root/package qualification still precede release integration.
