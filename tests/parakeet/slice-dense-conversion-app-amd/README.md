# Complete Parakeet comparison after positional-copy diagnosis

Require full-model correctness in both CPU modes, the unchanged3253 Core
methods and the sole guarded dense-conversion override, all395 tensor cases in
each mode, and successful matched positional attribution before preparation.
Current Coref95a13c5 and candidate49c3a958 use ordinary Dataa893952f.

Reuse AudioBenchmark7eca033a and the original ORT1.29 consumers unchanged.
Six fresh CPU2 processes run current, candidate, ORT, ORT, candidate, current,
each with20 clips, one warmup and three measured passes:480 requests and360
measurements. CPU0 monitors. There is no build or profiler in this campaign.

All63 repeatability controls must pass: corpus process max/min<=1.10 and each
clip<=1.20 for every engine. Require>=3% corpus gain and no clip>5% slower;
report candidate/ORT<=1.05 separately. Preserve every clock, setup interval,
complete public result, immutable input and held-output check. No trimming,
unchanged retry or threshold adjustment. consumer_scope.py verifies that only
prereqs changes from M70; request validation, worker, resources, exact scoring,
audit and admission tests remain byte-identical.

Keep11GiB available/3GiB tmpfs preflight,12GiB RSS,1GiB remaining memory/tmpfs,
1GiB output/job,2GiB stage,3600seconds/job,4hours/stage and monitor gaps<10s.
Reuse assets/products/consumers through verified immutable links. One VM
workload at a time; collect only terminal owners; never overwrite linked inputs.

From the repository root run Python3 with -B on consumer_scope.py and
 test_admission.py, then run.py actions prepare, stage, launch, observe while
live, collect once terminal, and audit.py once. Redirect audit output outside
its campaign directory. Tools freeze at prepare. Local namespace is
artifacts/parakeet-slice-dense-conversion-app-amd-20260925 and remote namespace
/dev/shm/lokad-parakeet-slice-dense-conversion-app-20260925.

Admission still requires shared/e5/Pyannote, graph comparisons, meetings,
normal root/full suites and package qualification before product integration
and BENCHMARK.md changes. A successful profile alone is not a release speedup.

The profile prerequisite joins the retained successful control and the sole
candidate from its missing-leg recovery. The immediate11GiB memory refusal
and original code1 remain bound; no control or candidate workload was repeated.
Both original raw collections and the prospective group checks are required.
