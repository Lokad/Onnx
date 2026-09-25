# Complete Parakeet comparison after owned-weight diagnosis

Require complete native/public correctness in both modes (bb1a758c), the corrected
87-weight census (577a07a6), focused contracts27/27/2 (0f8bdc94) and the actual
counter comparison before preparing this campaign. Compare isolated selectedM73
Core49c3a958/Dataa893952f against correctedM76 Core82c02785/Data3f80f8cb. Both Core
and Data differ. The selected candidate's earlier application result is admitted,
but its failed e5 repeatability controls still prevent release promotion.

Reuse AudioBenchmark and the original ORT1.29 consumers unchanged. Six fresh
CPU2 processes run current, candidate, ORT, ORT, candidate, current, each with20
clips, one warmup and three measured passes:480 requests and360 measurements.
CPU0 monitors. No compilation, diagnostic profiler or counter consumer runs here.

All63 repeatability controls must pass: corpus process max/min<=1.10 and each
clip<=1.20 for every engine. Require>=3% corpus gain and no clip>5% slower;
report candidate/ORT<=1.05 separately. Preserve every clock, setup interval,
complete public result, immutable input and held-output check. No trimming,
unchanged retry or threshold adjustment. consumer_scope.py verifies that only
prereqs changes from M70; request validation, worker, resources, exact scoring,
audit and admission tests remain byte-identical.

Keep11GiB available/3GiB tmpfs preflight,12GiB RSS,1GiB remaining memory/tmpfs,
1GiB output/job,2GiB stage,3600seconds/job,4hours/stage and monitor gaps<10s.
Reuse assets/products/consumers through verified immutable links. One VM workload
at a time; collect only terminal owners; never overwrite linked inputs.

Use Python3.13 -X utf8 -B on consumer_scope.py and test_admission.py, then run.py
prepare, stage, launch, observe while live, collect once terminal, and audit.py
once. Redirect audit stdout outside the campaign directory. Tools freeze at
prepare. Local artifacts use parakeet-owned-packed-weight-app-amd-20260925;
the VM uses /dev/shm/lokad-parakeet-owned-packed-weight-app-20260925.

The counter prerequisite joins one retained successful worker with only its
three unstarted successors. The original memory refusal and code1 remain bound.
Application admission is distinct from release admission. Resolve the existing
e5 repeatability failures and complete remaining release checks before integration
or BENCHMARK.md changes. No application gain is inferred from phase durations.
