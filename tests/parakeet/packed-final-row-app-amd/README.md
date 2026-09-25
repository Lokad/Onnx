# Complete Parakeet comparison for the packed final-row fix

Require complete native/public correctness in both modes, censusfa159300,
focused contracts41/41/2 at6295ad30 and the actual zero-reconstruction counter
comparison before preparation. Compare isolated selectedM73 Core49c3a958/
Dataa893952f with candidateM78 Core49901366/Data01e9e784. The prior e5 repeatability
failures remain release blockers.

Reuse AudioBenchmark and the original ORT1.29 consumers unchanged. Six fresh
CPU2 processes run current,candidate,ORT,ORT,candidate,current, each with20 clips,
one warmup and three measured passes:480 requests and360 measurements. CPU0
monitors. No compilation or diagnostic profiler runs during this campaign.

Require all63 repeatability controls: corpus process max/min<=1.10 and each
clip<=1.20 for every engine. Admission requires>=3% corpus gain and no clip>5%
slower; report candidate/ORT<=1.05 separately. Preserve every clock, setup
interval, public result, immutable input and held-output check. No trimming,
unchanged retry or threshold adjustment. consumer_scope.py proves that only
prerequisites change; request validation, worker, resources, exact scoring,
audit and admission tests remain byte-identical to M70.

Retain11GiB available/3GiB tmpfs preflight,12GiB RSS,1GiB remaining memory/tmpfs,
1GiB output/job,2GiB stage,3600seconds/job,4hours/stage and monitor gaps<10s.
One VM workload at a time; never overwrite linked inputs. The existing bounded
preflight wait retains memory observations and never lowers the threshold.

Use Python3.13 -X utf8 -B on consumer_scope.py and test_admission.py, then run.py
prepare, stage, launch, observe while live, collect once terminal, and audit.py
once. Tools freeze at prepare. Local artifacts use
parakeet-packed-final-row-app-amd-20260925; the VM uses
/dev/shm/lokad-parakeet-packed-final-row-app-20260925.

Full-model and counter closure identities remain required before preparation.
Application admission is distinct from release admission. Resolve retained e5
controls and all remaining qualification before integration or BENCHMARK.md changes.
