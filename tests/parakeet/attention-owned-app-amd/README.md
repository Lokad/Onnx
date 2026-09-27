# Complete Parakeet comparison for prepared attention weights

The exact ORT constant-weight path and scoped clocks identify 0.817785 seconds
of repeated packing across 92 attention weights per corpus. Test one policy:
prepare those weights once using the existing owned representation, arithmetic
kernels and cache budget. Expected opportunity is about 1.8% of total time,
not an achieved speedup. Compare current Core a6f7d9f9 with candidate ee5218db,
both using unchanged Data 1ba343fd and the original AudioBenchmark consumer.

Before staging, require the successful complete-model closure plus focused
contracts a293403a, exact-model census ef4e41ec and qualified root fc116763.
Keep the original native/public model prerequisite checks verbatim. The worker,
statistics, request validator, auditor and admission tests remain byte-exact.
Only prerequisite provenance, namespace and verified input copying are adapted.

Run current/candidate/ORT/ORT/candidate/current: twenty clips totaling 213.265
seconds, one warmup and three measured passes, 480 requests / 360 measured
clocks. Retain every clock. Require all 63 repeatability controls (corpus <=1.10,
each clip <=1.20 for every engine), at least 1% candidate corpus gain and no
clip more than 5% slower. Candidate/ORT <=1.05 is a separate parity target.
Preserve a failed verdict; no unchanged retry, trimming or option sweep.

CPU2 executes, CPU0 monitors. No other VM work overlaps timing. Preserve the
original 11GiB available RAM / 3GiB tmpfs preflight, 12GiB RSS cap, 1GiB free
RAM/tmpfs, 1GiB output/job, 2GiB stage, 3600 seconds/job, four hours total,
monitor gaps below ten seconds and foreign-CPU checks. No builds or downloads.

Bind the actual successful model closure, freeze the tools, then prefix with
C:/Python313/python.exe -X utf8 -B from the repository root:
tests/parakeet/attention-owned-app-amd/run.py prepare, stage, launch, observe.
Observe the same owner until all its processes terminate; collect and audit.py
once. Artifacts: artifacts/parakeet-attention-owned-app-amd-20260928.
VM: /dev/shm/lokad-attention-owned-app-20260928. Never replay closed stages.

An admitted application result permits broader shared/e5, Pyannote, graph and
root/package qualification. Root product and BENCHMARK.md remain unchanged
until that qualification completes. This experiment alone cannot establish parity.
