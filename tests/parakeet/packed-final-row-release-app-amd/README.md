# Compare M78 directly against the qualified Parakeet release

The closed M78 application score compared intermediate M73 with M78. This first
release comparison uses qualified Core f95a13c5/Data a893952f against M78 Core
49901366/Data 01e9e784 and fresh Microsoft ORT 1.29. Do not compose the earlier
timing ratios: two separate 5% per-clip regression limits could compound to
10.25%. The actual release must satisfy the original limits directly.

All prior contracts, actual owned-weight census, copy counters and admitted
intermediate application results remain prerequisites. The exact model lineage
binds release -> M73 -> M78, including all 784 intermediate tensors in each
instruction mode and complete public results for every clip and actual role.
Current manifests, public references and runtime files come from the release's
actual retained model run; candidate records come from M78. Nothing is relabeled.

Reuse the original AudioBenchmark and ORT consumers unchanged. Six fresh CPU 2
processes run release, candidate, ORT, ORT, candidate, release. Each executes all
20 clips with one warmup and three measured passes: 480 requests, 120 warmups,
360 measurements. CPU 0 monitors. No profiler or compilation runs concurrently.

Require all 63 repeatability controls: corpus max/min <= 1.10 and each clip
<= 1.20 for each engine. Require >= 3% corpus gain and no clip > 5% slower.
Report the independent candidate/ORT <= 1.05 target. Retain every request,
clock, setup interval, ownership check and complete public result. No trimming,
unchanged retry or threshold adjustment. Application admission alone does not
admit a release; graph, Pyannote and actual root/package qualification remain.

consumer_scope.py preserves the original worker, protocol, request validation,
exact scorer, thresholds and admission tests. Only prerequisite identities and
the auditor's runtime/manifest provenance mapping change. lineage.py and its
eight tests are identical to the reviewed Pyannote adapter.

Keep 11 GiB available/3 GiB tmpfs preflight, 12 GiB RSS, 1 GiB remaining RAM/tmpfs,
1 GiB/job, 2 GiB stage, 3,600 seconds/job, four hours total and monitor gaps < 10s.
Prepare locally while graphs run; stage and launch only after the VM is idle.
Use Python 3.13 -X utf8 -B on consumer_scope.py, test_admission.py and
test_lineage.py. Then run.py prepare, stage, launch, observe while live, collect
once terminal, and audit.py once. Preparation freezes all tools. Keep audit
stdout outside the artifact directory. Local artifacts use
artifacts/parakeet-packed-final-row-release-app-amd-20260925; the VM uses
/dev/shm/lokad-parakeet-packed-final-row-release-app-20260925.

Preserve both prior failed M73 graph controls and all prior application scores.
This comparison changes the baseline, not the measured candidate or protocol.
