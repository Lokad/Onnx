# Observe the exact relocation candidate's runtime phases

The graph comparison of release f95a13c5 and candidate e07a4518 passed numerical
checks and all eight regression gates, but failed short-e5 repeatability:
candidate process ratio 1.141206 exceeds 1.10. Preserve closure def19d3f.

The earlier release/40260aef traces ended their original 780-call prefix around
nine seconds while optimized Lokad methods kept arriving until 16.5–18.6 seconds.
Those traces do not establish the runtime phase of the new e07a4518 product.
This diagnostic answers that exact-product question before any prospective
benchmark correction. It introduces no new product variant or scored retry.

Reuse the already compiled 6,000-call observer 968d3beb and exporter 092bc21e,
including their two compiled reviews and lossless exporter proof. No build runs.
Only product/manifest bindings and the list of jobs change. The original worker,
supervisor, call validator, event reconciliation and resource limits remain exact.
The auditor changes one failed-case selector to retain the repeatability failure.

Four fresh processes run release, candidate, candidate, release. Keep the first
600 warmup and 180 original measurement labels; calls 780–5999 are an unscored
diagnostic extension. All timings, including the late tail, remain diagnostic.
Require 24,000 calls, 48,000 markers, lossless events, original numerical arrays,
immutable inputs, retained outputs and exact identities. Inspect all Lokad method
loads as well as the two matrix dispatchers; do not infer whole-CLR quiescence
from a namespace filter. Runtime optimization, tiering and SIMD settings stay fixed.

CPU2 computes; CPU0 captures and monitors. Retain 11 GiB available RAM/3 GiB
tmpfs before each worker, 8 GiB RSS, 1 GiB remaining RAM/tmpfs, 256 MiB output/job,
512 MiB stage, 900 seconds/job and four hours total. Preflight staging reserves
the full 512 MiB allowance. No other VM workload may overlap.

From the repository root, use C:/Python313/python.exe -X utf8 -B with
consumer_scope.py, then run.py prepare, stage, launch, observe while live,
collect once all owners are terminal, then audit.py once. Do not restart after
an observation timeout or repeat a completed operation. Keep audit stdout outside
the campaign directory. Tools freeze at preparation; update PLAN.md afterwards.

Local: artifacts/e5-relocation-tier-diagnostic-amd-20260925.
Remote: /dev/shm/lokad-e5-relocation-tier-diagnostic-20260925.
The new results cannot alter the failed graph closure or admit a release.
