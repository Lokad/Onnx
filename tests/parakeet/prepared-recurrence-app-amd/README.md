# Full Parakeet application comparison for prepared recurrence

This is the original production-consumer comparison with a newly qualified
candidate. Its runtime, request validation, exact statistics, admission tests,
resource limits and performance gates are unchanged from the inclusive-packing
application lane. `consumer_scope.py` verifies this; only prerequisite composition
changes. No product, native runner or managed consumer is rebuilt.

Selected Core672e5f30/Data065b7a7f and candidate Core3c23b44a/Datacc37b19e have passed
compiled-scope review1b7f8e21, 300 focused contracts aacf2dbe, actual decoder/call
qualification20b3bc4b, and all native/public trajectories in both modes c0b2f614.
The raw build-wrapper refusal b2d46d3b remains accepted only through its separate
compiler-name review. The original decoder-helper refusal bc2ce801 is retained.

The complete-call timing d0eb4898 remains **invalid and unadmitted**: 40/84
repeatability controls failed, despite all numerical/resource checks passing.
Its raw ratios are not accepted speedups. Preserve all 30,400 clocks and do not
repeat that experiment unchanged or trim its first measured passes. No cause for
the variation has been established. The component helper retains every internal
output across ten passes and its operation differs from complete transcription.

The prospective decision is to measure the user's actual target with the
unchanged production application runner. Full numerical qualification permits
this independent experiment; the earlier plan's component-admission prerequisite
is superseded. No failed component control is waived or relabeled as passing.
This application must independently satisfy every original acceptance criterion.

Six fresh processes execute current, candidate, ORT, ORT, candidate, current on
AMD CPU2, with monitoring on CPU0. Every process runs all twenty clips with one
warmup and three measured passes: 480 requests and 360 scored measurements.
The timer includes frontend, neural inference, decoding and independently owned
results. All clocks, setup costs and process accounting remain retained.

For each engine, duplicate-process means must have max/min <=1.10 for the corpus
and <=1.20 for each clip: all 63 repeatability controls must pass. Candidate must
improve corpus latency by at least 3%, with no clip over 5% slower. Exact fractions
and equal process weights are unchanged. No trimming, adaptive warmup or unchanged
retry. Candidate/ORT <=1.05 is reported separately from admission.

Managed public results must match the freshly qualified references exactly.
Native ORT1.29.0 uses CPUExecutionProvider, one intra/inter-op thread, sequential
execution, full optimization and no spinning. All original transcript/token,
input and ownership checks remain. No profiler or numerical override is enabled.
The older baseline supplies immutable native assets/runtime; both managed
products and consumers come from the current model qualification.

Run with `C:/Python313/python.exe -X utf8 -B` from the repository root:

    tests/parakeet/prepared-recurrence-app-amd/consumer_scope.py
    tests/parakeet/prepared-recurrence-app-amd/test_admission.py
    tests/parakeet/prepared-recurrence-app-amd/run.py prepare
    tests/parakeet/prepared-recurrence-app-amd/run.py stage
    tests/parakeet/prepared-recurrence-app-amd/run.py launch
    tests/parakeet/prepared-recurrence-app-amd/run.py observe
    tests/parakeet/prepared-recurrence-app-amd/run.py collect
    tests/parakeet/prepared-recurrence-app-amd/audit.py

Local and VM namespaces are respectively
`artifacts/parakeet-prepared-recurrence-app-amd-20260924` and
`/dev/shm/lokad-parakeet-prepared-recurrence-app-20260924`. Refuse reuse; collect
only terminal PID/birth owners and never observe a closed namespace. Bounds stay
12 GiB available/3 GiB tmpfs preflight, owned RSS<12 GiB, at least 1 GiB available
memory/tmpfs, output<=1 GiB/job, artifacts<=2 GiB/campaign, 3600s/job, four hours
total and monitoring gaps below ten seconds. An admitted result still requires
shared/e5/Pyannote, root/suite and package qualification before release changes.
