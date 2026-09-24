# Original Parakeet application comparison for slice materialization

Run only after model closure `60232ced` and the matched complete-pair profile
passes its prospective >=80% reduction prediction. Use selected Core `672e5f30`
and candidate `bafdb006`, both with original Data `065b7a7f` and original
AudioBenchmark `7eca033a`. No product or consumer build, profiler or numerical
override is part of this comparison.

Six fresh CPU2 processes run current, candidate, ORT, ORT, candidate, current.
Each runs all twenty clips with one warmup and three measured passes: 480
requests, 360 measurements. The original complete-transcription boundary,
request checks, native runner, exact statistics and performance gates stay
unchanged; `consumer_scope.py` verifies the retained code. Full native/public
qualification already covers both instruction modes. This lane measures the
normal instruction mode used by the selected release.

All 63 repeatability controls must pass: duplicate-process means max/min <=1.10
for corpus and <=1.20 for each clip, for every engine. Require >=3% corpus gain
and no clip >5% slower. Keep all clocks, no trimming or unchanged retry.
Candidate/ORT <=1.05 remains a separate unmet target until measured.

The available-memory preflight is prospectively 11 GiB, matching the successful
full-model and profile stages. This is the only resource-protocol change from
the prepared-recurrence application lane. Its full-application peak was 9.91 GB
and the current model qualification peaks at 9.38 GB. All other bounds stay:
3 GiB tmpfs before each job, RSS<12 GiB, >=1 GiB available memory/tmpfs,
output<=1 GiB/job, artifacts<=2 GiB, 3600s/job, four hours total, monitoring gaps
below ten seconds. Only one VM workload runs. Every PID/birth owner must be
terminal before collection; preserve any failure and never observe a closed stage.

From the repository root, use `C:/Python313/python.exe -X utf8 -B` with
`consumer_scope.py`, `test_admission.py`, `run.py prepare`, `stage`, `launch`,
`observe`, `collect`, then `audit.py` in this directory. Local artifacts are
`artifacts/parakeet-slice-materialization-app-amd-20260924`; VM namespace is
`/dev/shm/lokad-parakeet-slice-materialization-app-20260924`.

The profile stage establishes causal attribution only; its clocks never enter
this score. The independent prepared-recurrence candidate is not included.
An admitted result still needs shared/e5/Pyannote, root/suite and package
qualification before changing the release or BENCHMARK.md.
