# One complete Parakeet decision for contiguous padding

Compare the qualified current root Core f3992f40 / Data a8e0b583 with the existing
copy dispatcher Core a74acb17 / Data be954dc4 and ORT 1.29. Full model correctness
must close successfully before preparation. Reuse the compiled AudioBenchmark,
native consumer, fixtures and model weights; no build or download is needed.

The exact Parakeet attribution finds 2.880430 seconds of managed encoder padding
versus 0.050175 seconds in the retained ORT profile. ORT's source copies contiguous
blocks, and all 2,880 measured Pad calls show no new live arena space. The candidate
removes generic per-element index mapping but keeps independently owned output
arrays. The predicted benefit is lower encoder padding time, bounded at roughly
5.3% of complete transcription time. These dated diagnostic clocks are not scores.

The component screen remains rejected: six individual repeatability controls
failed. All twelve regression gates and the aggregate primary repeatability
control passed. The separate memory probe establishes fault-associated blocks
and switch-associated long calls, with other variability unresolved. PLAN explicitly
permits this full-application decision while retaining the component limitation.
No padding variant, compiler flag, warmup search or screen retry is introduced.

The decision is whether this one source-motivated change improves complete
transcription under the original application gates. Six fresh CPU2 processes run
current, candidate, ORT, ORT, candidate, current. Each performs one warmup and three
measured passes over 20 clips / 213.265 seconds of audio: 480 requests total.
CPU0 monitors. No other VM workload, cleanup or compilation overlaps timing.

Require all 63 repeatability controls: corpus process max/min <=1.10 and each
clip <=1.20 for each engine. Require >=3% corpus gain over the actual current root
and no clip >5% slower. Keep every clock and complete public result. Report the
separate <=1.05 candidate/ORT target; padding alone is not expected to reach it.
A failure remains a failure; no unchanged retry, trimming or threshold adjustment.

Reuse the original worker, request checks, exact-clock scorer, auditor and its
five admission tests byte-for-byte from direct-depthwise-app-amd. Only prerequisite
binding and transport namespaces change. Keep 11GiB available RAM / 3GiB tmpfs
preflight, 12GiB owned RSS, 1GiB remaining RAM/tmpfs, 1GiB output/job, 2GiB stage,
3,600 seconds/job, four hours overall and monitor gaps below ten seconds.

From this directory use C:/Python313/python.exe -X utf8 -B with consumer_scope.py,
test_admission.py and test_prerequisites.py. After model closure, use run.py
prepare, stage and launch once each; observe the same owner until terminal, then
collect and audit.py once. Keep audit stdout outside the artifact directory.
Tools freeze at preparation. Local namespace artifacts/parakeet-pad-current-app-amd-20260926;
remote /dev/shm/lokad-parakeet-pad-current-app-20260926.

Passing this application comparison does not promote the product. Shared/Pyannote
correctness and regressions, graph regressions, actual root/package tests and
public ownership remain required before integration or BENCHMARK.md changes.
