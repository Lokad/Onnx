# Complete Parakeet comparison of one ORT-derived sigmoid

This prospective comparison is conditional on the same candidate passing complete
model correctness. It reuses current Core 8bb22038 / Data d02dbf55 and rational
Core 946ddfb6 / Data dbe95936, with the original compiled application consumers,
twenty clips, model files and native runtime. No build or download is needed.

Executed ORT SiLU instructions use fixed rational logistic arithmetic. Current
managed sigmoid takes 2.458878 seconds in the complete-model diagnostic. The
fixed candidate keeps vector width, allocation and the graph unchanged; it changes
this arithmetic alone. Its ordinary operator screen reduced weighted time
74.992572%, narrowly missing its 75% threshold. Thirteen repeatability controls
and four fallback regression checks also failed. That screen remains rejected.

The subsequent diagnostic preserves every clock and emitted method version. It
finds equal allocation minima, matching scalar instruction counts and collection
increments associated with large empty/scalar spikes. The double slowdown remains
unresolved. These observations do not rescore the screen or establish universal
fallback performance equivalence. This application decision cannot resolve double
performance and cannot authorize release integration by itself.

The falsifiable application hypothesis is at least 3% complete-transcription gain
without a clip regressing more than 5%. The affected time bounds the opportunity
at about 4.9% of current transcription time; the isolated observed reduction would
save about 1.84 seconds, or 3.7%, only if it transfers to the application. Changes
in allocation, collection and surrounding execution can invalidate that estimate.
There is no arithmetic, ISA, compiler-option or warmup search.

The original six fresh CPU2 processes run current, candidate, ORT, ORT, candidate,
current. Each has one warmup and three measured passes over twenty clips totaling
213.265 seconds of audio. All 480 requests and 360 measured clocks are retained.
CPU0 monitors. No other VM workload, cleanup or compilation overlaps timing.

Keep all 63 original repeatability controls: corpus process max/min <=1.10 and
per-clip max/min <=1.20 for every engine. Require the original >=3% corpus gain
and no clip >5% slower, all exact public results, ownership and resource checks.
Report the separate candidate/ORT <=1.05 parity target. No clock trimming,
unchanged retry, threshold adjustment or retrospective admission is allowed.

The original request worker, validator, exact-clock scorer, auditor and five
admission tests remain byte-identical to direct-depthwise-app-amd. Only prerequisite
binding and transport namespaces change. Full-model prerequisite floats retain
the original native scaled-error bound 1e-4, and are compared to current with the
same bound; integers, decoder decisions and complete public results remain exact.

Resource limits remain 11GiB available RAM / 3GiB tmpfs at preflight, 12GiB owned
RSS, 1GiB remaining RAM/tmpfs, 1GiB output per job, 2GiB stage, 3,600 seconds per
job, four hours overall and monitor gaps below ten seconds.

From this directory use C:/Python313/python.exe -X utf8 -B with consumer_scope.py,
test_admission.py and test_prerequisites.py. After model closure and the prospective
PLAN decision, use run.py prepare, stage and launch once each; observe the same
owners until terminal, then collect and audit.py once. Keep audit stdout outside
the artifact. Tools freeze at preparation. Local namespace:
artifacts/parakeet-rational-sigmoid-app-amd-20260927; VM namespace:
/dev/shm/lokad-parakeet-rational-sigmoid-app-20260927.

A passing result establishes only this workload's application benefit. Shared/e5,
Pyannote and graph regression checks, actual-root/package tests, and an explicit
decision about the retained fallback limitation remain required before integrating
source or updating BENCHMARK.md. A failure stays failed and ends this candidate's
application trial.
