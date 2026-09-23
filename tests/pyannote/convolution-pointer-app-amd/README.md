# M28 complete application comparison with Microsoft ORT

Compare current Core `208371f6` / Data `b9358370`, M28 Core `e776cec2` /
Data `0c55b650`, and Microsoft ONNX Runtime 1.29.0. All normal product,
complete Pyannote, Parakeet and shared/e5 prerequisites must be closed and
passed for those exact binaries before preparation. Do not repeat completed
unchanged tests. All consumers, native scripts and model assets stay unchanged.

Fresh native public conformance covers four Pyannote fixtures and twenty
Parakeet clips. Both 600-second meetings and the 30-second recovery request
must preserve complete timelines, centroid tolerances, read-only inputs and
owned results. Then run six fresh timing processes in the fixed order:
selected, candidate, ORT, ORT, candidate, selected. Each performs one warmup
and three measured passes over the 30-second dialogue and three ten-second
crops: 96 total requests, 24 warmups, 72 measurements. Retain every clock.

All twelve repeatability controls remain mandatory: process max/min <=1.10
for dialogue and <=1.20 per crop for every engine. Candidate/current must be
<=0.97 for dialogue and <=1.05 for each crop. These are the same application
rules fixed before M28 component timing. The separate ORT parity target is
<=1.05. No sample exclusions, trimming, calibration, runtime overrides or
unchanged timing retries. The admission function is structurally identical
to M22; only descriptive engine labels change. Numerical/public/meeting
checkers are byte-identical to the qualified predecessor.

Complete public-request timing includes frontend, inference and owned
results; model setup is recorded separately. Native ORT uses one intra/inter
thread, sequential execution, full graph optimization and no spinning.
CPU2 affinity precedes startup; CPU0 monitoring preserves all snapshots and
foreign CPU accounting, including its known short-lived-process limitation.
Keep 12 GiB available/3 GiB tmpfs preflight, 12 GiB RSS cap, 3,600 seconds per
job, four hours per campaign, 1 GiB live available/tmpfs and output bounds,
and 2 GiB artifacts. No other benchmark worker may overlap.

From repository root use `C:/Python313/python.exe -X utf8 -B` with `run.py`
commands `prepare`, `stage`, `launch`, then `observe`. Only `collect` terminal
owners, then run `audit.py`. Never overwrite failed evidence or observe a
closed campaign. Admission precedes root integration and normal-root proof.
Only this complete campaign may refresh application/ORT results in BENCHMARK.md.
