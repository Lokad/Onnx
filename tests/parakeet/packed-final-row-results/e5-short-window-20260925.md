# Short-e5 timing changes inside the M78 measurement window

This is an inspection of the completed graph run, with no new inference or
rescoring. The [original report](graphs-20260925.md) remains failed: candidate
11.727438 ms versus release 10.882376 ms, a 7.7654% regression against the 5%
limit. All 24 process-repeatability controls pass. Neither that pass nor this
inspection establishes that execution had settled throughout measurement.

Each cell below is the arithmetic mean, in milliseconds, of 30 consecutive
measured calls from the [complete original clocks](graphs-clocks-20260925.csv).
Indices are zero-based; calls 0–599 retain their original warmup labels, and
600–779 retain their original measured labels. These six blocks cover every
measured call in every short-e5 timing process. Nothing is trimmed or relabeled.

| Process | 600–629 | 630–659 | 660–689 | 690–719 | 720–749 | 750–779 |
|---|---:|---:|---:|---:|---:|---:|
| Release A | 11.318750 | 11.600447 | 13.192671 | 8.856827 | 8.877901 | 8.826268 |
| Candidate A | 10.970947 | 11.111673 | 11.410401 | 10.846940 | 10.955413 | 13.343617 |
| ORT A | 5.928243 | 5.938504 | 5.932167 | 5.917882 | 5.923907 | 5.943210 |
| ORT B | 5.929630 | 5.934523 | 5.918583 | 5.918684 | 5.899661 | 5.933422 |
| Candidate B | 11.624474 | 12.343290 | 11.736911 | 11.722995 | 13.927524 | 10.735065 |
| Release B | 10.691131 | 10.699649 | 11.538327 | 10.645608 | 10.659963 | 13.680968 |

Release A changes substantially inside the measured window. The other managed
processes show changes at different positions; the two native processes remain
near 5.9 ms. This narrows a future diagnosis toward changing runtime execution
state, but timing alone does not identify compilation, garbage collection or a
kernel as the cause. The table is descriptive, not a statistical causal test.

The [earlier runtime observation](../../benchmarks/e5-repeatability-diagnostic-results/report-20260925.md)
recorded product code loading inside short-e5 measurement for release and M73.
That is relevant prior evidence, but its instrumented processes and candidate
are different from this uninstrumented M78 run. Their event timestamps cannot
be attached to these calls or used to assert the cause of this failure.

Keep the failed release verdict and all original limits. Do not infer a stable
tail score, choose a favorable subset, increase warmups from this table alone,
or repeat the unchanged run to seek a pass. When the e5 release blocker is
addressed, the missing evidence is runtime state for this exact short-input
candidate. The immediate VM work remains the direct Parakeet comparison, then
its exact-candidate phase profile against the retained ORT evidence.

Reproduction: read the CSV with `csv.DictReader`, select the six process names
beginning `timing-e5-8tok-`, require exactly indices 0–779 with `warmup=True`
exactly for indices below 600, and calculate each table cell as
`1000 * sum(Fraction(ticks, frequency)) / 30` over its stated inclusive index
interval. Round only the displayed value to six decimal places.

Graph closure: `0b82805aa6de28287c9bc9242416b3abd5c58b76fdff57a0416208f29f3707f2`.
This note adds no admission, timing-policy change or product change.
