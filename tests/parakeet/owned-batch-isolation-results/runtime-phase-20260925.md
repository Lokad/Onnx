# Retained runtime evidence: the short prefix ends before optimization settles

This read-only census uses the four already closed 6,000-call diagnostic traces.
No inference, new capture, score or threshold change was performed. It examines
all `Method/LoadVerbose` events whose namespace starts with `Lokad.Onnx.`; this
is a namespace filter, not a claim that the entire CLR is quiescent.

| Process | Core | Prefix ends (s) | Last optimized tier load (s) | Namespace loads | Loads after call 3,000 |
| --- | --- | ---: | ---: | ---: | ---: |
| a | f95a13c5 | 9.032351 | 17.531748 | 1580 | 0 |
| b | 40260aef | 9.045954 | 16.550295 | 1592 | 0 |
| c | 40260aef | 9.449322 | 18.557111 | 1592 | 0 |
| d | f95a13c5 | 9.073390 | 16.512494 | 1580 | 0 |

The original 600-warmup/180-measurement prefix ends around nine seconds.
Optimized graph and tensor method versions continue arriving well after it.
This is direct evidence that the prefix precedes later optimization in these
instrumented release and depthwise-parent processes. It does not establish the
runtime state or cause of the new e07a4518 repeatability failure.

The next bounded question is whether the exact release/e07a4518 pair exhibits
the same phase separation. Reuse the already compiled 6,000-call observer and
lossless exporter, with exact product/manifest identities, original call labels,
four release/candidate/candidate/release processes and unchanged runtime settings.
Inspect the complete call timeline and method-load events. This is diagnostic;
do not score its late tail, tune a warmup count by retries, or change the failed
campaign. A future measurement correction needs a prospective protocol and
fresh qualification with all existing numerical and performance limits.

[Bound census and source hashes](runtime-phase-20260925.json),
[earlier complete trace report](../../benchmarks/e5-direct-tier-results/report-20260925.md),
[new failed comparison](graphs-20260925.md).
