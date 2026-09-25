# The exact candidate also outlives the short-e5 timing window

All four exact-product observations place the final optimized versions of
both matrix dispatchers after call 779. The original timing labels therefore
observe an intermediate runtime state in this diagnostic. This confirms the
lead on the actual candidate; it does not explain the older 7.77% regression.

| Process | Product | Original 600–779 labels, ms | Matrix final tier, s / call | Batched final tier, s / call |
|---|---|---:|---|---|
| a | current | 10.499982 | 13.910434 / 1247 | 14.138745 / 1264 |
| b | candidate | 10.612061 | 13.989722 / 1254 | 14.149561 / 1265 |
| c | candidate | 12.401426 | 14.918854 / between 1235–1236 | 15.147722 / 1252 |
| d | current | 10.911380 | 13.947110 / 1268 | 14.164338 / 1284 |

Seconds are elapsed from the first call. The two candidate processes differ
by 16.86% in their original measurement labels; the release pair differs
by 3.92%. These are descriptive instrumented clocks, not new performance gates.
The full fixed 30-call table shows latency changes around and after final
tier loads. Other product methods continue loading afterward; simultaneous
compilation and runtime changes prevent attributing the change to one method.

All ten consecutive 600-call groups below provide a compact view of the
entire observation. This post-capture grouping omits no call and is not a
replacement for the predeclared 30-call blocks or a selected tail score.

| Calls | Release A, ms | Candidate B, ms | Candidate C, ms | Release D, ms |
|---|---:|---:|---:|---:|
| 0–599 | 11.720308 | 11.721599 | 11.844882 | 11.674549 |
| 600–1199 | 10.302610 | 10.372458 | 12.058939 | 10.396212 |
| 1200–1799 | 7.783371 | 7.582804 | 7.659458 | 7.338043 |
| 1800–2399 | 5.865933 | 6.040032 | 5.816279 | 5.837172 |
| 2400–2999 | 5.841301 | 6.040312 | 5.784985 | 5.830135 |
| 3000–3599 | 5.857344 | 6.033859 | 5.777107 | 5.821764 |
| 3600–4199 | 5.835553 | 6.053543 | 5.773580 | 5.818998 |
| 4200–4799 | 5.827013 | 6.033242 | 5.777514 | 5.807212 |
| 4800–5399 | 5.835348 | 6.033974 | 5.758229 | 5.771227 |
| 5400–5999 | 5.832776 | 6.038696 | 5.730350 | 5.777477 |

The runtime also reports different final native-code sizes across fresh
processes using the same product and method. This is a concrete remaining
lead; the event stream does not contain those machine instructions.

| Process | Matrix dispatcher final bytes | Batched dispatcher final bytes |
|---|---:|---:|
| a | 2560 | 10100 |
| b | 3589 | 11735 |
| c | 3703 | 11769 |
| d | 3703 | 10175 |

Do not infer identical optimized code from a shared tier label. Conversely,
code size alone neither identifies the differing instructions nor proves
their latency cost. Source-level kernel changes, a JIT-setting sweep or
a favorable rerun would be premature. The next bounded diagnosis should
inspect the actual emitted code for these two exact method identities and
their call routes, preserving runtime settings and the complete call history.
First establish whether retained evidence already contains usable code bytes;
prepare a capture only if that read-only inspection establishes a specific gap.

All original graph failures remain failed. Release admission, the original
warmup policy and BENCHMARK.md stay unchanged. Separate shared/Pyannote
correctness qualification remains available while that diagnosis proceeds.

[Complete report and every 30-call block](report-20260925.md),
[exact interpretation inputs and summaries](interpretation-20260925.json).

Closure: `e61941d9308e33b688f222665a0f79dd985f87b85d56123554a47202d8c531bc`.
