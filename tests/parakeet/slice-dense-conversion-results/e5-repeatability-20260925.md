# Retained e5 repeatability failures: clock review

The two failed controls remain failed. This read-only analysis uses all
9,360 retained calls from the twelve processes for e5-8tok and e5-512tok.
Fixed blocks contain 30 consecutive calls, including every warmup; no
clock is trimmed and no replacement performance score is produced.

Candidate process means differ by 11.980% at eight tokens and 13.229%
at 512 tokens, exceeding the original 10% limit. All eight regression
comparisons passed, but that does not override the repeatability failure.

At eight tokens, the selected processes and candidate B have late changes
within the measured window. At 512 tokens, candidate B is consistently
faster than candidate A across all six measured blocks and continues to
decline. The existing clocks cannot identify JIT compilation, collection
or scheduling as the cause. They have call indices and elapsed ticks,
but no absolute call boundaries to join to runtime events.

A previous [30-token runtime diagnostic](../../benchmarks/e5-runtime-diagnostic-results/report-20260924.md)
observed late compilation, but used different products and inputs.
It supports investigating runtime state; it does not explain these
failures or justify another warmup increase without observation.

If release work proceeds, the next bounded diagnostic should preserve
both products, both failing inputs, all 780 calls and their order, and
observe runtime compilation, collection pauses and process CPU at call
boundaries outside the existing timer. Include both products twice.
Treat observer timings as diagnostic only and preserve the failed score.
Do not overlap the active Parakeet application comparison.

All times below are milliseconds. Columns are the six consecutive
30-call blocks of the original 180-call measured window.

| Case | Process | Full mean | 600–629 | 630–659 | 660–689 | 690–719 | 720–749 | 750–779 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| e5-8tok | current-a | 11.969 | 11.956 | 12.889 | 11.950 | 11.984 | 14.018 | 9.016 |
| e5-8tok | candidate-a | 10.002 | 9.851 | 9.931 | 9.946 | 9.891 | 9.894 | 10.502 |
| e5-8tok | ort-a | 6.245 | 6.265 | 6.239 | 6.225 | 6.268 | 6.246 | 6.228 |
| e5-8tok | ort-b | 6.566 | 6.557 | 6.565 | 6.573 | 6.574 | 6.550 | 6.576 |
| e5-8tok | candidate-b | 11.201 | 10.528 | 11.743 | 10.788 | 10.785 | 14.222 | 9.138 |
| e5-8tok | current-b | 11.515 | 12.971 | 11.930 | 12.136 | 13.610 | 9.256 | 9.190 |
| e5-512tok | current-a | 398.341 | 403.991 | 389.502 | 403.686 | 410.525 | 393.065 | 389.274 |
| e5-512tok | candidate-a | 394.882 | 397.563 | 408.937 | 401.679 | 390.810 | 377.198 | 393.107 |
| e5-512tok | ort-a | 308.215 | 309.417 | 309.416 | 310.423 | 309.359 | 304.230 | 306.444 |
| e5-512tok | ort-b | 300.061 | 298.697 | 298.219 | 299.936 | 299.483 | 302.897 | 301.134 |
| e5-512tok | candidate-b | 348.746 | 357.585 | 352.996 | 353.227 | 348.239 | 338.964 | 341.463 |
| e5-512tok | current-b | 395.651 | 393.761 | 404.429 | 405.829 | 390.582 | 380.652 | 398.653 |

[All fixed blocks, including warmup](e5-repeatability-20260925.csv),
[source pins, original controls and process summaries](e5-repeatability-20260925.json).

Original closure: `e04e3a6a7434afd4af6bda1901c934004fc263db82a52f453f4321ca3f7a9fbd`. No VM workload or product edit was made.
