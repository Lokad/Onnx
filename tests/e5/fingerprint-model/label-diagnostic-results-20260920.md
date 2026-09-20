# Label placement in identical e5 work — September 20, 2026

Balancing labels within each six-cycle block reduces the sensitivity of the
control screen to their placement in the retained chronology. This supports
testing a distinct locally balanced design. It does **not** change the failed
[original A/A verdict](aa-results-20260920.md), qualify another protocol, or
measure the enabled fingerprint cache.

All three original measured labels used identical settings on one prepared
graph. This diagnostic preserves all 16,704 measured Execute/request durations,
including GC tails, and changes only their anonymous labels. Equal-sized batches
are reduced to arithmetic means; the original per-call estimator independently
reproduces all original control ratios and screens to within 1e-12.

Each design uses all 1,000 fixed seed assignments. No seed is selected for its
outcome. All designs preserve 24 cycles, four copies of every three-role
permutation and eight batches per role at each position in each worker.

| Cases passing both boundaries | Global 24-cycle shuffle | Balanced six-cycle blocks | Mirrored twelve-cycle blocks |
|---|---:|---:|---:|
| All five cases | 644 / 1,000 | 821 / 1,000 | 730 / 1,000 |
| 8 tokens | 934 | 939 | 878 |
| 30 tokens | 966 | 974 | 973 |
| 30 padded to 128 | 815 | 933 | 908 |
| 128 tokens | 947 | 964 | 955 |
| 512 tokens | 930 | 988 | 987 |

Every assignment passes the aggregate-ratio limit. The position-contrast screen
is the main source of rejection: by itself it passes all cases in 649, 826 and
739 assignments, respectively. The existing limits remain unchanged: 0.5% for
aggregate ratios, 1% for workers and individual positions, and 1.01 for the
maximum divided by minimum position ratio.

The mirrored schedule places each shuffled six-permutation sequence beside its
reverse. It gives each role the same mean cycle index at each position within
a block. All 20,000 generated worker schedules satisfy this property, and an
independent synthetic linear-drift check passes. Nevertheless, its conditional
acceptance is lower than the six-cycle design, particularly at eight tokens.
Linear balance alone therefore does not explain or remove all observed
sensitivity. This analysis does not identify drift, GC or any other single cause.

These counts describe relabelings of **the same retained timing traces**. They
are not independent executions, confidence intervals or probabilities of
passing on a future host. The enabled-cache comparison remains unrun. A future
design still needs prospectively fixed sampling and fresh matching A/A evidence;
using a lucky seed or treating this diagnostic as qualification would be invalid.

The [source](label_diagnostic.py) fixes 1,000 assignments per design, starting
with seed `20260920 + 100*visit + caseIndex + 10000*replicate`, using the original
LCG/Fisher–Yates arithmetic. Full original reconstruction and every simulated
ratio/screen remain under `artifacts/e5-fingerprint-label-diagnostic-20260920`.
Its closed receipt is
`4f44cb0d2011c45887f8576a7daec7de85a13e069a296371bf594c7500d09400`.
The original experiment remains closed at
`97e3ecd9230f3aa169da51805be565bc789569c3864f88cad0c2ecea727e3ce9`.

Independent verification rechecks every diagnostic file and uses the original
per-call estimator on eighteen complete remappings, comparing every case,
boundary, pair, worker and position. It also checks a synthetic linear drift.
The verification is retained in
`artifacts/e5-fingerprint-label-check-20260920.json`. No model inference, VM load,
source/default change or benchmark-scoreboard change occurred.
